#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行失败结构化 + 单 Agent 回灌（Agent 闭环第②步）
====================================================
执行结果（PASS/FAIL/ERROR/SKIPPED）不再只是展示：失败用例被结构化后回灌给
LLM（单 Agent 一次调用完成"诊断分类 + 修正用例"），产出改进后的用例集。

【失败三分类】（对齐开发计划 7.7.12 的 B 修复对象分类）
  - case_defect   用例缺陷（URL 错/参数错/缺 URL/预期写错）→ 修用例（安全，默认自动）
  - potential_bug 疑似被测系统缺陷（如 5xx）→ 保留原用例作为 bug 复现，只报告不掩盖
  - env           环境缺陷（服务不可达/鉴权未配置）→ 提示先修环境，用例本身不动

【规则先行，LLM 兜底】结构化分类先用确定性规则（状态码/verdict 特征），
LLM 在此基础上做语义诊断——规则给信号，LLM 给结论，避免纯凭 LLM 猜。

【铁律】判定为 potential_bug 的用例，禁止把预期改成"错误行为"来让它变绿——
测试的价值恰是发现 bug；这类用例原样保留并在诊断里标注"建议报告 bug"。
"""
import json
import re
from typing import Any, Callable

from .assertions import INTENT_LABEL
from ..core.generate import (_extract_json_array, _default_llm_invoke,
                              _parse_llm_json, _validate_test_cases)

# 失败类别
CAT_CASE = "case_defect"       # 用例缺陷 → 修用例
CAT_BUG = "potential_bug"      # 疑似被测系统缺陷 → 保留复现 + 报告
CAT_ENV = "env"                # 环境缺陷 → 修环境，用例不动

_CAT_LABEL = {CAT_CASE: "用例缺陷", CAT_BUG: "疑似被测系统缺陷", CAT_ENV: "环境缺陷"}
# 对外别名（webapp 执行结果展示用）
CAT_LABEL = _CAT_LABEL

# 判定为"失败"的 verdict 集合（WARN 无法自动断言，不进回灌，避免噪音）
_FAILED_VERDICTS = {"FAIL", "ERROR", "SKIPPED"}


def classify_failure(result: dict[str, Any]) -> tuple[str, str]:
    """确定性规则初分类：返回 (类别, 提示语)。LLM 诊断在此基础上做语义修正。"""
    verdict = result.get("verdict", "")
    status = result.get("status") or 0
    reason = str(result.get("reason", ""))
    if verdict == "ERROR":
        blob = reason + str(result.get("error", "")) + str(result.get("response_snippet") or "")
        if "latin-1" in blob or "codec can't encode" in blob or "非 ASCII" in blob:
            return CAT_CASE, ("请求头含非 ASCII 字符（如中文占位 token）——HTTP 头仅允许 ASCII，"
                              "请求根本没发出去；属于用例缺陷（占位符未替换为合法格式凭证），"
                              "不是环境问题。修正：用格式合法的凭证占位并注明前置登录步骤")
        return CAT_ENV, "请求本身失败（连接拒绝/超时等），疑似被测服务不可达，先确认服务与地址"
    if verdict == "SKIPPED":
        return CAT_CASE, "用例不可执行（缺 URL 或非 HTTP 方法），需修正 request 构造"
    if verdict == "FAIL":
        # 未解析占位符优先识别（7.7.47 实测）：资源占位符 <item_id> 没被替换成真实 ID
        # 就发出 → 服务端 422（非法 UUID）——这不是参数构造错误，是资源准备失败
        import re as _re
        req_blob = json.dumps(result.get("request") or {}, ensure_ascii=False)
        if _re.search(r"<(item|resource|existing)[_-]?id>", req_blob, _re.I):
            return CAT_CASE, ("请求中仍含未替换的资源占位符 <item_id>——资源工厂未能准备"
                              "真实资源 ID（需先有通过的成功创建类用例，从中提取资源 ID）。"
                              "修法：确认创建类用例通过后再执行本用例，或用真实存在的资源 ID 替换")
        if "Bearer <valid_jwt>" in req_blob and not result.get("token_applied", True):
            return CAT_CASE, ("本次执行没有生效的测试凭证（未配置，或服务重启后会话凭证已清空）——"
                              "<valid_jwt> 占位不会被替换，需鉴权用例全部 401/403。"
                              "这不是用例错误：请配置账号密码后重新执行")
        # 5xx 优先于意图断言（7.7.49 实测）：实际 500 是服务端异常，不是"预期码写错"——
        # LLM 预期 422 的负向用例（如 skip=-1）打出 500，是真缺陷（potential_bug），
        # 修预期为 500 等于掩盖 bug
        if 500 <= status < 600:
            return CAT_BUG, (f"实际 {status}：服务端内部错误——参数/输入触发了服务端未处理异常。"
                             "若用例本意是测校验拒绝（预期 4xx），这里暴露的是真实缺陷："
                             "服务端缺少参数约束（如负数/越界值未校验），应报告而非修改用例预期")
        # 意图断言证据优先（7.7.45）：意图不符的 FAIL，正确预期就在 acceptable 里
        a = result.get("assertion") or {}
        if a.get("mode") == "intent" and a.get("acceptable"):
            return CAT_CASE, (f"意图断言不符: 用例意图「{INTENT_LABEL.get(a.get('intent'), a.get('intent'))}」"
                              f"可接受状态 {a['acceptable']}，实际 {status}——预期状态码写错，"
                              f"修预期为 {a['acceptable']} 中符合业务语义的值即可（服务端行为本身正常）")
        desc = str(result.get("description") or "")
        if any(kw in desc for kw in ("故障注入", "模拟数据库", "数据库异常", "模拟异常",
                                     "模拟服务", "内部状态", "mock 数据库")):
            return CAT_CASE, ("该用例依赖故障注入/内部状态模拟能力——平台从外部发 HTTP 请求，"
                              "无法控制服务内部，实际发送的是普通请求（必然 200）。"
                              "此类场景不可自动化，应剔除出用例集或改为外部可观测的等价场景")
        snippet = str(result.get("response_snippet") or "")
        if status == 400 and "already exists" in snippet:
            return CAT_CASE, ("测试数据重复：该资源的唯一键（如 email）已存在——上一轮执行已经创建过。"
                              "正向创建类用例的唯一键值必须每轮唯一：email 用 '前缀.<random_uuid>@example.com' "
                              "占位（执行时自动替换为新随机值），或用不存在的值；这不是服务端缺陷")
        if status in (400, 422) and "json_invalid" in snippet:
            return CAT_CASE, (f"实际 {status}：请求体不是合法 JSON（type=json_invalid）——"
                              "body 构造错误或发送格式与 Content-Type 不符，先核对 body 与请求头")
        if status == 404:
            return CAT_CASE, "实际 404：URL 可能不存在/拼错/路由前缀错误，对照接口契约核对"
        if status in (400, 422):
            return CAT_CASE, (f"实际 {status}：参数校验失败，body/query 构造可能与接口契约不符；"
                              "也可能是预期状态码写错（校验失败常为 422 而非 400，修预期即可）")
        if status == 200 and any(kw in desc for kw in ("空白字符", "仅空格", "空白值", "全空格")):
            return CAT_CASE, ("服务端接受了空白字符串——pydantic 的 min_length 只查长度、不 strip 空白。"
                              "若业务不允许空白值，这是一个值得报告的服务端质量瑕疵（空白数据入库）；"
                              "预期应修为 200，并在描述/报告中标注该发现")
        if status == 403:
            req = result.get("request") or {}
            hdrs = (req.get("headers") or {}) if isinstance(req, dict) else {}
            has_auth = any(str(k).lower() == "authorization" for k in hdrs)
            if has_auth:
                return CAT_CASE, ("实际 403：已携带凭证但被服务端拒绝。若用例故意携带无效/伪造令牌"
                                  "（负向用例），预期应修为 403——不少服务（如 full-stack-fastapi-template）"
                                  "把'凭证无效'返回 403 而非 401；若携带的是真实凭证仍 403，才需检查凭证有效性/权限")
            return CAT_ENV, ("实际 403：未携带凭证被拒。需真实凭证——执行前配置测试凭证（登录路径+账号密码），"
                             "用例里的 Bearer <valid_jwt> 占位符会被自动替换")
        if status == 401:
            return CAT_ENV, ("实际 401：未携带凭证（服务端未认证）。①需真实凭证——执行前配置测试凭证，"
                             "用例里的 Bearer <valid_jwt> 占位符会被自动替换；"
                             "②若该用例本意就是测试未认证访问（负向用例），则属预期正确场景，"
                             "检查实际响应是否与预期结果描述一致")
        if 500 <= status < 600:
            return CAT_BUG, "实际 5xx：服务端内部错误，疑似被测系统缺陷；保留用例作为复现，勿改预期"
        return CAT_CASE, f"实际 {status} 与预期不符：核对预期状态码是否写错（注意校验失败常为 422 而非 400）"
    return CAT_CASE, reason or "与预期不符"


def structure_failures(results: list[dict[str, Any]]) -> dict[str, Any]:
    """把执行结果里的失败条目结构化，并产出聚合特征提示。

    Returns:
        {
          "failures": [结构化失败条目],
          "stats": {"total", "failed", "by_verdict", "by_category"},
          "aggregate_hints": [全局特征提示],
        }
    """
    failures: list[dict[str, Any]] = []
    by_verdict: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for r in results:
        verdict = r.get("verdict", "")
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        if verdict not in _FAILED_VERDICTS:
            continue
        cat, hint = classify_failure(r)
        by_category[cat] = by_category.get(cat, 0) + 1
        failures.append({
            "entry": r.get("entry", ""),
            "description": r.get("description", ""),
            "verdict": verdict,
            "method": r.get("method", ""),
            "url": r.get("url", ""),
            "request": r.get("request"),
            "expected_status": r.get("expected_status"),
            "actual_status": r.get("status"),
            "reason": r.get("reason", ""),
            "response_snippet": (r.get("response_snippet") or "")[:300],
            "error": r.get("error", ""),
            "category": cat,
            "category_label": _CAT_LABEL.get(cat, cat),
            "hint": hint,
        })

    # 聚合特征：按失败整体形态给系统性提示（7.7.12"诊断分类是闭环灵魂"）
    hints: list[str] = []
    if failures:
        statuses = [f["actual_status"] for f in failures if f["actual_status"]]
        n = len(failures)
        # 注意：ERROR 条目无状态码，只有 statuses 覆盖全部失败时才说"全部"
        if len(statuses) == n and all(s == 404 for s in statuses):
            hints.append(f"全部 {n} 条失败均为 404 → 疑似 base_url 或路由前缀整体错误，"
                         "请对照接口契约里的真实 URL 修正")
        if all(f["verdict"] == "ERROR" for f in failures):
            hints.append("全部失败为连接错误 → 被测服务大概率未启动或地址不可达，先修环境")
        if len(statuses) == n and all(s in (401, 403) for s in statuses):
            hints.append("全部失败为 401/403 → 鉴权未配置（无有效 token），属于环境准备问题")
        if len(statuses) == n and all(500 <= s < 600 for s in statuses):
            hints.append("全部失败为 5xx → 大面积服务端错误，疑似被测系统缺陷，请重点报告")
    return {"failures": failures, "stats": {"total": len(results), "failed": len(failures),
                                            "by_verdict": by_verdict, "by_category": by_category},
            "aggregate_hints": hints}


def _render_case(c: dict[str, Any], idx: int) -> str:
    """渲染一条原用例（含其执行结果，若有）。"""
    exec_render = ""
    if c.get("_exec"):
        e = c["_exec"]
        exec_render = (f"\n    执行结果: {e.get('verdict')} (实际状态码 {e.get('status')})"
                       f"\n    失败原因: {e.get('reason', '')}"
                       f"\n    实际响应片段: {e.get('response_snippet', '')[:200]}")
    return (f"[{idx}] {c.get('description', '')}\n"
            f"    request: {c.get('request')}\n"
            f"    预期状态码: {c.get('expected_status', '无')}\n"
            f"    预期结果: {c.get('expected_results', [])}"
            f"{exec_render}")


def build_feedback_prompt(feature: dict[str, Any], cases: list[dict[str, Any]],
                          structured: dict[str, Any], case_count: int = 10) -> str:
    """构建单 Agent 回灌 prompt：功能点档案 + 原用例（带执行结果）+ 结构化失败 → 诊断 + 修正用例集。"""
    entry = feature.get("entry", "?")
    signature = feature.get("signature") or entry
    docstring = feature.get("docstring") or "（无）"
    iface = feature.get("interface")
    ifaces = feature.get("interfaces") or ([iface] if iface else [])
    iface_render = "\n".join(
        f"  URL: {i.get('url')} | Method: {', '.join(i.get('methods', []))}" for i in ifaces
    ) or "  （无接口信息）"
    control_render = "\n".join(
        f"  L{c.get('line', '?')}: {c.get('kind', '?')} {c.get('cond', '') or ''}"
        for c in (feature.get("control_sites") or [])
    ) or "  （无）"

    # PASS 用例降级为摘要清单（token 优化）：LLM 不需要改它们，只需知道"已覆盖哪些场景"以避免
    # 补充新用例时重复。仅当存在带执行结果的用例（_exec）时才分组；纯函数等无执行结果的调用全量全文
    has_exec = any(c.get("_exec") for c in cases)
    if has_exec:
        failed_render, passed_render = [], []
        for c in cases:
            if c.get("_exec"):
                failed_render.append(_render_case(c, len(failed_render) + 1))
            else:
                passed_render.append(
                    f"  - {c.get('description', '')}（预期 {c.get('expected_status', '无')}）")
        cases_render = ("\n".join(failed_render) or "（无）")
        if passed_render:
            cases_render += ("\n\n【已通过的用例（场景清单，仅供参照）】共 "
                             f"{len(passed_render)} 条——这些场景已验证通过，禁止改动、禁止生成重复场景：\n"
                             + "\n".join(passed_render))
    else:
        cases_render = "\n".join(_render_case(c, i) for i, c in enumerate(cases, 1)) or "（无）"
    failures_render = []
    for i, f in enumerate(structured.get("failures") or [], 1):
        failures_render.append(
            f"[{i}] {f['description']}\n"
            f"    {f['method']} {f['url']}\n"
            f"    预期: {f['expected_status']} | 实际: {f['actual_status']} | {f['verdict']}\n"
            f"    规则初分类: {f['category_label']} — {f['hint']}\n"
            f"    失败原因: {f['reason']}\n"
            f"    实际响应片段: {f['response_snippet'] or f['error'] or '（无）'}"
        )
    hints_render = "\n".join(f"  - {h}" for h in (structured.get("aggregate_hints") or [])) or "  （无）"
    # 已知缺陷复现用例单列（prompt 层防护）：这些用例禁止修改，LLM 不得判为 case_defect
    repro = structured.get("bug_repro") or []
    repro_render = "\n".join(
        f"  - {b.get('description', '')}（预期 {b.get('expected_status', '无')}）"
        f"｜证据: {b.get('evidence', '')[:100]}"
        for b in repro) or "  （无）"
    # 轮次间记忆（多轮迭代）：前几轮已尝试未成功的修正，附修正后用例快照与下轮执行结果
    prev = structured.get("previous_attempts") or []
    prev_lines = []
    for p in prev:
        fr = p.get("fixed_request") or {}
        line = (f"  - [{p.get('entry', '')}] {p.get('description', '')}\n"
                f"      上轮修正：{p.get('cause', '')}（{p.get('action', '')}，类别 {p.get('category', '')}）\n"
                f"      修正后用例：预期 {p.get('fixed_expected', '无')} | "
                f"{fr.get('method', '')} {fr.get('url', '')} | body≈{fr.get('body', '无')}")
        if p.get("next_result"):
            line += f"\n      修正后执行结果：仍失败 → {p['next_result']}"
        prev_lines.append(line)
    prev_render = "\n".join(prev_lines) or "  （无）"

    return f"""你是一位资深测试诊断专家。下面这批测试用例已经对真实服务执行过，部分失败。
请你作为"诊断修复 Agent"完成一次综合处理：**逐条诊断失败原因，并输出修正后的完整用例集**。

【失败三分类】（诊断必须归入其中一类）
- case_defect 用例缺陷：URL 错 / 参数构造错 / 缺 URL / 预期状态码写错 → 直接修正用例
- potential_bug 疑似被测系统缺陷：如 5xx 服务端错误 → 【不要修正该用例】，原样保留它作为 bug 复现依据，
  在 diagnosis 里标注 action=建议报告bug。严禁为了让用例通过而把预期改成错误行为——测试的价值恰是发现 bug
- env 环境缺陷：服务不可达 / 鉴权未配置 → 用例不动，在 diagnosis 里标注 action=需先修复环境

【功能点档案】
入口符号: {entry}
签名: {signature}
业务说明: {docstring}
控制流骨架:
{control_render}
接口信息（URL/method 以此为权威）:
{iface_render}

【原测试用例（含执行结果）】共 {len(cases)} 条
{cases_render}

【结构化失败清单】共 {len(structured.get('failures') or [])} 条
{chr(10).join(failures_render) or '  （无）'}

【聚合特征提示】（失败的整体形态，往往指向系统性原因）
{hints_render}

【已知缺陷复现用例——禁止修改】（此前已被确认为被测系统缺陷，这些用例是复现依据）
{repro_render}
对以上用例：禁止修改其 request/expected_status 的任何字段，必须在 cases 中原样保留；
它们的持续失败是预期行为（缺陷尚在），只能归入 potential_bug 并在报告中标注，严禁判为 case_defect 改预期。

【前几轮已尝试但未成功的修正】（多轮迭代记忆——每条含上轮修正内容与其执行结果，已被证伪；请勿重复同样的修改，在证据基础上换思路）
{prev_render}

【任务】
1. 逐条诊断失败用例：归类（三分类之一）+ 失败原因 + 处理动作
2. case_defect：依据接口信息修正 request/expected_status，修正后的用例放入 cases
3. potential_bug / env：对应用例原样保留进 cases（不得修改其预期），只在 diagnosis 里说明
4. 通过（PASS）的用例原样保留，不要改动
5. 输出修正后的完整用例集（原 {len(cases)} 条的修正版），目标 {case_count} 条左右；禁止为凑数生成重复用例

【输出格式】（严格 JSON 对象，不要添加额外文字）
{{
  "diagnosis": [
    {{"description": "失败用例描述", "category": "case_defect|potential_bug|env",
      "cause": "失败原因分析", "action": "已修正用例|建议报告bug|需先修复环境"}}
  ],
  "cases": [
    {{
      "description": "用例描述",
      "request": {{"method": "GET", "url": "/path", "headers": {{}}, "query": {{}}, "body": {{}}, "files": {{}}}},
      "expected_status": 200,
      "test_steps": ["1. 步骤1"],
      "expected_results": ["1. 结果1"]
    }}
  ]
}}

【请求头与凭证约束】（实测高频错误，务必遵守）
- headers 的值只能含 ASCII 字符（HTTP 协议限制），【严禁】出现中文占位符如 "Bearer 超级用户token"——
  这会导致请求根本发不出去（编码错误）
- 需要有效凭证的场景：在 test_steps 第一步写明前置步骤（如"1. 前置：调用登录接口获取有效 token"），
  header 值【必须】使用统一占位符 "Authorization": "Bearer <valid_jwt>"——执行时平台会自动替换为真实 token；
  禁止使用其他占位写法或编造 token 内容
- 【负向凭证占位】测试"无效/过期凭证被拒绝"的用例，用 "Bearer <invalid_jwt>" /
  "Bearer <expired_jwt>"（平台原样发送，服务端返回 401/403 正是预期）；
  严禁给这类负向用例写 <valid_jwt>——会被替换为有效 token，用例意图被破坏
- 【鉴权顺序】服务通常先校验鉴权再校验参数：无 token / 无效 token 时请求先得到 401，
  到不了参数校验/业务逻辑——所以"聚焦参数校验/业务分支"的用例必须先解决凭证问题，
  而不是"去掉 token 来聚焦"（那是方向性错误）

【资源占位符协议】（update/delete 类用例的测试数据由平台自动准备）
- 需要"真实存在的资源 ID"（如更新/删除已存在物品）：url 或 body 里写
  "<item_id>"（也接受 <existing_uuid>）——平台会自动预创建资源并注入真实 ID，禁止写死任何具体 UUID
- 需要"格式合法但不存在的 ID"（验证 404 路径）：写 "<random_uuid>"
- 严禁把 {{id}}、{{nonexistent_id}} 等模板占位符直接留在 url 里（会被当作字面量导致 422 解析失败）

【用例可自动化约束】（实测高频错误，务必遵守）
- 【严禁】生成依赖故障注入/数据库 mock/内部状态篡改的用例（如"模拟数据库插入失败验证 500"）——
  测试从外部发起 HTTP 请求，无法控制服务内部，这类用例永远无法按预期执行；
  想测错误处理只能通过外部可见手段（非法参数、越界值、不存在的资源 ID）
- 超长字符串边界用例：request 里写 "<超长字符串：...>" 占位即可，执行时平台会自动展开为
  真实超长字符串（占位里可注明提示长度，如"（如 256 字符）"）

【严禁】字符串值内部出现未转义的英文引号 "（会破坏 JSON 导致解析失败）；
如需表达引用请使用中文引号 “ ”。request 里的 URL/参数名必须来自上面的接口信息，不得编造。
【边界值表示】超长字符串边界用简短占位描述，禁止真的重复几百字。
"""


def _parse_diagnosis_cases(response: str) -> tuple[list[dict[str, Any]], Any]:
    """解析回灌响应（diagnosis + cases）。cases 提取失败时退回纯数组模式
    （LLM 可能只输出用例数组）。供单次回灌与工具化回灌共用。"""
    from .review import _extract_json_object

    diagnosis: list[dict[str, Any]] = []
    obj_text = _extract_json_object(response)
    raw_cases: Any = None
    if obj_text:
        try:
            obj = _parse_llm_json(obj_text)
            if isinstance(obj, dict):
                diag = obj.get("diagnosis")
                if isinstance(diag, list):
                    for d in diag:
                        if isinstance(d, dict) and d.get("description"):
                            diagnosis.append({
                                "description": str(d.get("description", "")),
                                "category": str(d.get("category", "")),
                                "category_label": _CAT_LABEL.get(str(d.get("category", "")),
                                                                 str(d.get("category", ""))),
                                "cause": str(d.get("cause", "")),
                                "action": str(d.get("action", "")),
                            })
                raw_cases = obj.get("cases")
        except Exception:
            raw_cases = None
    if raw_cases is None:
        # 退回数组模式：整个响应当用例数组解析
        array_text = _extract_json_array(response)
        if not array_text:
            raise ValueError("无法从 LLM 响应中提取诊断/用例 JSON")
        raw_cases = _parse_llm_json(array_text)
    return diagnosis, raw_cases


def feedback_refine_cases(feature: dict[str, Any], cases: list[dict[str, Any]],
                          structured: dict[str, Any], provider: str = "deepseek",
                          case_count: int = 10,
                          llm_invoke: Callable[[str, str], str] | None = None,
                          **llm_kwargs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """单 Agent 回灌：诊断失败 + 产出修正后用例集。

    Args:
        feature: 功能点档案 dict
        cases: 该功能点的原用例（dict 列表）
        structured: structure_failures() 中属于该 entry 的部分（含 failures/stats/aggregate_hints）
        provider: LLM 提供商
        case_count: 目标用例条数
        llm_invoke: 可注入的 LLM 回调（默认走独立 llm_client）

    Returns:
        (diagnosis 列表, 修正后用例 dict 列表)
    """
    prompt = build_feedback_prompt(feature, cases, structured, case_count=case_count)
    invoke = llm_invoke or (lambda p, prov: _default_llm_invoke(p, prov, **llm_kwargs))
    response = invoke(prompt, provider)
    diagnosis, raw_cases = _parse_diagnosis_cases(response)

    # 复用用例校验（api 形态：request 必须是合法 HTTP 请求）
    validated = _validate_test_cases(raw_cases, case_count if isinstance(case_count, int) else 15)
    if not validated:
        raise ValueError("LLM 未返回任何合法测试用例")
    refined_dicts = [c.to_dict() for c in validated]
    return diagnosis, refined_dicts


# ---------------- 工具化回灌（B Agent 第一步：诊断时读被测源码，7.7.51） ----------------
_FEEDBACK_READ_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_source",
        "description": "读取被测项目源码片段（带行号）。当失败原因存疑（尤其 5xx、"
                       "需要判断是用例错还是服务端缺陷）时，先读相关代码验证假设再下结论。",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string", "description": "相对被测项目根的文件路径"},
            "start": {"type": "integer", "description": "起始行（默认 1）"},
            "end": {"type": "integer", "description": "结束行（默认 400）"}},
            "required": ["file"]}}},
]

_FEEDBACK_TOOLS_APPENDIX = """

【可用工具】read_source(file, start, end)——读取被测项目源码片段（路径相对项目根）。
工作方式：若你需要查看代码来确认失败原因，输出 tool_calls 调用 read_source，
平台会返回代码片段；确认结论后，最终一轮输出【输出格式】要求的 JSON（不要带 tool_calls）。
诊断明确、无需看代码时可直接输出最终 JSON，不必调用工具。
【提示】功能点档案中的控制点/签名含源文件路径，可从那里开始读。"""


def feedback_refine_cases_with_tools(feature: dict[str, Any], cases: list[dict[str, Any]],
                                     structured: dict[str, Any], provider: str = "deepseek",
                                     case_count: int = 10,
                                     project_root: str = "",
                                     max_turns: int = 6,
                                     llm_client: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """工具化回灌：与 feedback_refine_cases 同一任务，但 Agent 可调用 read_source
    查看被测源码后再下结论（7.7.12 设想的 B Agent 工具化第一小节）。

    与 FixAgent 复用同一个 read_source 实现（read_source_snippet，含路径沙箱）；
    只读不写——回灌 Agent 没有 edit 权限，改的是用例不是被测代码。
    LLM 不调用工具时行为退化为单次调用（多一层系统提示的开销）。

    Returns:
        (diagnosis 列表, 修正后用例 dict 列表)
    """
    from ..llm.client import LLMClient
    from .fixer import read_source_snippet, FixAgent

    prompt = build_feedback_prompt(feature, cases, structured, case_count=case_count)
    messages = [
        {"role": "system", "content": (
            "你是测试用例诊断与修正专家。基于给出的执行失败证据修正测试用例。"
            "你可以用 read_source 工具查看被测项目源码来验证失败原因——"
            "但禁止修改被测代码，你的产出只有诊断结论和修正后的用例。"
            "最终必须输出【输出格式】要求的 JSON。")},
        {"role": "user", "content": prompt + _FEEDBACK_TOOLS_APPENDIX},
    ]
    client = llm_client or LLMClient(provider=provider)
    response = ""
    for _turn in range(1, max_turns + 1):
        message = client.chat_raw(messages, temperature=0.2,
                                  tools=_FEEDBACK_READ_TOOLS)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            response = message.get("content") or ""
            break
        for tc in tool_calls:
            try:
                fn = json.loads(tc["function"]["arguments"] or "{}")
                result = read_source_snippet(
                    project_root, str(fn.get("file", "")),
                    int(fn.get("start") or 1), int(fn.get("end") or 400),
                    deny_parts=FixAgent._DENY_PARTS)
            except Exception as e:
                result = f"错误：工具调用失败 {e}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": result})
    if not response:
        raise ValueError(f"工具化回灌在 {max_turns} 轮内未得到最终诊断结果")

    diagnosis, raw_cases = _parse_diagnosis_cases(response)
    validated = _validate_test_cases(raw_cases, case_count if isinstance(case_count, int) else 15)
    if not validated:
        raise ValueError("LLM 未返回任何合法测试用例")
    return diagnosis, [c.to_dict() for c in validated]


if __name__ == "__main__":
    print(__doc__)
