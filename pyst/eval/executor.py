#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实 HTTP 执行器（webapp 与 runner CLI 共用）
==============================================
把 runner.py 的执行逻辑抽成公共模块，两个入口复用：
  - runner.py（命令行：python runner.py cases.json --base http://...）
  - webapp /api/execute mode=ai（前端"AI 执行"按钮）

断言策略（现状，待问题1增强）：
  1. expected_status 精确比对（主断言）
  2. 无 expected_status 时关键词兜底（响应与预期特征匹配）
  3. 都无法判断 → WARN 人工
"""

from __future__ import annotations

import itertools
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

_VERDICT_MARKS = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIPPED": "⏭️", "ERROR": "💥"}


def verdict_mark(verdict: str) -> str:
    return _VERDICT_MARKS.get(verdict, "•")


def _classify_status_code(code: int) -> str:
    """HTTP 状态码粗分类标签。"""
    if 200 <= code < 300:
        return "成功"
    if 400 <= code < 500:
        return "客户端错误"
    if 500 <= code < 600:
        return "服务器错误"
    return "其他"


def _keyword_assert(resp_text: str, expected_text: str) -> bool | None:
    """关键词级断言：从 expected_results 里挑"成功/错误"特征词比对响应体。

    返回 True/False，无法判断返回 None。这是无 expected_status 时的兜底。
    """
    if not resp_text or not expected_text:
        return None
    if any(k in expected_text for k in ("成功", "返回", "attachment", "下载")):
        if any(k in resp_text for k in ("error", "错误", "失败", "404", "500")):
            return False
    if any(k in expected_text for k in ("错误", "失败", "不存在", "不能为空", "非法")):
        if any(k in resp_text for k in ("error", "错误", "失败", "not found", "不存在")):
            return True
        return None
    return None


def _assert(case: dict[str, Any], resp: requests.Response, result: dict[str, Any]) -> dict[str, Any]:
    """执行断言（7.7.45 意图断言增强）：expected_status 精确比对 + 意图语义断言 + 关键词兜底。

    判定优先级（pyst/eval/assertions.py）：
      1. 预期码精确匹配 → PASS（附语义矛盾检查）
      2. 意图明确且实际 ∈ 可接受集 → PASS（预期码偏差标注为"预期码写错"）
      3. 意图明确且实际 ∉ 可接受集 → FAIL（意图不符）
      4. 探查/意图不明确 → 退回精确对比；无预期码 → 关键词兜底 / WARN

    【遗留】安全加固（路径沙箱祖先判断/敏感拒绝清单）与意图断言均已落地，
    对应回归测试见 tests/（68 用例）；断言可靠性第二阶段（OpenAPI 响应 schema
    字段级断言）待 OpenAPI 探测扩展 responses 提取后实施。
    """
    from .assertions import assert_case
    exp_status = case.get("expected_status")
    if exp_status is not None:
        result["expected_status"] = exp_status
    a = assert_case(case, resp.status_code, resp.text or "")
    result["pass"] = {"PASS": True, "FAIL": False}.get(a["verdict"])
    result["verdict"] = a["verdict"]
    result["reason"] = a["reason"]
    result["assertion"] = {"mode": a["mode"], "intent": a["intent"],
                           "acceptable": a["acceptable"], "checks": a["checks"]}
    if a["verdict"] == "WARN" and exp_status is None:
        expected_text = " ".join(str(x) for x in (case.get("expected_results") or []))
        kw = _keyword_assert(resp.text or "", expected_text)
        if kw is True:
            result["pass"] = True
            result["verdict"] = "PASS"
            result["reason"] = f"响应与预期特征匹配（关键词兜底; 状态码分类: {_classify_status_code(resp.status_code)}）"
            result["assertion"]["mode"] = "fallback"
        elif kw is False:
            result["pass"] = False
            result["verdict"] = "FAIL"
            result["reason"] = "响应与预期特征冲突（关键词兜底）"
            result["assertion"]["mode"] = "fallback"
    return result


# 凭证占位符：回灌 prompt 约束 LLM 用 "Bearer <valid_jwt>" 表达"需要有效 token"，
# 执行时由平台用真实 token 替换（token 来自登录接口换发，见 fetch_login_token）
_TOKEN_PLACEHOLDER = "<valid_jwt>"
# token 缓存有效期（秒）：多数服务 access token 有效期远长于 50 分钟，缓存避免每条执行都登录
_TOKEN_TTL = 50 * 60


def fetch_login_token(base: str, login_path: str, username: str, password: str,
                      timeout: int = 30) -> str:
    """用账号密码调登录接口换 access_token（OAuth2 password 表单流，FastAPI 模板标准）。

    失败抛 ValueError，错误信息含 HTTP 状态/响应片段，便于用户修正凭证。
    """
    if not base or not login_path:
        raise ValueError("需要被测服务地址 base 和登录路径 login_path")
    # base 归一化：用户常填 "localhost:8100"（漏 http://）——与 execute_http_case
    # 的归一化行为保持一致，否则拼出的 URL 无协议，requests 报 No connection adapters
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    url = login_path if login_path.startswith("http") else (
        base.rstrip("/") + "/" + login_path.lstrip("/"))
    try:
        resp = requests.post(url, data={"username": username, "password": password},
                             timeout=timeout)
    except Exception as e:
        raise ValueError(f"登录接口不可达 {url}: {e}") from e
    if resp.status_code != 200:
        raise ValueError(f"登录失败: HTTP {resp.status_code} {resp.text[:120]}（尝试的账号: {username}——"
                         f"请核对账号密码是否正确、有无多余空格；被测服务凭证配置可查其 .env 的 FIRST_SUPERUSER*）")
    try:
        data = resp.json()
    except Exception as e:
        raise ValueError(f"登录响应不是 JSON: {resp.text[:120]}") from e
    tok = data.get("access_token")
    if not tok:
        raise ValueError(f"登录响应中无 access_token 字段: {str(data)[:120]}")
    return tok


# 负向用例描述特征：用例本意就是"不带凭证"测鉴权拦截，注入真实 token 会破坏其意图
_UNAUTH_DESC_KEYWORDS = ("未认证", "不携带", "无认证", "缺少认证", "无 token", "不带 token",
                         "缺 token", "未登录", "无令牌", "无效 token", "伪造", "过期 token")


def _is_token_placeholder(value: str) -> bool:
    """判断 Authorization 头的值是否为"token 占位"（归一化识别，不限写法）。

    实测 LLM 的占位写法会漂移：<valid_jwt>、<有效token>、<your_token_here>、
    "Bearer 超级用户token"……逐个枚举追不上。归一化特征（去掉 Bearer 前缀后）：
    尖括号包裹的占位串，或含任何非 ASCII 字符（中文占位必然非法，只能当占位）。
    真实 JWT/字面 token 是 ASCII 且无尖括号包裹，不会误判。
    """
    s = value.replace("Bearer", "").replace("bearer", "").strip()
    if not s:
        return False
    if s.startswith("<") and s.endswith(">"):
        return True
    return any(ord(ch) > 127 for ch in s)


def _apply_token(case: dict[str, Any], headers: dict[str, Any], token: str) -> dict[str, Any]:
    """token 应用（三级策略，返回新 dict 不改原用例）：

    1. Authorization 头为占位写法（任意变体）→ 归一化替换为真实 token
    2. 已有真实 Authorization 头（非占位）→ 原样保留（用例自带 token 表达其意图）
    3. 没有 Authorization 头 → 自动注入真实 token，但负向用例除外：
       预期状态码为 401/403，或描述含"未认证/不携带"等特征——这些用例本意就是
       测"无凭证被拦截"，注入会破坏意图（实测旧批用例多数压根没写 Authorization）
    """
    out: dict[str, Any] = {}
    has_auth = False
    for k, v in (headers or {}).items():
        sv = str(v)
        if k.lower() == "authorization":
            has_auth = True
            if token and _is_token_placeholder(sv):
                out[k] = f"Bearer {token}"      # 归一化替换（占位写法不限）
            else:
                out[k] = v
        elif isinstance(v, str) and _TOKEN_PLACEHOLDER in v:
            out[k] = v.replace(_TOKEN_PLACEHOLDER, token)
        else:
            out[k] = v
    if has_auth:
        return out
    if not token:
        return out
    desc = str(case.get("description") or "")
    exp = case.get("expected_status")
    negative = (isinstance(exp, int) and exp in (401, 403)) \
        or any(kw in desc for kw in _UNAUTH_DESC_KEYWORDS)
    if not negative:
        out["Authorization"] = f"Bearer {token}"
    return out


# 超长占位符默认展开长度：常见字段上限为 128/256/1024，5000 必然超过；无上限字段
# 展开后仍 200 → 判定失败 → 回灌修预期（真实暴露"服务端不限制"这一契约事实）
_LONG_PLACEHOLDER_DEFAULT_LEN = 5000


def _expand_long_value(v: Any) -> Any:
    """递归展开 body/query 里的"超长字符串"占位符为真实长度的字符串。

    prompt 约束 LLM 用占位描述表达长度边界（防打爆输出 token），但占位串只有
    几十字符——执行时长度校验必然通过，边界用例形同虚设（实测"超长 title"用例
    发了 40 字符得到 200）。占位特征：以 < 开头、以 > 结尾、含"超长"。

    展开语义按意图分两种（实测 LLM 会把"恰好等于上限"的用例也套"超长"占位）：
      - 含"恰好/等于" → 展开为恰好 N 字符（N 取占位里的提示长度，默认 5000）
      - 否则（"超过上限"意图）→ 展开为 N×2（必超，N 默认 5000 已必超常见上限）
    """
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("<") and s.endswith(">") and "超长" in s:
            import re as _re
            m = _re.search(r"(\d{2,5})\s*字符", s)
            if m:
                n = int(m.group(1))
                return "A" * (n if ("恰好" in s or "等于" in s) else n * 2)
            return "A" * _LONG_PLACEHOLDER_DEFAULT_LEN
        return v
    if isinstance(v, dict):
        return {k: _expand_long_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand_long_value(x) for x in v]
    return v


def execute_http_case(case: dict[str, Any], base: str = "", timeout: int = 30,
                      token: str = "") -> dict[str, Any]:
    """真实执行单条 HTTP 测试用例（发请求 + 断言），返回结构化结果。

    Args:
        case: 测试用例 dict（含 request/expected_status/expected_results/description，可带 _entry）
        base: 服务根地址（url 为相对路径时拼接）；url 已是绝对地址则原样使用
        timeout: 请求超时（秒）
        token: 真实访问 token（用于替换 headers 里的 <valid_jwt> 占位符；空=不替换）
    """
    req = case.get("request")
    if not isinstance(req, dict) or not req:
        return {"entry": case.get("_entry", ""), "description": case.get("description", "?"),
                "verdict": "SKIPPED", "status": 0,
                "reason": "缺少 request 字段（不可执行；纯函数用例请走 func 模式）"}

    method = str(req.get("method") or "GET").upper()
    # 防御层：非 HTTP 方法不执行（实测 LLM 会把非 HTTP 入口的函数调用伪装成
    # method=CALL / url=callable://... 的伪请求），发了也只会得到无意义的 400
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return {"entry": case.get("_entry", ""), "description": case.get("description", "?"),
                "verdict": "SKIPPED", "status": 0, "method": method, "request": req,
                "reason": f"非 HTTP 方法（{method}），疑似把函数调用伪装成了 HTTP 请求；"
                          f"非 HTTP 入口请改用「纯函数用例」模式生成（call_args）"}
    # 防御：LLM 可能输出 "url": null——键存在但值为 None，.get 的默认值不生效
    url = req.get("url") or ""
    if not url:
        return {"entry": case.get("_entry", ""), "description": case.get("description", "?"),
                "verdict": "SKIPPED", "status": 0, "method": method, "request": req,
                "reason": "用例缺少请求 URL（LLM 未生成或为 null），请重新生成或人工补全"}
    # base_url 归一化：用户常填 "localhost:8100"（漏 http://），requests 会报
    # "No connection adapters were found"——与 probe_openapi 的归一化行为保持一致
    if base and not base.startswith(("http://", "https://")):
        base = "http://" + base
    if not url.startswith("http"):
        url = (base.rstrip("/") + "/" + url.lstrip("/")) if base else url
    headers = _apply_token(case, req.get("headers") or {}, token)
    query = _expand_long_value(req.get("query") or {})
    body = _expand_long_value(req.get("body"))
    files = req.get("files")

    # 预检：HTTP 头仅允许 ASCII（latin-1）——实测 LLM 会把中文占位符当真值
    # 填进 Authorization（如 "Bearer 超级用户token"），requests 编码时抛
    # UnicodeEncodeError，请求根本没发出去，报错文本也让人摸不着头脑
    for hk, hv in headers.items():
        try:
            str(hv).encode("latin-1")
        except UnicodeEncodeError:
            return {"entry": case.get("_entry", ""), "description": case.get("description", "?"),
                    "verdict": "ERROR", "status": 0, "method": method, "url": url,
                    "request": req,
                    "reason": f"请求头 {hk} 的值含非 ASCII 字符（HTTP 头仅允许 ASCII）——"
                              f"疑似把中文占位符（如 Bearer 超级用户token）直接填进了头部；"
                              f"若未配置测试凭证，请在步骤4填写账号密码（平台自动登录换 token 并替换）；"
                              f"若已配置仍报错，请联系管理员排查登录接口"}

    try:
        file_objs = None
        if files:
            file_objs = {}
            for field, val in files.items():
                path = Path(val)
                file_objs[field] = (path.name, path.open("rb")) if path.exists() else (str(val), b"dummy")
        # body 发送格式以 Content-Type 为准（修复：之前 dict 一律走 data=（表单编码），
        # 而用例常带 Content-Type: application/json → 服务端拿 urlencoded 串解析 JSON
        # → 全部 json_invalid 422，失败是执行器造成的，不是用例/服务的错）
        ct = ""
        for k, v in headers.items():
            if k.lower() == "content-type":
                ct = str(v).lower()
                break
        is_form = "application/x-www-form-urlencoded" in ct
        resp = requests.request(
            method, url, headers=headers, params=query,
            data=body if (is_form or isinstance(body, str)) else None,
            json=body if isinstance(body, dict) and not is_form else None,
            files=file_objs, timeout=timeout,
        )
        result: dict[str, Any] = {
            "entry": case.get("_entry", ""),
            "description": case.get("description", "?"),
            "status": resp.status_code,
            "method": method,
            "url": url,
            "request": req,
            "response_snippet": (resp.text or "")[:300],
            "response_body": (resp.text or "")[:2000],   # 资源工厂提取 id 用（完整一点）
        }
        return _assert(case, resp, result)
    except Exception as e:
        return {"entry": case.get("_entry", ""), "description": case.get("description", "?"),
                "verdict": "ERROR", "status": 0, "method": method, "url": url,
                "request": req, "reason": str(e), "error": str(e)}


def execute_suite(cases: list[dict[str, Any]], base: str = "", token: str = "") \
        -> tuple[list[dict[str, Any]], dict[str, int]]:
    """批量执行用例（自动跳过非 HTTP 用例），返回 (结果列表, 统计)。

    token 非空时替换各用例 headers 里的 <valid_jwt> 占位符（每条执行同一真实 token）。
    """
    results: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    for c in cases:
        r = execute_http_case(c, base, token=token)
        results.append(r)
        v = r.get("verdict", "ERROR")
        stats[v] = stats.get(v, 0) + 1
    return results, stats


# ---------------- 资源占位符协议（update/delete 类用例的测试数据准备） ----------------
# 背景（7.7.45 实测）：update/delete 用例需要"真实存在的资源 ID"，LLM 无法预知——
# 第 1 轮写 {id} 占位（422 解析失败），第 2 轮写固定 UUID（404 不存在），回灌反复失败。
# 解法：占位符协议扩展 + 资源工厂——平台从已通过的成功创建用例派生资源，注入真实 ID。
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_RANDOM_UUID_PATS = [re.compile(p, re.I) for p in
                     (r"<(random|随机)[_-]?uuid>", r"<(不存在|invalid|非法)[_-]?的?uuid>")]
_EXISTING_UUID_PATS = [re.compile(p, re.I) for p in
                       (r"<(item|resource|existing)[_-]?id>", r"<(existing|有效|真实)[_-]?uuid>",
                        r"<item_id>")]


def _sub_uuid(v: Any, repl: Callable[[str], str], pats: list) -> Any:
    """递归替换字符串值中命中的占位符（repl 接收匹配片段，返回替换文本）。"""
    if isinstance(v, str):
        out = v
        for pat in pats:
            out = pat.sub(lambda m: repl(m.group(0)), out)
        return out
    if isinstance(v, dict):
        return {k: _sub_uuid(x, repl, pats) for k, x in v.items()}
    if isinstance(v, list):
        return [_sub_uuid(x, repl, pats) for x in v]
    return v


def _build_resource_pool(cases: list[dict[str, Any]], base: str, token: str,
                         limit: int = 3, log: Callable[[str], None] | None = None) -> list[str]:
    """资源工厂：重放"已通过的成功创建类用例"（POST + 描述含"创建"），从 2xx 响应
    中提取 UUID 格式的资源 ID，构建真实资源池。失败/不足时返回已收集部分。"""
    log = log or (lambda m: None)
    pool: list[str] = []
    factories = [c for c in cases
                 if (c.get("request") or {}).get("method") == "POST"
                 and "创建" in (c.get("description") or "")]
    for f in factories[:limit]:
        r = execute_http_case(f, base, token=token)
        if r.get("verdict") == "PASS" and 200 <= (r.get("status") or 0) < 300:
            try:
                d = json.loads(r.get("response_body") or r.get("response_snippet") or "{}")
            except Exception:
                d = {}
            uid = d.get("id") or d.get("uuid") or d.get("item_id")
            if uid and _UUID_RE.match(str(uid)):
                pool.append(str(uid))
                log(f"资源工厂：创建资源成功 id={uid}")
            if len(pool) >= limit:
                break
    return pool


def execute_suite_with_resources(cases: list[dict[str, Any]], base: str = "", token: str = "",
                                 log: Callable[[str], None] | None = None) \
        -> tuple[list[dict[str, Any]], dict[str, int]]:
    """两阶段执行（资源占位符协议）：

    阶段 1：静态替换 <random_uuid>/<不存在的uuid>（每条用例独立生成，用于 404 类用例）
    阶段 2：资源工厂预创建真实资源 → 替换 <item_id>/<existing_uuid>（轮转分配池中不同 id）
    无资源池时占位符原样保留（用例会失败，结果 hint 会指向资源准备问题）。
    """
    import itertools
    log = log or (lambda m: None)
    prepared: list[dict[str, Any]] = []
    for idx, c in enumerate(cases):
        c = dict(c)
        if isinstance(c.get("request"), dict):
            c["request"] = _sub_uuid(c["request"], lambda m: str(uuid.uuid4()),
                                     _RANDOM_UUID_PATS)
        c["_case_idx"] = idx
        prepared.append(c)
    pool = _build_resource_pool(prepared, base, token, log=log)
    if pool:
        ids = itertools.cycle(pool)
        for c in prepared:
            if isinstance(c.get("request"), dict):
                c["request"] = _sub_uuid(c["request"], lambda m: next(ids), _EXISTING_UUID_PATS)
    else:
        for c in prepared:
            if isinstance(c.get("request"), dict):
                has = any(p.search(json.dumps(c["request"], ensure_ascii=False))
                          for p in _EXISTING_UUID_PATS)
                if has:
                    c["_missing_resource"] = True
    results: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    for c in prepared:
        missing = c.pop("_missing_resource", False)
        r = execute_http_case(c, base, token=token)
        if missing and r.get("verdict") in ("FAIL", "ERROR"):
            r["reason"] = (r.get("reason", "") +
                           "｜资源占位符无可用资源：本轮成功创建类用例均未通过，"
                           "无法为 update/delete 用例准备真实资源 ID——请先确保创建类用例通过")
        results.append(r)
        v = r.get("verdict", "ERROR")
        stats[v] = stats.get(v, 0) + 1
    return results, stats
