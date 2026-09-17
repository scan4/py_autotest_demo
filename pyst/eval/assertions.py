#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
断言可靠性增强（问题1 第一阶段）：意图断言 + 响应体结构校验
============================================================
核心洞察：expected_status 是 LLM 猜的（实测常写错 400/422），但用例描述与
预期结果里的"测试意图"（成功 / 校验拒绝 / 鉴权拦截 / 资源不存在）是稳定的。
断言以**意图**为准，而不是以猜测的状态码为准：

  1. 预期码精确匹配                    → PASS（最强证据）
  2. 意图明确 且 实际 ∈ 意图可接受集    → PASS（语义一致；预期码偏差作为注记）
  3. 意图明确 且 实际 ∉ 可接受集       → FAIL（意图不符）
  4. 意图无法识别（探查型）            → 退回 expected_status 精确对比 / WARN

另附响应体结构检查（作为回灌证据，不改变 verdict）：
  - 4xx：FastAPI 错误结构（detail 字段存在）
  - 2xx：响应体 JSON 可解析性

意图从 description + expected_results 的关键词提取（词表按优先级排序，
见 _intent_of_text）——这几轮全部"预期码写错"的 FAIL（400/422 混淆、
401/403 语义）都可被意图断言自动正确归类。
"""

from __future__ import annotations

import json
from typing import Any

# ---------------- 意图分类 ----------------

INTENT_SUCCESS = "success"          # 期望 2xx：功能正常工作
INTENT_VALIDATION = "validation"    # 期望 4xx（400/413/415/422）：参数/格式校验拒绝
INTENT_AUTH = "auth"                # 期望 401/403：鉴权拦截
INTENT_NOT_FOUND = "not_found"      # 期望 404：资源不存在
INTENT_PROBE = "probe"              # 探查型（"验证是否..."）：无固定预期，退回精确对比
INTENT_NONE = "none"                # 意图不明确

INTENT_LABEL = {
    INTENT_SUCCESS: "功能正常",
    INTENT_VALIDATION: "校验拒绝",
    INTENT_AUTH: "鉴权拦截",
    INTENT_NOT_FOUND: "资源不存在",
    INTENT_PROBE: "探查行为",
    INTENT_NONE: "意图不明确",
}

# 每种意图可接受的状态码集合（基于 FastAPI 生态的真实行为：
# 校验失败 422、body 解析失败 400、媒体类型不支持 415）
_ACCEPTED: dict[str, set[int] | None] = {
    INTENT_SUCCESS: set(range(200, 300)),
    INTENT_VALIDATION: {400, 413, 415, 422},
    INTENT_AUTH: {401, 403},
    INTENT_NOT_FOUND: {404},
    INTENT_PROBE: None,
    INTENT_NONE: None,
}

# 意图关键词（小写化后子串匹配；按优先级顺序判定）
_AUTH_KW = ("未认证", "未鉴权", "不携带", "缺少认证", "缺少鉴权", "鉴权失败", "认证失败",
            "无效 token", "无效令牌", "无效 jwt", "伪造", "过期 token", "未提供 authorization",
            "缺少 authorization", "不携带认证", "无认证", "鉴权拦截", "未登录")
_NOT_FOUND_KW = ("不存在", "找不到")
_PROBE_KW = ("是否被", "是否接受", "是否触发", "是否仍", "是否忽略", "是否拒绝")
_SUCCESS_KW = ("成功创建", "成功返回", "成功删除", "成功路径", "可成功", "正常创建", "正常返回",
               "正常工作", "合法请求", "合法的请求体", "仍能", "仍正常", "可省略", "有效鉴权",
               "通过分支", "写入数据库", "数据持久化", "恰好达到", "恰好达到最小", "合法 uuid",
               "合法且完整", "持久化")
_VALIDATION_KW = ("校验", "缺少", "缺失", "为空", "空字符串", "空值", "空白", "非法",
                  "类型错误", "类型非法", "非 json", "格式错误", "非法 json", "解析失败",
                  "解析错误", "超长", "长度上限", "长度边界", "超出", "必填", "参数错误",
                  "值错误", "text/plain", "content-type", "无效格式", "边界值", "字段类型",
                  "类型校验", "非字符串", "传入数字")


def _intent_of_text(text: str) -> str:
    """从用例文本（description + expected_results 拼接）提取测试意图。"""
    t = (text or "").lower()
    if any(kw in t for kw in _AUTH_KW):
        return INTENT_AUTH
    if any(kw in t for kw in _NOT_FOUND_KW):
        return INTENT_NOT_FOUND
    if any(kw in t for kw in _PROBE_KW):
        return INTENT_PROBE
    if any(kw in t for kw in _SUCCESS_KW):
        return INTENT_SUCCESS
    if any(kw in t for kw in _VALIDATION_KW):
        return INTENT_VALIDATION
    return INTENT_NONE


def intent_of_case(case: dict[str, Any]) -> str:
    """从用例提取测试意图（description + expected_results 联合匹配）。"""
    parts = [str(case.get("description") or "")]
    parts += [str(x) for x in (case.get("expected_results") or []) if x]
    return _intent_of_text(" ".join(parts))


def acceptable_statuses(intent: str) -> set[int] | None:
    """意图对应的可接受状态码集合（probe/none 返回 None = 退回精确对比）。"""
    return _ACCEPTED.get(intent)


# ---------------- 响应体结构检查（不改变 verdict，作为回灌证据） ----------------

def _fastapi_error_structure(body: str) -> bool:
    """FastAPI 错误结构：detail 字段存在（校验错误时为 [{type,loc,msg}...] 列表）。"""
    try:
        d = json.loads(body)
    except Exception:
        return False
    return isinstance(d, dict) and "detail" in d


def response_body_checks(intent: str, status: int, body: str) -> list[str]:
    """响应体结构检查，返回检查结论列表（供回灌证据/前端展示）。"""
    checks: list[str] = []
    body = body or ""
    if 400 <= status < 500:
        checks.append("FastAPI 错误结构(detail) ✓" if _fastapi_error_structure(body)
                      else "⚠ 4xx 响应未见 detail 字段（非标准错误结构）")
    elif 200 <= status < 300 and body.strip():
        try:
            json.loads(body)
            checks.append("响应体 JSON 可解析 ✓")
        except Exception:
            checks.append("⚠ 2xx 响应体不是合法 JSON（疑似异常响应，如 HTML 错误页）")
    return checks


# ---------------- 意图断言主入口 ----------------

def assert_case(case: dict[str, Any], actual: int, body: str) -> dict[str, Any]:
    """对单条用例的响应做意图断言（纯函数，供 executor._assert 调用与测试）。

    Returns:
        {"verdict": PASS|FAIL|WARN, "reason": str, "mode": exact|intent|probe|fallback,
         "intent": str, "acceptable": list[int]|None, "checks": [str]}
    """
    exp = case.get("expected_status")
    intent = intent_of_case(case)
    acceptable = acceptable_statuses(intent)
    checks = response_body_checks(intent, actual, body)
    mode = "fallback"

    def _acc() -> str:
        return f"意图「{INTENT_LABEL[intent]}」可接受状态 {sorted(acceptable)}"

    if exp is not None and actual == exp:
        # 最强证据：精确匹配。但补一个语义矛盾检查（防"预期码错但碰巧相等"）
        if intent not in (INTENT_NONE, INTENT_PROBE) and acceptable and actual not in acceptable:
            checks.append(f"⚠ 语义矛盾: 用例意图{_acc()}，实际 {actual}——"
                          "预期码与意图同时可疑，请人工复核")
        return {"verdict": "PASS", "mode": "exact", "intent": intent,
                "acceptable": sorted(acceptable) if acceptable else None,
                "reason": f"实际状态码 {actual} == 预期 {exp}", "checks": checks}
    if intent != INTENT_PROBE and intent != INTENT_NONE and acceptable and actual in acceptable:
        deviation = (f"（预期码 {exp} 偏差——属预期码写错，实际行为符合用例意图）"
                     if exp is not None and exp != actual else "")
        return {"verdict": "PASS", "mode": "intent", "intent": intent,
                "acceptable": sorted(acceptable),
                "reason": f"语义一致: {_acc()}，实际 {actual}{deviation}", "checks": checks}
    if intent != INTENT_PROBE and intent != INTENT_NONE and acceptable:
        return {"verdict": "FAIL", "mode": "intent", "intent": intent,
                "acceptable": sorted(acceptable),
                "reason": (f"意图不符: {_acc()}，实际 {actual}"
                           + (f"；预期码 {exp} 也与实际不符" if exp is not None and exp != actual else "")),
                "checks": checks}
    if exp is not None:
        # probe/none：退回精确对比
        return {"verdict": "PASS" if actual == exp else "FAIL", "mode": "probe",
                "intent": intent, "acceptable": None,
                "reason": (f"探查/意图不明确用例，按精确对比: 实际 {actual} "
                           f"{'==' if actual == exp else '!='} 预期 {exp}"), "checks": checks}
    return {"verdict": "WARN", "mode": "fallback", "intent": intent, "acceptable": None,
            "reason": "无法自动断言（意图不明确且无预期状态码），请人工判断", "checks": checks}


if __name__ == "__main__":
    print(__doc__)
