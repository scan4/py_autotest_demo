#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 测试用例评审模块（对齐 TestBrain 的 test_case_reviewer）
============================================================
对生成的测试用例做 LLM 自动评审，输出：评分、优点、缺点、缺失场景、通过建议。

【评审维度】（对齐 TestBrain prompt_config.yaml）
- 完整性 / 清晰度 / 可执行性 / 覆盖率
- 是否覆盖所有功能点和边界条件 / 是否考虑异常情况 / 是否符合测试最佳实践

【输出 JSON】
{
  "score": 1-10,
  "strengths": [...],
  "weaknesses": [...],
  "suggestions": [...],
  "missing_scenarios": [...],
  "recommendation": "通过" | "不通过" | "修改后通过",
  "comments": "总体评价"
}
"""

from typing import Any, Callable

from ..core.generate import _extract_json_array, _default_llm_invoke


def build_review_prompt(feature: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    """构建评审 prompt：功能点档案 + 待评审用例 → 评审 JSON。"""
    entry = feature.get("entry", "?")
    signature = feature.get("signature") or entry
    docstring = feature.get("docstring") or "（无）"
    control_sites = feature.get("control_sites") or []
    control_render = "\n".join(
        f"  L{c.get('line', '?')}: {c.get('kind')} {c.get('cond', '') or ''}"
        for c in control_sites
    ) or "  （无控制流信息）"

    cases_render = []
    for i, c in enumerate(cases, 1):
        req = c.get("request")
        req_render = f", request={req}" if req else ""
        cases_render.append(
            f"[{i}] {c.get('description', '')}{req_render}\n"
            f"    步骤: {c.get('test_steps', [])}\n"
            f"    预期: {c.get('expected_results', [])}"
        )
    cases_text = "\n".join(cases_render)

    return f"""你是一位资深的测试专家，负责对生成的测试用例进行评审。

【被测功能点】
入口: {entry}
签名: {signature}
业务说明: {docstring}
控制流骨架（用于判断覆盖是否充分）:
{control_render}

【待评审的测试用例】
{cases_text}

【评审任务】
请从以下维度评审这些测试用例：
1. 完整性：是否覆盖了功能点的全部控制流分支和边界条件
2. 清晰度：描述、步骤、预期结果是否清晰无歧义
3. 可执行性：request 参数是否完整、能否直接执行
4. 覆盖率：是否覆盖正常路径、异常路径、边界值、空值
5. 是否符合测试最佳实践

【输出格式】（严格 JSON，不要添加额外文字）
{{
  "score": 1到10的整数,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["缺点1", "缺点2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "missing_scenarios": ["缺失的测试场景1", "缺失的测试场景2"],
  "recommendation": "通过"或"不通过"或"修改后通过",
  "comments": "总体评价"
}}
"""


def _extract_json_object(text: str) -> str:
    """从 LLM 响应中提取单个 JSON 对象文本（容忍前后废话）。

    与 _extract_json_array 类似，但提取 { ... } 对象（评审输出是对象而非数组）。
    用括号配平扫描，跳过字符串内的花括号。
    """
    if not text:
        return ""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return ""


def parse_review(text: str) -> dict[str, Any]:
    """从 LLM 响应解析评审 JSON，缺失字段用默认值兜底。"""
    default = {
        "score": 0,
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "missing_scenarios": [],
        "recommendation": "未通过",
        "comments": "评审解析失败",
    }
    json_text = _extract_json_object(text)
    if not json_text:
        return default
    import json
    try:
        data = json.loads(json_text)
    except Exception:
        return default
    if not isinstance(data, dict):
        return default
    default.update({k: v for k, v in data.items() if k in default})
    # 评分归一化
    try:
        default["score"] = max(0, min(10, int(default["score"])))
    except (TypeError, ValueError):
        default["score"] = 0
    return default


def review_test_cases(feature: dict[str, Any], cases: list[dict[str, Any]],
                      provider: str = "deepseek",
                      llm_invoke: Callable[[str, str], str] | None = None,
                      **kwargs) -> dict[str, Any]:
    """对一组测试用例执行 AI 评审。

    Args:
        feature: 功能点档案 dict
        cases: 测试用例 dict 列表（TestCase.to_dict()）
        provider: LLM 提供商
        llm_invoke: 可注入的 LLM 回调（默认走独立 llm_client）
        **kwargs: 透传给 LLM 的额外参数
    """
    prompt = build_review_prompt(feature, cases)
    invoke = llm_invoke or (lambda p, prov: _default_llm_invoke(p, prov, **kwargs))
    response = invoke(prompt, provider)
    return parse_review(response)


if __name__ == "__main__":
    print(__doc__)
