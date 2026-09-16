#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRD 需求文档分析器（分层管线）
==================================
【两层架构，对齐 AST 代码分析】

  第一层：确定性预处理（不调 LLM，像 AST 分析代码）
      PRD 文档(.md/.pdf/.docx/.txt) → 功能点列表（标题 + 要点）
      解析 Markdown 标题树 / PDF 字体大小推断标题层级，产出结构化功能点。

  第二层：LLM 生成（只喂选中的功能点要点）
      用户选功能点 → LLM 基于该功能点要点生成测试用例
      上下文小、聚焦、幻觉少。

  旧 API analyze_prd（一次到位：PRD→测试点+场景）保留作为兼容/兜底。

【格式支持】
  - Markdown / TXT：直接读文本，解析标题层级（# ## ###）
  - DOCX：python-docx 提取段落，用样式/字号推断标题
  - PDF：PyMuPDF 提取文本，用字体大小/加粗推断标题层级
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.generate import _extract_json_array
from ..llm.client import chat_completion


# ============================================================
# 第一层：确定性预处理（Markdown 标题解析 → 功能点列表）
# ============================================================

@dataclass
class PrdFeature:
    """PRD 里识别出的一个功能点（由标题层级划分）"""
    title: str                    # 功能点标题
    level: int                    # 标题层级（#=1, ##=2, ...）
    section: str                  # 所属模块（父标题链）
    content: str                  # 该功能点的正文要点（去除噪音）
    raw_content: str = ""         # 原始正文（未清理）

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level,
            "section": self.section,
            "content": self.content,
        }


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _clean_content(text: str, max_len: int = 800) -> str:
    """清理正文：去掉图片、HTML、空行压缩，截断到 max_len 字符。"""
    if not text:
        return ""
    # 去掉图片 ![...](...) 和 HTML
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    # 压缩多余空行
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    cleaned = "\n".join(lines)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "\n...（已截断）"
    return cleaned


def parse_markdown_features(markdown: str) -> list[PrdFeature]:
    """解析 Markdown，按标题层级提取功能点列表。

    规则：
      - 一级/二级标题视为"模块"（section）
      - 三级及以下标题视为"功能点"（若文档没有三级标题，二级标题作为功能点）
      - 功能点的 content = 标题到下一个同级或更高级标题之间的正文
    """
    features: list[PrdFeature] = []
    lines = markdown.splitlines()

    # 先收集标题行及其位置
    headings: list[tuple[int, int, str]] = []   # (index, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        # 无标题：整个文档作为一个功能点
        return [PrdFeature("PRD 文档", 0, "", _clean_content(markdown), markdown)]

    # 决定"功能点"用哪一级标题
    # 用最深的标题级别作为功能点粒度（但至少是二级）
    max_level = max(lvl for _, lvl, _ in headings)
    feature_level = max_level if max_level >= 2 else 1

    for idx, (line_i, level, title) in enumerate(headings):
        if level != feature_level:
            continue
        # 取该标题到下一个同级标题之间的正文
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_i + 1:end])
        # 计算所属 section（父级标题）
        section = ""
        for j in range(idx - 1, -1, -1):
            if headings[j][1] < level:
                section = headings[j][2]
                break
        features.append(PrdFeature(
            title=title, level=level, section=section,
            content=_clean_content(body), raw_content=body,
        ))
    return features


def _infer_pdf_headings(text: str) -> str:
    """把 PDF 提取的文本按字体大小推断标题层级，转成 Markdown 风格的标题。

    输入是 PyMuPDF get_text('dict') 得到的 spans（含 size/font）。
    此函数接收预处理的文本块（见 read_document 的 pdf 分支），
    通过段落是否加粗/字号是否明显大于正文来标记标题。
    """
    return text


def extract_prd_features(markdown: str) -> list[PrdFeature]:
    """PRD 预处理入口：把文档内容转成功能点列表（确定性，不调 LLM）。"""
    return parse_markdown_features(markdown)


# ============================================================
# 第二层：LLM 生成测试用例（选中的功能点 → 用例）
# ============================================================

def build_prd_generate_prompt(feature: dict, section: str = "", case_count: int = 5) -> str:
    """构建"单个 PRD 功能点 → 测试用例"的 prompt。"""
    return f"""你是一位专业的软件测试专家。请根据下面的 PRD 功能点，设计可直接执行的测试用例。

【PRD 功能点】
所属模块: {section or "（无）"}
功能点标题: {feature.get('title', '')}
功能点要点:
{feature.get('content', '（无详细描述）')}

【任务】
请为该功能点设计 {case_count} 条测试用例，覆盖：
- 正常路径（核心功能能正常工作的场景）
- 边界条件（参数边界、数量边界、大小边界等）
- 异常情况（非法输入、资源不存在、系统异常等）
- 每个功能点描述里提到的具体校验规则

【输出格式】（严格 JSON 数组，不要添加多余文字，不要用 ```json``` 包裹）
[
  {{
    "description": "测试用例描述",
    "request": {{"method": "POST", "url": "", "headers": {{}}, "query": {{}}, "body": {{}}, "files": {{}}}},
    "expected_status": 200,
    "test_steps": ["1. 步骤1", "2. 步骤2"],
    "expected_results": ["1. 结果1", "2. 结果2"]
  }}
]

【说明】
- 若功能点没有明确的接口信息，request 中的 url 可留空，但 method/body 等能推断的尽量给出
- request 里的参数名应尽量基于功能点描述中的字段名
- expected_status 无法确定时可省略
【引号规则】所有字符串值用英文双引号包裹，字符串内部【严禁】使用英文引号（如需引用用中文引号 “ ”）。场景描述中的换行用 \\n 表示。
"""


def generate_prd_cases(feature: dict, provider: str = "deepseek", case_count: int = 5,
                       llm_invoke: Callable[[str, str], str] | None = None,
                       **kwargs) -> list[dict]:
    """为单个 PRD 功能点生成测试用例。

    Returns:
        校验后的测试用例 dict 列表
    """
    from ..core.generate import _validate_test_cases, _parse_llm_json

    prompt = build_prd_generate_prompt(feature, section=feature.get("section", ""), case_count=case_count)
    invoke = llm_invoke or (lambda p, prov: chat_completion(p, provider=prov, **kwargs))
    response = invoke(prompt, provider)

    json_text = _extract_json_array(response)
    if not json_text:
        return [{"description": "[生成失败] 无法从响应提取 JSON", "test_steps": [], "expected_results": []}]
    try:
        raw = _parse_llm_json(json_text)
    except Exception as e:
        return [{"description": f"[生成失败] JSON 解析错误: {e}", "test_steps": [], "expected_results": []}]
    cases = _validate_test_cases(raw, case_count)
    return [c.to_dict() for c in cases] if cases else [{"description": "[生成失败] 无合法用例", "test_steps": [], "expected_results": []}]


# ============================================================
# 文档读取（支持 md/pdf/docx/txt）
# ============================================================

def read_document(file_path: Path) -> tuple[str, str]:
    """读取 PRD 文档内容，返回 (文本, 格式)。

    支持 .md/.txt/.markdown/.docx/.pdf。
    PDF 用 PyMuPDF 提取文本 + 推断标题层级。
    """
    suffix = file_path.suffix.lower()
    if suffix in (".md", ".txt", ".markdown"):
        return file_path.read_text(encoding="utf-8", errors="ignore"), "markdown"
    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise RuntimeError("解析 .docx 需要 python-docx，请 pip install python-docx")
        doc = docx.Document(str(file_path))
        lines = []
        for p in doc.paragraphs:
            if not p.text.strip():
                continue
            # 用样式名推断标题层级
            style = (p.style.name or "").lower()
            if "heading" in style or "标题" in style:
                level = re.findall(r"\d+", style)
                hash_level = int(level[0]) if level else 1
                lines.append("#" * hash_level + " " + p.text.strip())
            else:
                lines.append(p.text.strip())
        return "\n".join(lines), "markdown"
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("解析 .pdf 需要 PyMuPDF，请 pip install PyMuPDF")
        doc = fitz.open(str(file_path))
        lines = []
        # 估算正文的平均字号，用于判断标题
        all_sizes = []
        page_spans = []
        for page in doc:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        all_sizes.append(span.get("size", 0))
                        page_spans.append(span)
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12
        # 重新按页组织输出，用字号/加粗推断标题
        output = []
        for page in doc:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    size = max(s.get("size", 0) for s in spans)
                    bold = any("bold" in s.get("font", "").lower() for s in spans)
                    # 标题判定：字号明显大于正文(>1.15倍) 或 加粗且较短
                    is_heading = (size > avg_size * 1.15) or (bold and len(text) < 50)
                    if is_heading:
                        # 根据字号相对正文的比例估计标题级别
                        ratio = size / avg_size if avg_size else 1
                        level = 1 if ratio > 1.5 else (2 if ratio > 1.25 else 3)
                        output.append("#" * level + " " + text)
                    else:
                        output.append(text)
        return "\n".join(output), "markdown"
    raise RuntimeError(f"不支持的文件类型: {suffix}（支持 .md/.txt/.docx/.pdf）")


# ============================================================
# 旧 API：一次到位（PRD → 测试点+场景），保留兼容
# ============================================================

def build_prd_prompt(markdown: str) -> str:
    """构建 PRD 分析 prompt：需求文本 → 测试点/场景 JSON。"""
    return f"""你是一位资深的软件测试分析师，负责从 PRD（产品需求文档）中提取测试点并设计测试场景。

【PRD 文档内容】
{markdown[:20000]}

【任务】
请分析上面的 PRD，提取出所有测试点（Test Point），并为每个测试点设计测试场景（Test Scenario）。

要求：
1. 测试点：按功能模块/业务逻辑拆分，覆盖 PRD 中的每个功能需求点
2. 每个测试点标注优先级（高/中/低）
3. 每个测试点下设计 2-5 个测试场景，场景要覆盖正常路径、边界条件、异常情况
4. 每个场景标注 test_type（功能测试/性能测试/兼容性测试/安全性测试/异常测试）
5. 场景的 description 要含编号步骤（1.2.3...）和预期结果

【输出格式】（严格 JSON，不要添加多余文字，不要用 ```json``` 包裹）
{{
  "test_points": [
    {{
      "id": "TP-001",
      "title": "测试点标题",
      "description": "测试点详细描述",
      "priority": "高",
      "scenarios": [
        {{
          "id": "TS-001-001",
          "title": "测试场景标题",
          "description": "1. 步骤1\\n2. 步骤2\\n预期结果：...",
          "test_type": "功能测试"
        }}
      ]
    }}
  ],
  "summary": {{
    "total_test_points": 0,
    "total_test_scenarios": 0,
    "high_priority_points": 0,
    "medium_priority_points": 0,
    "low_priority_points": 0
  }}
}}

【引号规则】所有字符串值用英文双引号包裹，字符串内部【严禁】使用英文引号（如需引用用中文引号 “ ” 或不加）。场景描述中的换行用 \\n 表示。
"""


def _extract_json_object(text: str) -> str:
    """从 LLM 响应提取 JSON 对象（括号配平，跳过字符串内花括号）。"""
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


def _normalize_priority(p: str) -> str:
    """优先级归一化。"""
    mapping = {"p0": "高", "p1": "高", "p2": "中", "p3": "低",
               "high": "高", "medium": "中", "low": "低",
               "High": "高", "Medium": "中", "Low": "低"}
    return mapping.get(str(p).strip().lower(), str(p) or "中")


def _validate_analysis_result(raw: Any) -> dict:
    """校验清洗分析结果。"""
    if not isinstance(raw, dict):
        raw = {}
    test_points = []
    for tp in raw.get("test_points") or []:
        if not isinstance(tp, dict):
            continue
        if not tp.get("title"):
            continue
        scenarios = []
        for ts in tp.get("scenarios") or []:
            if isinstance(ts, dict) and ts.get("title"):
                scenarios.append({
                    "id": ts.get("id", ""),
                    "title": ts.get("title", ""),
                    "description": ts.get("description", ""),
                    "test_type": ts.get("test_type", "功能测试"),
                })
        if not scenarios:
            continue
        test_points.append({
            "id": tp.get("id", f"TP-{len(test_points)+1:03d}"),
            "title": tp.get("title", ""),
            "description": tp.get("description", ""),
            "priority": _normalize_priority(tp.get("priority", "中")),
            "scenarios": scenarios,
        })

    total_scenarios = sum(len(tp["scenarios"]) for tp in test_points)
    summary = {
        "total_test_points": len(test_points),
        "total_test_scenarios": total_scenarios,
        "high_priority_points": sum(1 for tp in test_points if tp["priority"] == "高"),
        "medium_priority_points": sum(1 for tp in test_points if tp["priority"] == "中"),
        "low_priority_points": sum(1 for tp in test_points if tp["priority"] == "低"),
    }
    return {"test_points": test_points, "summary": summary}


def analyze_prd(markdown: str, provider: str = "deepseek",
                llm_invoke: Callable[[str, str], str] | None = None,
                **kwargs) -> dict:
    """【旧 API 兼容】分析 PRD 文档，返回测试点 + 测试场景。"""
    prompt = build_prd_prompt(markdown)
    invoke = llm_invoke or (lambda p, prov: chat_completion(p, provider=prov, **kwargs))
    response = invoke(prompt, provider)

    json_text = _extract_json_object(response)
    if not json_text:
        return {"test_points": [], "summary": {}, "error": "无法从响应中提取 JSON", "raw": response}

    import json as _json
    try:
        raw = _json.loads(json_text)
    except Exception:
        try:
            from ..core.generate import _repair_quotes_in_values
            raw = _json.loads(_repair_quotes_in_values(json_text))
        except Exception:
            return {"test_points": [], "summary": {}, "error": "JSON 解析失败", "raw": response}

    return _validate_analysis_result(raw)


if __name__ == "__main__":
    print(__doc__)
