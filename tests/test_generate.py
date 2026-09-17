"""generate.py 测试：用例校验、JSON 容错解析、prompt 约束段（7.7.15/7.7.19/7.7.25）。"""
from pyst.core.generate import (_build_api_output_section, _parse_llm_json,
                                _validate_test_cases, build_prompt)


def _c(desc="x", method="GET", url="/x", exp=200):
    return {"description": desc, "request": {"method": method, "url": url},
            "expected_status": exp, "test_steps": ["1"], "expected_results": ["1"]}


def test_validate_accepts_valid_case():
    out = _validate_test_cases([_c()], 5)
    assert len(out) == 1 and out[0].expected_status == 200


def test_validate_rejects_pseudo_protocol():
    """7.7.19：非 HTTP 入口伪装成 request（CALL/callable://）必须拒绝"""
    bad = _c()
    bad["request"] = {"method": "CALL", "url": "callable://mod.func"}
    assert _validate_test_cases([bad], 5) == []


def test_validate_rejects_missing_steps():
    c = _c(); c["test_steps"] = []
    assert _validate_test_cases([c], 5) == []


def test_validate_caps_case_count():
    """7.7.15：输出条数硬截断（防打爆 token）"""
    out = _validate_test_cases([_c(f"c{i}") for i in range(20)], 5)
    assert len(out) == 5


def test_parse_llm_json_quote_repair():
    s = '{"a": "说"你好""}'
    assert _parse_llm_json(s)["a"].startswith("说")


def test_parse_llm_json_valid():
    assert _parse_llm_json("[1,2]") == [1, 2]


def test_prompt_has_security_sections():
    p = build_prompt({"description": "d", "entry": "e", "module": "m"}, 5)
    assert "占位描述并以 < > 包裹" in p
    assert "Bearer <valid_jwt>" in p
    assert "故障注入" in p


def test_api_output_section_contract_authority():
    p = _build_api_output_section({"url": "/api/x", "methods": ["GET"],
                                   "params": [{"name": "id", "location": "path"}]})
    assert "/api/x" in p and "id" in p
