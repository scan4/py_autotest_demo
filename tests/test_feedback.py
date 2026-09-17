"""feedback.py 测试：三分类规则表、聚合提示、意图断言集成、prompt 渲染段（7.7.35/7.7.45）。"""
from pyst.eval.feedback import classify_failure, structure_failures, build_feedback_prompt


def _r(**kw):
    base = {"verdict": "FAIL", "status": 200, "entry": "e1", "description": "d",
            "reason": "r", "request": {"method": "GET", "url": "/x"}, "response_snippet": ""}
    base.update(kw)
    return base


# ---------------- classify_failure 全规则表 ----------------

def test_error_latin1_is_case_defect():
    cat, hint = classify_failure(_r(verdict="ERROR", reason="latin-1 codec can't encode"))
    assert cat == "case_defect"


def test_error_other_is_env():
    assert classify_failure(_r(verdict="ERROR", reason="connection refused"))[0] == "env"


def test_skipped_is_case_defect():
    assert classify_failure(_r(verdict="SKIPPED"))[0] == "case_defect"


def test_404_is_case_defect():
    assert classify_failure(_r(status=404, expected_status=200))[0] == "case_defect"


def test_validation_json_invalid_hint():
    cat, hint = classify_failure(_r(status=422, expected_status=200,
                                    response_snippet='{"detail":[{"type":"json_invalid"}]}'))
    assert cat == "case_defect" and "JSON" in hint


def test_403_with_auth_header_is_case_defect():
    r = _r(status=403, expected_status=401,
           request={"headers": {"Authorization": "Bearer bad"}}, description="无效令牌")
    assert classify_failure(r)[0] == "case_defect"


def test_403_without_auth_header_is_env():
    assert classify_failure(_r(status=403, request={"headers": {}}))[0] == "env"


def test_401_is_env():
    assert classify_failure(_r(status=401))[0] == "env"


def test_5xx_is_potential_bug():
    assert classify_failure(_r(status=500, expected_status=200))[0] == "potential_bug"


def test_blank_title_200_special_hint():
    cat, hint = classify_failure(_r(status=200, expected_status=400,
                                    description="title 为仅空白字符创建物品"))
    assert cat == "case_defect" and "空白" in hint


# ---------------- 聚合提示（7.7.40 教训：ERROR 无状态码不算"全部"） ----------------

def test_aggregate_mixed_failures_no_false_all_hint():
    r = structure_failures([
        _r(verdict="FAIL", status=404, expected_status=200),
        _r(verdict="ERROR", error="conn refused")])
    assert r["aggregate_hints"] == []


def test_aggregate_all_error_hint():
    r = structure_failures([
        _r(verdict="ERROR", error="refused"), _r(verdict="ERROR", error="refused")])
    assert any("未启动" in h or "不可达" in h for h in r["aggregate_hints"])


# ---------------- 意图断言集成（7.7.45） ----------------

def test_intent_assertion_fail_hint():
    r = _r(status=400, expected_status=422,
           assertion={"mode": "intent", "intent": "validation", "acceptable": [400, 413, 415, 422]})
    cat, hint = classify_failure(r)
    assert cat == "case_defect" and "意图断言不符" in hint and "修预期" in hint


# ---------------- prompt 渲染段（7.7.34/7.7.30） ----------------

def test_prompt_contains_bug_repro_and_memory_sections():
    p = build_feedback_prompt(
        {"entry": "e1", "signature": "s", "control_sites": [], "interface": None},
        [], {"failures": [],
             "bug_repro": [{"description": "空白用例", "expected_status": 400, "evidence": "实际200"}],
             "previous_attempts": [{"entry": "e1", "description": "d", "cause": "c",
                                    "action": "a", "next_result": "FAIL（实际 403）"}]})
    assert "已知缺陷复现用例——禁止修改" in p and "空白用例" in p
    assert "前几轮已尝试但未成功的修正" in p and "FAIL（实际 403）" in p
