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


# ---------------- 7.7.48 未替换凭证占位符（无 token 执行的专项提示） ----------------

def test_classify_unfilled_valid_jwt_hint():
    """Authorization 仍为 <valid_jwt> 字面量 → 提示未配置凭证（不是用例错误）。"""
    from pyst.eval.feedback import classify_failure, CAT_CASE
    cat, hint = classify_failure({
        "verdict": "FAIL", "status": 401, "expected_status": 200,
        "description": "正常创建条目", "token_applied": False,
        "request": {"method": "POST", "url": "/api/v1/items/",
                    "headers": {"Authorization": "Bearer <valid_jwt>"}}})
    assert cat == CAT_CASE and "测试凭证" in hint and "不是用例错误" in hint
    # token 生效时（request 里的占位符只是展示原文）不误报"未替换"
    cat2, hint2 = classify_failure({
        "verdict": "FAIL", "status": 404, "expected_status": 200,
        "description": "正常创建条目", "token_applied": True,
        "request": {"method": "POST", "url": "/api/v1/items/",
                    "headers": {"Authorization": "Bearer <valid_jwt>"}}})
    assert "测试凭证" not in hint2


def test_classify_5xx_beats_intent_assertion():
    """实际 500 优先判 potential_bug，不被意图断言判成"预期码写错"（7.7.49）。"""
    from pyst.eval.feedback import classify_failure, CAT_BUG
    cat, hint = classify_failure({
        "verdict": "FAIL", "status": 500, "expected_status": 422,
        "description": "skip 为负数时请求物品列表，验证参数校验失败",
        "token_applied": True,
        "request": {"method": "GET", "url": "/api/v1/items/",
                    "headers": {"Authorization": "Bearer <valid_jwt>"},
                    "query": {"skip": -1}}})
    assert cat == CAT_BUG and "真实缺陷" in hint


# ---------------- 7.7.51 工具化回灌（诊断 Agent 复用 read_source） ----------------

def test_read_source_snippet_sandbox(tmp_path):
    """共用读源码工具：正常读取带行号；越界/敏感路径拒绝。"""
    from pyst.eval.fixer import read_source_snippet
    proj = tmp_path / "proj"
    (proj / "app").mkdir(parents=True)
    (proj / "app" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (proj / ".env").write_text("SECRET=x", encoding="utf-8")
    out = read_source_snippet(str(proj), "app/m.py", 1, 10)
    assert "def f()" in out and "1: " in out
    assert "越界" in read_source_snippet(str(proj), "../outside.py")
    assert "敏感" in read_source_snippet(str(proj), ".env")


class _ToolLLM:
    """第一轮回 read_source 工具调用，第二轮回最终 JSON。"""
    def __init__(self):
        self.calls = []

    def chat_raw(self, messages, tools=None, **kw):
        self.calls.append(list(messages))
        last = messages[-1]
        if last.get("role") == "tool":
            return {"role": "assistant", "content":
                    '{"diagnosis": [{"description": "d", "category": "case_defect", '
                    '"cause": "看到源码确认", "action": "已修正用例"}], '
                    '"cases": [{"description": "c1", "request": {"method": "GET", "url": "/x"}, '
                    '"expected_status": 200, "test_steps": ["1. 发请求"], '
                    '"expected_results": ["1. 返回200"]}]}'}
        return {"role": "assistant", "tool_calls": [{"id": "t1", "function": {
            "name": "read_source", "arguments": '{"file": "src/m.py", "start": 1, "end": 20}'}}]}


def test_feedback_with_tools_loop(tmp_path):
    """工具化回灌：Agent 调 read_source 看代码后输出最终诊断+用例。"""
    from pyst.eval.feedback import feedback_refine_cases_with_tools
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("def create():\n    pass\n", encoding="utf-8")
    llm = _ToolLLM()
    diagnosis, cases = feedback_refine_cases_with_tools(
        {"entry": "e", "signature": "def x()", "control_sites": []},
        [], {"failures": [], "stats": {}, "aggregate_hints": [], "previous_attempts": []},
        case_count=5, project_root=str(tmp_path), llm_client=llm)
    assert diagnosis and diagnosis[0]["cause"] == "看到源码确认"
    assert cases and cases[0]["description"] == "c1"
    # 工具结果确实进入了对话（LLM 第二轮看到 tool 消息）
    assert any(m.get("role") == "tool" and "create" in m.get("content", "")
               for m in llm.calls[1])


def test_classify_duplicate_email_400():
    """400 "already exists" → 测试数据重复专项提示（7.7.55）。"""
    from pyst.eval.feedback import classify_failure, CAT_CASE
    cat, hint = classify_failure({
        "verdict": "FAIL", "status": 400, "expected_status": 200,
        "description": "超级管理员创建新用户",
        "response_snippet": '{"detail": "The user with this email already exists in the system."}'})
    assert cat == CAT_CASE and "唯一" in hint and "random_uuid" in hint
