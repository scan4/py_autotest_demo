"""webapp 路由级集成测试（TestClient + mock LLM/executor，7.7.44 集成层）。"""
import os
import pytest
from fastapi.testclient import TestClient

import pyst.webapp as webapp
from pyst import webapp as W


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_DB_PATH", tmp_path / "t.db")
    return TestClient(webapp.app), monkeypatch


DIR = "/tmp/pyst-it-proj"


def _ensure_dir():
    import os
    os.makedirs(DIR, exist_ok=True)


def _session():
    return list(W._sessions.values())[-1]


def test_health(client):
    c, _ = client
    assert c.get("/api/health").json()["status"] == "ok"


def test_feedback_forced_restore_e2e(client, monkeypatch):
    """7.7.34 端到端：LLM 假修复（改预期 200）→ 强制恢复回注册表预期 400"""
    _ensure_dir()
    c, mp = client
    c.post("/api/analyze", json={"source_dir": DIR})
    s = _session()
    s.test_cases = {"e1": [{"description": "空白 title 用例",
                            "request": {"method": "POST", "url": "/x"},
                            "expected_status": 400, "test_steps": ["1"],
                            "expected_results": ["1"]}]}
    s.results = [{"entry": "e1", "description": "空白 title 用例", "verdict": "FAIL",
                  "status": 200, "expected_status": 400,
                  "request": {"method": "POST", "url": "/x"},
                  "response_snippet": '{"title":"   "}', "reason": "200!=400"}]
    mp.setattr("pyst.eval.feedback.feedback_refine_cases_with_tools", lambda *a, **k: (
        [{"description": "空白 title 用例", "category": "case_defect", "cause": "c",
          "action": "已修正用例"}],
        [{"description": "空白 title 用例", "request": {"method": "POST", "url": "/x"},
          "expected_status": 200, "test_steps": ["1"], "expected_results": ["1"]}]))
    mp.setattr("pyst.storage.db.TestCaseStore.list_bug_findings",
               lambda self, entry=None, source_dir="": [
                   {"id": 1, "entry": "e1", "source_dir": source_dir,
                    "description": "空白 title 用例",
                    "request": {"method": "POST", "url": "/x"}, "expected_status": 400,
                    "evidence": {"actual_status": 200}, "created_at": "t"}])
    r = c.post("/api/feedback", json={"source_dir": DIR}).json()
    e1 = r["entries"]["e1"]["refined_cases"][0]
    assert e1["expected_status"] == 400   # 假修复的 200 被强制恢复


def test_iterate_no_improvement_stop(client, monkeypatch):
    """7.7.32：无改进停止 + potential_bug 不参与判定（7.7.42 变体收敛语义）"""
    _ensure_dir()
    c, mp = client
    c.post("/api/analyze", json={"source_dir": DIR})
    s = _session()
    s.test_cases = {"e1": [{"description": "title 为纯空白字符创建物品",
                            "request": {"method": "POST", "url": "/x"},
                            "expected_status": 400, "test_steps": ["1"],
                            "expected_results": ["1"]}]}
    mp.setattr("pyst.eval.executor.execute_suite_with_resources", lambda cases, base, token="": (
        [{"entry": "e1", "description": "title 为纯空白字符创建物品", "verdict": "FAIL", "status": 200,
          "category": "potential_bug", "request": {"method": "POST", "url": "/x"}}],
        {"FAIL": 1}))
    mp.setattr("pyst.eval.feedback.feedback_refine_cases_with_tools", lambda *a, **k: (
        [{"description": "title 为纯空白字符创建物品", "category": "potential_bug", "cause": "c",
          "action": "建议报告bug"}],
        [{"description": "title 为纯空白字符创建物品", "request": {"method": "POST", "url": "/x"},
          "expected_status": 400, "test_steps": ["1"], "expected_results": ["1"]}]))
    mp.setattr("pyst.storage.db.TestCaseStore.list_bug_findings", lambda self, entry=None, source_dir="": [
                   {"id": 1, "entry": "e1", "source_dir": source_dir,
                    "description": "title 为纯空白字符创建物品",
                    "request": {"method": "POST", "url": "/x"}, "expected_status": 400,
                    "evidence": {"actual_status": 200}, "created_at": "t"}])
    r = c.post("/api/iterate", json={"source_dir": DIR, "base_url": "http://localhost:59999",
                                "max_rounds": 3}).json()
    assert r["converged"] is True and "已知缺陷复现用例" in r["stopped_reason"]
    assert all(rd["failed_bug"] == 1 and rd["failed_other"] == 0 for rd in r["rounds"])


def test_fix_start_requires_confirm(client):
    c, _ = client
    r = c.post("/api/fix/start", json={"source_dir": DIR, "finding_id": 999})
    assert r.status_code == 400 and "显式确认" in r.json()["detail"]


# ---------------- 7.7.53 replay 路由回归（branches 未赋值导致 500） ----------------

def _mk_bug(store, source_dir, actual_status) -> int:
    """登记一条缺陷（复现请求带 url 供 mock 用），返回 finding id。"""
    return store.upsert_bug_finding(
        "e1", source_dir, "负数 limit 请求列表", 
        {"method": "GET", "url": "/api/v1/items/", "headers": {}, "query": {"limit": -1}},
        422, {"actual_status": actual_status, "response_snippet": "err"})


def test_replay_behavior_changed_path(client, monkeypatch, tmp_path):
    """重放判定"行为已变化"（still=False）路径不得 500（7.7.53 回归：branches 未赋值）。"""
    tc, mp = client
    src = str(tmp_path / "proj")
    os.makedirs(src, exist_ok=True)
    from pyst.storage.db import TestCaseStore
    _mk_bug(TestCaseStore(webapp._DB_PATH), src, 500)
    import pyst.eval.fixer as fx
    mp.setattr(fx, "find_git_root", lambda p: None)          # 无 git → project=None
    mp.setattr("pyst.eval.executor.execute_http_case",
               lambda case, base, token="": {"verdict": "FAIL", "status": 422})
    r = tc.post("/api/bugs/1/replay", json={"source_dir": src, "base_url": "http://x"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["still_reproduces"] is False                  # 500 → 422，行为已变化
    assert "解除登记" in data["suggestion"]


def test_replay_still_reproduces_path(client, monkeypatch, tmp_path):
    """重放仍复现路径（修复分支不存在 → project=None）：suggestion 正常拼接。"""
    tc, mp = client
    src = str(tmp_path / "proj2")
    os.makedirs(src, exist_ok=True)
    from pyst.storage.db import TestCaseStore
    fid = _mk_bug(TestCaseStore(webapp._DB_PATH), src, 500)
    import pyst.eval.fixer as fx
    mp.setattr(fx, "find_git_root", lambda p: None)
    mp.setattr("pyst.eval.executor.execute_http_case",
               lambda case, base, token="": {"verdict": "FAIL", "status": 500})
    r = tc.post(f"/api/bugs/{fid}/replay", json={"source_dir": src, "base_url": "http://x"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["still_reproduces"] is True
    assert "保留登记" in data["suggestion"]


# ---------------- 7.7.54 AI 修复并发防护 ----------------

def test_fix_start_rejects_concurrent_run(client, monkeypatch, tmp_path):
    """已有修复任务 running 时，新的修复请求必须 409（防并行线程互踩 git 工作区）。"""
    tc, mp = client
    src = str(tmp_path / "proj")
    os.makedirs(src, exist_ok=True)
    from pyst.storage.db import TestCaseStore
    fid = TestCaseStore(webapp._DB_PATH).upsert_bug_finding(
        "e1", src, "负数 limit", {"method": "GET", "url": "/x"}, 422,
        {"actual_status": 500})
    webapp._fix_tasks["fake_running"] = {"status": "running", "log": [], "success": False,
                                         "summary": "", "branch": "", "patch": ""}
    try:
        r = tc.post("/api/fix/start", json={
            "source_dir": src, "finding_id": fid, "base_url": "http://x",
            "provider": "deepseek", "max_turns": 3, "confirm": True})
        assert r.status_code == 409, r.text
        assert "已有 AI 修复任务在运行" in r.json()["detail"]
    finally:
        webapp._fix_tasks.pop("fake_running", None)
