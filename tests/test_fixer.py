"""fixer.py 测试：路径沙箱六场景、占位识别、验收判定、fake-LLM 端到端工具循环与回滚。"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyst.eval.fixer import FixAgent
from pyst.eval.executor import (_expand_long_value, _is_token_placeholder,
                                _apply_token)


# ---------------- 路径沙箱（7.7.41） ----------------

@pytest.fixture()
def agent(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = FixAgent("t", None, str(proj), {"id": 1}, "", "", "", "deepseek")
    a.project = str(proj)
    return a


def test_safe_path_allows_inside(agent, tmp_path):
    f = tmp_path / "proj" / "app.py"
    f.write_text("x = 1")
    p, err = agent._safe_path("app.py")
    assert err == "" and p == f


def test_safe_path_rejects_sibling_dir(agent, tmp_path):
    p, err = agent._safe_path("../proj2/evil.py")
    assert p is None and "越界" in err


def test_safe_path_rejects_escape(agent):
    p, err = agent._safe_path("../../etc/passwd")
    assert p is None and "越界" in err


def test_safe_path_rejects_env_and_git(agent):
    assert agent._safe_path(".env")[0] is None
    assert agent._safe_path(".env.local")[0] is None
    assert agent._safe_path(".git/config")[0] is None


def test_safe_path_rejects_github_workflows(agent):
    assert agent._safe_path(".github/workflows/ci.yml")[0] is None


# ---------------- 占位识别与展开（7.7.26/7.7.25） ----------------

def test_placeholder_variants():
    assert _is_token_placeholder("Bearer <valid_jwt>")
    assert _is_token_placeholder("Bearer <有效token>")
    assert _is_token_placeholder("Bearer 超级用户token")
    assert not _is_token_placeholder("Bearer eyJhbGciOi.J9.abc")


def test_expand_exact_and_over():
    assert len(_expand_long_value("<超长字符串：恰好达到 title 长度上限（如 255 字符）>")) == 255
    assert len(_expand_long_value("<超长字符串：重复A至超过 title 长度上限（如 256 字符）>")) == 512


# ---------------- 验收判定（7.7.45） ----------------

def _agent_for_validate():
    a = FixAgent("t", None, "/proj", {"id": 1, "evidence": {"actual_status": 200}}, "", "", "", "x")
    return a


def test_validate_behavior_change_pass():
    a = _agent_for_validate()
    a.last_results = []
    ok, why = a._validate({}, 422)
    assert ok and "消失" in why


def test_validate_new_regression_fails():
    a = _agent_for_validate()
    a.last_results = [{"entry": "e", "description": "其他用例", "verdict": "FAIL", "status": 500}]
    baseline = {("e", "其他用例"): ("PASS", 200)}
    ok, _ = a._validate(baseline, 422)
    assert not ok and "新增失败" in _


# ---------------- fake-LLM 端到端工具循环（成功 + 失败回滚） ----------------

def _mk_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "backend").mkdir(parents=True)
    (Path(proj) / "backend" / "app.py").write_text("def validate_title(t):\n    return len(t) >= 1\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(proj)])
    subprocess.run(["git", "-C", str(proj), "add", "-A"])
    subprocess.run(["git", "-C", str(proj), "commit", "-qm", "init"])
    return str(proj)


def _tc(name, args):
    return {"id": "c1", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class FakeLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat_raw(self, messages, tools=None, **kw):
        if self.turns:
            return self.turns.pop(0)
        return {"content": "no more", "role": "assistant"}


@pytest.fixture()
def fix_env(tmp_path, monkeypatch):
    proj = _mk_project(tmp_path)
    import pyst.eval.fixer as fx
    monkeypatch.setattr(fx, "_restart_service", lambda *a, **k: None)
    monkeypatch.setattr("pyst.eval.fixer.execute_http_case",
                        lambda case, base, token="": {"verdict": "PASS", "status": 422})
    monkeypatch.setattr("pyst.eval.fixer.execute_suite_with_resources",
                        lambda cases, base, token="", log=None: ([], {"PASS": 1}))
    return proj, fx


def _finding():
    return {"id": 1, "entry": "e1", "description": "空白 title 可入库",
            "request": {"method": "POST", "url": "/x"}, "expected_status": 400,
            "evidence": {"actual_status": 200}}


def test_fix_e2e_success(fix_env, tmp_path):
    proj, fx = fix_env
    old = "def validate_title(t):\n    return len(t) >= 1\n"
    new = "def validate_title(t):\n    return bool(t.strip())\n"
    turns = [
        {"role": "assistant", "tool_calls": [_tc("edit", {
            "file": "backend/app.py",
            "old": old, "new": new})]},
        {"role": "assistant", "tool_calls": [_tc("run_tests", {})]},
        {"role": "assistant", "tool_calls": [_tc("finish", {"summary": "加了 strip 校验"})]},
    ]
    a = FixAgent("t1", SimpleNamespace(test_cases={}), str(proj), _finding(), str(proj),
                 "", "", "deepseek", max_turns=5, llm_client=FakeLLM(turns))
    a.run()
    assert a.task["success"] is True
    # 设计语义：成功后主工作区切回原分支，修复内容在 ai-fix 分支上待人工审 merge
    fixed = subprocess.run(["git", "-C", proj, "show", "ai-fix/bug-1:backend/app.py"],
                           capture_output=True, text=True).stdout
    assert "bool(t.strip())" in fixed
    assert "ai-fix/bug-1" not in subprocess.run(
        ["git", "-C", proj, "branch", "--show-current"], capture_output=True, text=True).stdout


def test_fix_e2e_rollback_on_llm_error(fix_env, tmp_path):
    proj, fx = fix_env
    class BoomLLM:
        def chat_raw(self, messages, tools=None, **kw):
            raise RuntimeError("network down")
    a = FixAgent("t2", SimpleNamespace(test_cases={}), str(proj), _finding(), str(proj),
                 "", "", "deepseek", max_turns=3, llm_client=BoomLLM())
    a.run()
    assert a.task["success"] is False
    assert (Path(proj) / "backend" / "app.py").read_text() == "def validate_title(t):\n    return len(t) >= 1\n"
    assert "ai-fix/bug-1" not in subprocess.run(
        ["git", "-C", proj, "branch", "--list"], capture_output=True, text=True).stdout


def test_fix_e2e_rollback_on_failed_validation(fix_env, tmp_path):
    proj, fx = fix_env
    import pyst.eval.executor as ex
    # 复现请求仍返回 200（缺陷行为未变化）→ 验收失败
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ex, "execute_http_case",
                   lambda case, base, token="": {"verdict": "FAIL", "status": 200})
    turns = [{"role": "assistant", "tool_calls": [_tc("finish", {"summary": "没验证就说完成"})]}]
    a = FixAgent("t3", SimpleNamespace(test_cases={}), str(proj), _finding(), str(proj),
                 "", "", "deepseek", max_turns=3, llm_client=FakeLLM(turns))
    try:
        a.run()
    finally:
        monkey.undo()
    assert a.task["success"] is False
    assert (Path(proj) / "backend" / "app.py").read_text() == "def validate_title(t):\n    return len(t) >= 1\n"
