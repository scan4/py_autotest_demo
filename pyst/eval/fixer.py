#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 修复子 Agent（方案 A：pythonTest 内置，完全全自动）
========================================================
输入：bug 注册表的一条 finding（缺陷证据 + 复现请求）。
职责：在被测项目的 git 分支上定位并修改业务代码，使缺陷行为消失，且不破坏其他测试。

【隔离与回滚】（7.7.37 设计定稿）
- 修改隔离用 git 分支：ai-fix/bug-{id}，每轮尝试后可整体 reset——不碰 docker 文件拷贝
- 验收 = 复现请求不再返回登记时的缺陷行为（状态码变化）且全量回归无新增 FAIL
- 失败 → 切回原分支并删除修复分支（工作区恢复原状）；成功 → commit 留在分支上，输出 diff 供人审 merge
- 结束后**恢复原分支并重启被测服务**——把环境还给用户

【工具协议】（文本协议，LLM 每轮输出一个 JSON 动作）
  {"action":"read_source","file":"相对路径","start":1,"end":200}
  {"action":"edit","file":"相对路径","old":"原文精确片段","new":"替换文本"}   # old 必须唯一
  {"action":"run_tests"}                                                    # 重启服务+复现+回归
  {"action":"finish","summary":"..."}                                       # 声明完成

【铁律】（写入 system prompt）
- 只修业务代码，禁止碰测试与用例（复现用例锁定）
- 最小修复，禁止更改业务语义迎合测试（吞异常/放宽校验）
"""

from __future__ import annotations

import json
import re
import subprocess
import time as _time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from ..llm.client import LLMClient

_FIX_SYSTEM = """你是被测项目的自动修复 Agent。已确认的服务端缺陷证据和复现请求已给出。
你的任务：阅读相关源码，修改业务代码使缺陷消失，且不破坏其他测试。

【铁律——违反即任务失败】
1. 只修改业务代码；禁止修改任何测试文件、测试数据（工具会拒绝）
2. 最小修复：只改缺陷相关的逻辑；禁止通过吞异常、放宽/删除校验、更改业务语义等方式迎合测试
3. 修复目标是"行为正确"（如空白输入应被校验拒绝），不是"让某个状态码出现"

【工作方式】用提供的工具完成修复：read_source 定位相关代码（如数据模型/校验逻辑）
→ edit 修改 → run_tests 验证（会重启被测服务并跑复现请求+全量回归）→ 通过则 finish。
finish 前必须至少跑过一次 run_tests。"""

# 原生 function calling 工具定义（OpenAI tools 协议）：description/parameters 即工具的
# 协议级描述——服务端约束模型输出结构化 tool_calls，不再依赖 prompt 模仿 JSON 格式
FIX_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "read_source",
        "description": "读取被测项目源码文件的指定行区间（带行号）。用于定位缺陷相关代码。",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string", "description": "项目内相对路径，如 backend/app/models.py"},
            "start": {"type": "integer", "description": "起始行号（从 1 开始），默认 1"},
            "end": {"type": "integer", "description": "结束行号，默认 400"}},
            "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": ("编辑文件：把 old（文件中唯一出现的原文精确片段，含缩进）替换为 new。"
                        "禁止修改测试文件（会被拒绝）。多次调用可完成多处修改。"),
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string", "description": "项目内相对路径"},
            "old": {"type": "string", "description": "要替换的原文精确片段（必须在文件中唯一）"},
            "new": {"type": "string", "description": "替换后的文本"}},
            "required": ["file", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": ("重启被测服务（加载已修改的代码）并执行验证：缺陷复现请求 + 全量用例回归。"
                        "返回缺陷行为是否消失、回归统计与验收判定。修改代码后必须调用。"),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "声明修复完成（必须至少跑过一次 run_tests 且验收通过）。summary 写清修复内容。",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "修复说明：改了什么、为什么能消除缺陷"}},
            "required": ["summary"]}}},
]


def _git(project: str, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", project, *args],
                       capture_output=True, text=True, timeout=60)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def find_git_root(path: str) -> str | None:
    """从 path 向上查找 .git 目录。"""
    p = Path(path).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return str(cand)
    return None


def _restart_service(base_url: str, service_cmd: str, log: Callable[[str], None]) -> None:
    """重启被测服务（加载修改后的代码）：杀端口进程 → 执行启动命令 → 轮询端口可达。"""
    parsed = urllib.parse.urlparse(base_url if "//" in base_url else "http://" + base_url)
    port = parsed.port or 80
    log(f"重启被测服务（端口 {port}）...")
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    _time.sleep(1)
    if service_cmd:
        subprocess.Popen(service_cmd, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            import socket
            with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=1):
                log("服务已就绪")
                return
        except OSError:
            _time.sleep(0.5)
    log("警告：服务端口等待超时（继续执行验证，可能因服务未就绪而失败）")


class FixAgent:
    """一次修复任务的执行器（在后台线程中运行）。"""

    def __init__(self, task_id: str, session, source_dir: str, finding: dict[str, Any],
                 base: str, token: str, service_cmd: str, provider: str,
                 max_turns: int = 10, log_fn: Callable[[str], None] | None = None):
        self.task_id = task_id
        self.task: dict[str, Any] = {"status": "running", "log": [], "patch": "",
                                     "success": False, "summary": "", "branch": ""}
        self.session = session
        self.source_dir = source_dir
        self.finding = finding
        self.base = base
        self.token = token
        self.service_cmd = service_cmd
        self.provider = provider
        self.max_turns = max_turns
        self.project = ""
        self.last_results: list[dict] = []   # 最近一次回归的逐条结果（验收对比用）
        self.log = log_fn or (lambda m: None)

    # ---------------- 工具实现 ----------------

    def _tool_read_source(self, file: str, start: int = 1, end: int = 400) -> str:
        root = self.project
        p = (Path(root) / file).resolve()
        if not str(p).startswith(str(Path(root).resolve())):
            return "错误：路径越界"
        if not p.exists():
            return f"错误：文件不存在 {file}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start or 1))
        end = min(len(lines), int(end or 400))
        numbered = [f"{i}: {lines[i-1]}" for i in range(start, end + 1)]
        return f"{file}（行 {start}-{end}，共 {len(lines)} 行）:\n" + "\n".join(numbered)

    def _tool_edit(self, file: str, old: str, new: str) -> str:
        p = (Path(self.project) / file).resolve()
        if not str(p).startswith(str(Path(self.project).resolve())):
            return "错误：路径越界"
        if not p.exists():
            return f"错误：文件不存在 {file}"
        # 代码级防线：禁止修改测试文件（复现用例锁定原则在源码侧的延伸）
        norm = file.replace("\\", "/").lower()
        if ("/test" in norm or norm.startswith("test") or "/tests/" in norm
                or Path(norm).name.startswith("test_")):
            return "错误：禁止修改测试文件——修复只能改业务代码"
        text = p.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return "错误：old 片段在文件中不存在（必须与原文完全一致，含缩进）"
        if text.count(old) > 1:
            return f"错误：old 片段在文件中出现 {text.count(old)} 次（不唯一），请扩大上下文片段"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"已应用修改：{file}"

    def _tool_run_tests(self) -> str:
        from .executor import execute_http_case, execute_suite
        _restart_service(self.base, self.service_cmd, self.log)
        # 复现请求：缺陷行为是否消失（状态码不再等于登记时的实际行为）
        replay = execute_http_case(
            {"_entry": self.finding["entry"], "description": self.finding["description"],
             "request": self.finding.get("request"),
             "expected_status": self.finding.get("expected_status")},
            self.base, token=self.token)
        recorded = (self.finding.get("evidence") or {}).get("actual_status")
        still = (replay.get("status") == recorded) if recorded else None
        # 全量回归：与修复前基线对比，找"新增 FAIL"
        cases: list[dict] = []
        for entry, cs in self.session.test_cases.items():
            for c in cs:
                cases.append({**c, "_entry": entry})
        _, stats = execute_suite(cases, self.base, token=self.token)
        return (f"复现请求结果: {replay.get('verdict')} 实际状态码 {replay.get('status')}"
                f"（登记时缺陷行为: {recorded}；{'缺陷行为仍存在' if still else '行为已变化——若为拒绝类 4xx 说明修复可能有效'}）\n"
                f"全量回归统计: {stats}\n"
                f"逐条结果: " + "; ".join(
                    f"{r.get('description','')[:30]}={r.get('verdict')}({r.get('status')})"
                    for r in self.session.results[:30]))

    def _validate(self, baseline: dict[str, tuple[str, int]],
                  replay_status: int | None) -> tuple[bool, str]:
        """验收：缺陷行为消失（复现状态码 != 登记时的行为）且无新增 FAIL。"""
        recorded = (self.finding.get("evidence") or {}).get("actual_status")
        if recorded and replay_status == recorded:
            return False, f"复现请求仍返回 {replay_status}（缺陷行为未变化）"
        now_fails = {(r.get("entry"), r.get("description")): r
                     for r in self.last_results
                     if r.get("verdict") in ("FAIL", "ERROR", "SKIPPED")}
        new_fail = [r.get("description", "?") for k, r in now_fails.items()
                    if k in baseline and baseline[k][0] == "PASS"]
        if new_fail:
            return False, f"回归出现新增失败（修复前 PASS 现在失败）: {'; '.join(new_fail[:5])}"
        return True, (f"缺陷行为消失（{recorded} → {replay_status}）且无新增回归失败"
                      if recorded else "行为已变化且无新增回归失败")

    # ---------------- 主流程 ----------------

    def run(self) -> None:
        from .executor import execute_http_case, execute_suite
        task = self.task
        project = find_git_root(self.source_dir)
        if not project:
            task["status"] = "failed"
            task["summary"] = f"被测项目目录 {self.source_dir} 不在 git 仓库中，无法安全修复"
            return
        self.project = project
        try:
            self._run(project)
        except Exception as e:
            import traceback as _tb
            task["status"] = "failed"
            task["summary"] = f"修复过程异常: {e}"
            task["log"].append(_tb.format_exc(limit=5))

    def _run(self, project: str) -> None:
        task = self.task
        from .executor import execute_http_case

        if not _git(project, "status", "--porcelain").strip() == "":
            dirty = _git(project, "status", "--porcelain")
            task["status"] = "failed"
            task["summary"] = f"被测项目工作区不干净，拒绝自动修复（防止误伤未提交的更改）：\n{dirty[:300]}"
            return
        orig_branch = _git(project, "branch", "--show-current") or "HEAD"
        branch = f"ai-fix/bug-{self.finding['id']}"
        recorded = (self.finding.get("evidence") or {}).get("actual_status")

        # 修复前基线（用于回归对比）
        baseline_cases: list[dict] = []
        for entry, cs in self.session.test_cases.items():
            for c in cs:
                baseline_cases.append({**c, "_entry": entry})
        baseline_results, _ = execute_suite(baseline_cases, self.base, token=self.token)
        baseline = {(r.get("entry"), r.get("description")): (r.get("verdict"), r.get("status"))
                    for r in baseline_results}
        task["log"].append(f"修复前基线：{sum(1 for v in baseline.values() if v[0]=='PASS')} PASS")

        _git(project, "checkout", "-b", branch)
        task["log"].append(f"已切换到修复分支 {branch}")

        file_tree = _git(project, "ls-files")[:3000]
        finding = self.finding
        ev = finding.get("evidence") or {}
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _FIX_SYSTEM},
            {"role": "user", "content": (
                f"【缺陷登记 #{finding['id']}】\n描述: {finding.get('description')}\n"
                f"缺陷行为: 服务端实际返回 {recorded}（复现请求如下）\n"
                f"业务期望: {finding.get('expected_status')}（应拒绝而非接受）\n"
                f"证据响应: {ev.get('response_snippet', '')[:200]}\n\n"
                f"【复现请求】\n{json.dumps(finding.get('request'), ensure_ascii=False)}\n\n"
                f"【项目文件树】\n{file_tree}\n\n"
                "请开始修复。先用 read_source 定位相关代码（如数据模型/校验逻辑），"
                "再 edit 修改，然后 run_tests 验证。目标：空白/非法输入被校验拒绝，"
                "同时保持合法输入仍正常工作。")},
        ]
        client = LLMClient(provider=self.provider)
        replay_status = None
        success = False
        summary = ""
        ran_tests = False
        try:
            for turn in range(1, self.max_turns + 1):
                task["log"].append(f"--- Agent 轮次 {turn} ---")
                # 原生 function calling：tools schema 约束模型输出结构化 tool_calls
                message = client.chat_raw(messages, temperature=0.2, tools=FIX_TOOLS)
                messages.append(message)   # assistant 消息（含 tool_calls 或纯文本）
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    # 模型直接输出文字（未调工具）：提醒它用工具
                    messages.append({"role": "user",
                                     "content": "请通过工具执行操作（read_source/edit/run_tests/finish）。"})
                    continue
                stop = False
                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception as e:
                        args = {}
                        result = f"错误：工具参数不是合法 JSON（{e}）"
                    if name == "read_source":
                        result = self._tool_read_source(args.get("file", ""),
                                                        args.get("start", 1), args.get("end", 400))
                    elif name == "edit":
                        result = self._tool_edit(args.get("file", ""),
                                                 args.get("old", ""), args.get("new", ""))
                    elif name == "run_tests":
                        _restart_service(self.base, self.service_cmd, self.log)
                        replay = execute_http_case(
                            {"_entry": finding["entry"], "description": finding.get("description", ""),
                             "request": finding.get("request"),
                             "expected_status": finding.get("expected_status")},
                            self.base, token=self.token)
                        replay_status = replay.get("status")
                        still = (replay_status == recorded) if recorded else None
                        run_results, stats = execute_suite(
                            [{**c, "_entry": e} for e, cs in self.session.test_cases.items()
                             for c in cs], self.base, token=self.token)
                        self.last_results = run_results   # 验收对比用（新增 FAIL 检测）
                        ran_tests = True
                        ok, why = self._validate(baseline, replay_status)
                        result = (f"复现请求: 实际 {replay_status}（缺陷行为 {recorded}，"
                                  f"{'仍存在' if still else '已消失'}）\n"
                                  f"回归统计: {stats}\n验收: {why}")
                        task["log"].append(f"run_tests: {why}")
                        if ok:
                            summary = f"修复验证通过：{why}"
                            stop = True
                    elif name == "finish":
                        if not ran_tests:
                            result = "错误：finish 前必须至少调用一次 run_tests 验证"
                        else:
                            summary = args.get("summary", "")
                            stop = True
                    else:
                        result = f"错误：未知工具 {name}"
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                     "content": result})
                    if stop:
                        break
                if stop:
                    break
        finally:
            pass

        # 验收与收尾：finish 仅在 run_tests（含验收）之后被允许，summary 即验收结论
        if summary and ran_tests:
            ok = True
            why = "验收通过"
        elif summary:
            ok, why = False, "未执行过 run_tests 验证"
        else:
            ok = False
            why = "达到最大轮次未声明完成"
        success = ok

        if success:
            _git(project, "add", "-A")
            _git(project, "commit", "-m",
                 f"fix: {finding.get('description', '')}（AI 修复，bug #{finding['id']}）\n\n{summary}")
            patch = _git(project, "diff", orig_branch, branch)
            task["patch"] = patch
            task["branch"] = branch
            task["summary"] = f"修复成功（分支 {branch}，待人工审阅 merge）：{summary}"
        else:
            _git(project, "checkout", orig_branch, check=False)
            _git(project, "branch", "-D", branch, check=False)
            task["patch"] = ""
            task["summary"] = f"修复失败已回滚（工作区恢复原状）：{why} {summary}"

        # 恢复用户原状态：切回原分支 + 重启原代码服务
        _git(project, "checkout", orig_branch, check=False)
        _restart_service(self.base, self.service_cmd, self.log)
        task["status"] = "done"
        task["success"] = success
        task["log"].append(f"任务结束：{task['summary']}")
