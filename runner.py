#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例执行器：把生成的可执行测试用例直接执行
============================================================
两种形态（M8 盲区覆盖）：
  1. HTTP 形态：用例带 request 字段 → 发 HTTP 请求 + 断言 PASS/FAIL
  2. 纯函数形态：用例带 call_args/call_kwargs 字段 → 同进程 import 被测函数调用
     + 用 coverage.py 采集语句/分支覆盖，输出盲区报告（--cover 指定被测源码）

用法：
    1. 生成测试用例并导出 JSON：
       python -m pyst <root> --phase 4 ...   （或用 webapp /generate，见下）
       curl -X POST localhost:8001/generate -H 'Content-Type: application/json' \
            -d '{"source_dir":"/path","provider":"deepseek"}' > cases.json

    2. 执行 HTTP 用例（默认读 cases.json，POST 到 base_url）：
       python runner.py cases.json --base http://localhost:8000

    3. 纯函数覆盖（被测函数可同进程 import）：
       python runner.py cases.json --cover /path/to/module.py \
            --entry 'apps.core.utils.validate'

       # 被测入口限定名若与功能点档案 entry 一致，可用 --entry 显式指定；
       # 否则在 cases.json 的 test_cases 键里按 entry 对应。
"""

import argparse
import json
import sys
from pathlib import Path

# HTTP 执行逻辑抽到 pyst/eval/executor.py，与 webapp /api/execute（AI 执行）共用
sys.path.insert(0, str(Path(__file__).parent))
from pyst.eval.executor import execute_http_case, verdict_mark  # noqa: E402


def _exec_func_one(case: dict) -> dict:
    """校验一条纯函数用例的调用参数是否齐备（不真正执行，参数攒给覆盖采集统一跑）。"""
    args = case.get("call_args")
    if args is None:
        return {"description": case.get("description", "?"), "status": "SKIPPED",
                "verdict": "SKIPPED", "reason": "缺少 call_args 字段（纯函数用例需提供调用参数）"}
    return {"description": case.get("description", "?"), "status": "OK",
            "verdict": "PENDING", "reason": "参数已收集，等待覆盖采集执行"}


def _collect_func_calls(cases: list[dict]) -> tuple[list[list], list[dict], list[dict]]:
    """从纯函数用例里收集全部调用参数。

    Returns:
        (args_list, kwargs_list, func_case_results)
    """
    args_list: list[list] = []
    kwargs_list: list[dict] = []
    results: list[dict] = []
    for c in cases:
        r = _exec_func_one(c)
        if r["verdict"] == "PENDING":
            args_list.append(c.get("call_args") or [])
            kwargs_list.append(c.get("call_kwargs") or {})
        results.append(r)
    return args_list, kwargs_list, results


def _run_coverage(cases: list[dict], cover_path: str, entry: str):
    """对纯函数用例跑盲区覆盖分析，返回覆盖报告文本 + dict。

    按 entry（功能点）分组：同一入口的所有用例参数攒一起，
    调 pyst.eval.coverage.analyze_function 做语句/分支覆盖 + 盲区比对。
    """
    from pyst.eval.coverage import analyze_function, render_report

    # 按入口分组（用例可能带 _entry，来自 test_cases 字典结构）
    by_entry: dict[str, list[dict]] = {}
    for c in cases:
        e = c.get("_entry") or entry
        by_entry.setdefault(e, []).append(c)

    reports: list[str] = []
    all_results = []
    for e, ecases in by_entry.items():
        args_list, kwargs_list, _ = _collect_func_calls(ecases)
        if not args_list:
            reports.append(f"[{e}] 无用例携带 call_args，跳过覆盖采集")
            continue
        result = analyze_function(
            feature={"entry": e},          # 覆盖采集不依赖完整档案（纯函数路径）
            module_path=cover_path,
            call_args_list=args_list,
            call_kwargs_list=kwargs_list,
            entry_qualname=e,
        )
        result.passed = sum(1 for c in ecases if c.get("expected_status") is not None)
        reports.append(render_report(result))
        all_results.append(result.to_dict())
    return "\n\n".join(reports), all_results


def main():
    ap = argparse.ArgumentParser(description="执行可执行的测试用例（HTTP 或纯函数覆盖）")
    ap.add_argument("input", help="测试用例 JSON 文件（数组，或含 test_cases 字段的对象）")
    ap.add_argument("--base", default="", help="接口基础地址，如 http://localhost:8000（url 为相对路径时拼接）")
    ap.add_argument("--cover", default="", metavar="MODULE.py",
                    help="被测纯函数模块 .py 路径，启用代码覆盖率采集 + 盲区报告")
    ap.add_argument("--entry", default="", help="被测函数限定名（--cover 模式下，缺省用用例 _entry）")
    ap.add_argument("--coverage-report", default="", metavar="FILE",
                    help="把覆盖报告（JSON）写入指定文件")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    # 兼容两种格式：纯数组，或 {"test_cases": {entry: [...]}}
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        tc = data.get("test_cases", {})
        cases = []
        for entry, arr in tc.items():
            for c in arr:
                c = dict(c)
                c.setdefault("_entry", entry)
                cases.append(c)
    else:
        print("无法识别的输入格式")
        sys.exit(1)

    # ---- 覆盖模式：纯函数执行 + 覆盖率 ----
    if args.cover:
        report_text, report_dicts = _run_coverage(cases, args.cover, args.entry)
        print(report_text)
        if args.coverage_report:
            Path(args.coverage_report).write_text(
                json.dumps(report_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n覆盖报告已写入: {args.coverage_report}")
        # 纯函数覆盖：有盲区不算失败（盲区是改进信号），退出码 0
        sys.exit(0)

    # ---- HTTP 模式（默认） ----
    print(f"共 {len(cases)} 条测试用例，执行中...\n")
    stats = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIPPED": 0, "ERROR": 0, "PENDING": 0}
    for c in cases:
        # 纯函数用例（有 call_args）在无 --cover 时也提示需要 --cover
        if "call_args" in c:
            r = _exec_func_one(c)
            print(f"⏭️ [{r['verdict']}] {c.get('_entry','')} {r.get('description','?')}  (需 --cover 采集覆盖)")
            stats[r.get("verdict", "SKIPPED")] = stats.get(r.get("verdict", "SKIPPED"), 0) + 1
            continue
        r = execute_http_case({**c, "_entry": c.get("_entry", "")}, args.base)
        entry = c.get("_entry", "")
        verdict = r.get("verdict", r.get("status", "?"))
        stats[verdict] = stats.get(verdict, 0) + 1
        mark = verdict_mark(verdict)
        print(f"{mark} [{verdict}] {entry} {r.get('description', '?')}  (HTTP {r.get('status')})")
        reason = r.get("reason")
        if reason:
            print(f"       {reason}")
        if r.get("response_snippet"):
            print(f"       响应: {r['response_snippet']}")
        if r.get("error"):
            print(f"       错误: {r['error']}")
    print(f"\n完成：PASS {stats['PASS']} / FAIL {stats['FAIL']} / WARN {stats['WARN']} "
          f"/ SKIPPED {stats['SKIPPED']} / ERROR {stats['ERROR']}（共 {len(cases)}）")
    if stats["FAIL"] > 0:
        sys.exit(1)   # 有失败用例，返回非零退出码（便于 CI）


if __name__ == "__main__":
    main()
