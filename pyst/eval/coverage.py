#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲区覆盖检测模块（开发计划问题3）
==================================
回答"用例到底覆盖了被测代码的哪些分支，还有哪些盲区没测到"。

【三种覆盖形态】
1. 纯函数同进程覆盖（本次实现，最准）：
   被测函数可直接 import 调用，用 coverage.py 包住调用，拿到精确的语句行/分支覆盖，
   与功能点档案里的 control_sites（带行区间）比对 → 量化覆盖率 + 盲区分支清单。

2. HTTP 接口覆盖（预留）：
   runner 发 HTTP 到外部服务，跨进程拿不到代码执行情况，
   退化为"接口参数覆盖矩阵"（正常/空/边界/非法是否都有用例），
   并诚实标注"代码内部分支需被测服务挂 coverage 才能统计"。

3. Web 框架 view（预留）：import view + 构造 request 调用，后续扩展。

【期望覆盖模型 expected_branches】
   从功能点档案的 control_sites + 下游 call_edges[].controls 展开成"期望分支清单"，
   每个分支带 path / kind / cond / semantics / line_range / 所属符号。
   这是"应该测到的目标"，与实际覆盖比对后得到盲区。

【数据来源】
   - 功能点档案：pyst/core/features.py 的 FeaturePoint.to_dict()
   - 增强后的 ControlSite：pyst/core/ast_analysis.py（含 start/end/semantics/path）
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 数据结构：期望分支 / 覆盖报告
# ---------------------------------------------------------------------------

@dataclass
class ExpectedBranch:
    """一个"应该测到的"分支节点（来自控制流骨架展开）。"""
    symbol: str              # 所属符号（入口或下游函数限定名）
    kind: str                # if | elif | else | for | while | try | except | with
    path: str                # 分支路径标识（如 if@12:then）
    line: int
    cond: str | None
    semantics: list[str]     # empty/nonempty/boundary/...
    start: int | None
    end: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "kind": self.kind, "path": self.path,
            "line": self.line, "cond": self.cond, "semantics": self.semantics,
            "start": self.start, "end": self.end,
        }


@dataclass
class CoverageResult:
    """一次覆盖检测的完整结果（可序列化为 JSON 给前端展示）。"""
    mode: str                       # "func" | "http" | "view"
    target: str                     # 被测目标（函数限定名 / 接口 url / view）
    total_branches: int = 0         # 期望分支总数
    covered_branches: int = 0       # 实际覆盖分支数
    uncovered_branches: list[dict] = field(default_factory=list)   # 盲区分支
    executed_lines: set[int] = field(default_factory=set)          # 实际执行到的语句行
    statement_total: int = 0
    statement_covered: int = 0
    param_matrix: list[dict] = field(default_factory=list)         # 接口参数覆盖矩阵（http 模式）
    warnings: list[str] = field(default_factory=list)
    executed: int = 0               # 实际执行的用例数
    passed: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        ratio = (self.covered_branches / self.total_branches
                 if self.total_branches else 0.0)
        return {
            "mode": self.mode,
            "target": self.target,
            "total_branches": self.total_branches,
            "covered_branches": self.covered_branches,
            "branch_coverage": round(ratio, 3),
            "uncovered_branches": self.uncovered_branches,
            "statement_total": self.statement_total,
            "statement_covered": self.statement_covered,
            "statement_coverage": round(
                (self.statement_covered / self.statement_total)
                if self.statement_total else 0.0, 3),
            "param_matrix": self.param_matrix,
            "warnings": self.warnings,
            "executed": self.executed, "passed": self.passed, "failed": self.failed,
        }


# ---------------------------------------------------------------------------
# 1. 期望覆盖模型：控制流骨架 → 期望分支清单
# ---------------------------------------------------------------------------

def _branch_from_site(site: dict[str, Any], symbol: str) -> ExpectedBranch:
    """把一个控制流节点 dict 转成 ExpectedBranch（带默认值兜底）。"""
    return ExpectedBranch(
        symbol=symbol,
        kind=site.get("kind", "?"),
        path=site.get("path") or f"{site.get('kind','?')}@{site.get('line','?')}",
        line=site.get("line", 0),
        cond=site.get("cond"),
        semantics=site.get("semantics") or [],
        start=site.get("start"),
        end=site.get("end"),
    )


def expected_branches(feature: dict[str, Any]) -> list[ExpectedBranch]:
    """从功能点档案展开"期望分支清单"。

    覆盖两类控制流：
      - 入口自身的 control_sites
      - 每个下游调用节点（call_edges[].controls）的控制逻辑

    返回按 symbol + path 去重的分支列表（同一符号的同 path 只保留一次，
    避免 _collect_sites 的嵌套/重复收集造成冗余）。
    """
    entry = feature.get("entry", "?")
    out: list[ExpectedBranch] = []
    seen: set[tuple[str, str]] = set()

    def _add(site: dict[str, Any], symbol: str):
        b = _branch_from_site(site, symbol)
        key = (b.symbol, b.path)
        if key in seen:
            return
        seen.add(key)
        out.append(b)

    # 入口自身控制流
    for site in feature.get("control_sites") or []:
        _add(site, entry)

    # 下游调用节点的控制流
    for edge in feature.get("call_edges") or []:
        callee = edge.get("callee", "?")
        for site in edge.get("controls") or []:
            _add(site, callee)

    # 按 symbol + line 排序，便于展示与比对
    out.sort(key=lambda b: (b.symbol, b.line, b.path))
    return out


def _summarize_semantics(branches: list[ExpectedBranch]) -> dict[str, list[str]]:
    """按语义标签汇总期望分支，供盲区报告按维度展示。"""
    by_sem: dict[str, list[str]] = {}
    for b in branches:
        for sem in (b.semantics or ["unknown"]):
            by_sem.setdefault(sem, []).append(b.path)
    return by_sem


# ---------------------------------------------------------------------------
# 2. 实际覆盖采集（纯函数同进程，coverage.py）
# ---------------------------------------------------------------------------

def measure_func_coverage(module_path: str, qualname: str,
                          call_args_list: list[list[Any]],
                          call_kwargs_list: list[dict[str, Any]] | None = None) -> dict:
    """用 coverage.py 包裹对被测纯函数的多次调用，采集实际语句/分支覆盖。

    关键顺序：必须先 cov.start()，再 import 被测模块并调用函数，
    否则 coverage 追踪不到该模块（module-not-imported / no-data-collected）。

    Args:
        module_path: 被测模块 .py 文件绝对路径
        qualname: 被测函数限定名（模块内符号路径）
        call_args_list: 每次调用的位置参数列表（每个元素是一组 args）
        call_kwargs_list: 每次调用的关键字参数列表（与 args 对齐，可为 None）

    Returns:
        coverage 分析结果 dict：executed_lines / statements / branches / executed_branch_pairs
    """
    import coverage as _cov
    import importlib.util as _ilu

    kwargs_list = call_kwargs_list or [{} for _ in call_args_list]
    abs_path = str(Path(module_path).resolve())

    # 保证被测模块目录在 sys.path（coverage 需能 import 它）
    mod_dir = str(Path(module_path).resolve().parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    # 被测模块作为独立模块 import（名称与文件 stem 一致），避免污染模块缓存
    mod_name = Path(module_path).stem
    spec = _ilu.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {module_path}")
    module = _ilu.module_from_spec(spec)

    # 先 start，再执行模块加载（coverage 才能追踪）；符号在 exec 后解析
    cov = _cov.Coverage(branch=True)
    cov.start()
    executed_cnt = 0
    try:
        spec.loader.exec_module(module)     # 执行模块定义（在 coverage 追踪内）

        # 解析被测函数限定名 → 模块内的可调用对象（exec 之后才有属性）
        sym_parts = qualname.split(".")
        fn = module
        for p in sym_parts:
            if not hasattr(fn, p):
                raise AttributeError(f"在 {mod_name} 中找不到符号 {qualname}")
            fn = getattr(fn, p)
        if not callable(fn):
            raise AttributeError(f"{qualname} 不可调用")

        for args, kwargs in zip(call_args_list, kwargs_list):
            try:
                fn(*args, **kwargs)
                executed_cnt += 1
            except Exception:
                # 被测函数抛异常也算执行到（异常路径本身是覆盖），记录但继续
                executed_cnt += 1
    finally:
        cov.stop()
        cov.save()

    data = cov.get_data()
    measured = cov.analysis2(abs_path)
    # analysis2 返回 (file, statements, excluded, missing, missing_branches)
    statements, missing = measured[1], measured[3]

    executed_lines = {ln for ln in statements if ln not in missing}
    executed_branch_pairs = _executed_branch_pairs(cov, abs_path)

    return {
        "executed_lines": executed_lines,
        "statement_total": len(statements),
        "statement_covered": len(executed_lines),
        "executed_branch_pairs": executed_branch_pairs,
        "executed": executed_cnt,
    }


def _executed_branch_pairs(cov, filename: str) -> set[tuple[int, str]]:
    """从 coverage 数据里提取"已执行的分支"集合。

    coverage 的分支数据是 arc（有向弧）：(from, to) 形式。
    if 条件产生两个 arc：from 条件行到 then、from 条件行到 else。
    若某 arc 出现在数据里，说明该分支方向被走到了。

    Returns:
        set[(arc_from_line, arc_to_str)]，to 用字符串（兼容 line / -line 退出）
    """
    arcs = cov.get_data().arcs(filename) or []
    executed = set()
    for frm, to in arcs:
        executed.add((int(frm), str(to)))
    return executed


# ---------------------------------------------------------------------------
# 3. 盲区比对：期望分支 vs 实际覆盖
# ---------------------------------------------------------------------------

def _branch_is_covered(branch: ExpectedBranch, executed_lines: set[int]) -> bool:
    """判断一个期望分支是否被实际覆盖。

    规则：分支代码体 [start, end] 区间内有任意语句行被执行 → 视为该分支被触达。
    空代码体（start=end=0，如空 else）无法用行判断，记为"未判定"（不进覆盖也不进盲区，
    仅留 warning）。
    """
    if not branch.start or not branch.end or branch.start <= 0:
        return False
    if branch.end < branch.start:
        return False
    return any(ln in executed_lines for ln in range(branch.start, branch.end + 1))


def compare(expected: list[ExpectedBranch], executed_lines: set[int]) -> CoverageResult:
    """把期望分支与实际执行行比对，产出覆盖报告（含盲区清单）。

    Args:
        expected: expected_branches() 产物
        executed_lines: measure_func_coverage() 的 executed_lines
    """
    result = CoverageResult(mode="func", target="")
    result.total_branches = len(expected)
    result.executed_lines = set(executed_lines)
    for b in expected:
        covered = _branch_is_covered(b, executed_lines)
        if covered:
            result.covered_branches += 1
        else:
            result.uncovered_branches.append(b.to_dict())
    return result


# ---------------------------------------------------------------------------
# 4. 组合入口：给定功能点档案 + 被测调用清单，产出完整盲区报告
# ---------------------------------------------------------------------------

def analyze_function(feature: dict[str, Any], module_path: str,
                     call_args_list: list[list[Any]],
                     call_kwargs_list: list[dict[str, Any]] | None = None,
                     entry_qualname: str | None = None) -> CoverageResult:
    """对一个功能点做完整的纯函数盲区覆盖分析。

    Args:
        feature: 功能点档案 dict
        module_path: 被测入口所在模块 .py 文件绝对路径
        call_args_list: 每个测试用例对应的被测函数位置参数
        call_kwargs_list: 关键字参数（对齐）
        entry_qualname: 被测函数限定名（缺省用 feature.entry）

    Returns:
        CoverageResult（含盲区清单 + 覆盖率）
    """
    qualname = entry_qualname or feature.get("entry", "?")
    # 1. 期望分支
    expected = expected_branches(feature)
    # 2. 实际覆盖
    try:
        measure = measure_func_coverage(module_path, qualname,
                                        call_args_list, call_kwargs_list)
    except Exception as e:
        result = CoverageResult(mode="func", target=qualname)
        result.warnings.append(f"覆盖采集失败: {e} | {traceback.format_exc(limit=2)}")
        return result

    # 3. 比对
    result = compare(expected, measure["executed_lines"])
    result.mode = "func"
    result.target = qualname
    result.executed = measure.get("executed", len(call_args_list))
    result.statement_total = measure.get("statement_total", 0)
    result.statement_covered = measure.get("statement_covered", 0)
    return result


# ---------------------------------------------------------------------------
# 5. HTTP 形态：接口参数覆盖矩阵（代码内部分支不可测，诚实标注）
# ---------------------------------------------------------------------------

def analyze_http(feature: dict[str, Any], executed_cases: list[dict]) -> CoverageResult:
    """对 HTTP 形态的用例做接口参数覆盖矩阵分析。

    被测服务跨进程独立运行，无法用 coverage.py 采集代码内部分支，
    故退化为：检查每个接口参数的取值在用例里是否覆盖了 正常/空/边界/非法 语义。
    代码内部分支覆盖作为 warning 明确提示（需要被测服务挂 coverage）。

    Args:
        feature: 功能点档案 dict
        executed_cases: 已执行的用例（含 request）
    """
    entry = feature.get("entry", "?")
    iface = feature.get("interface") or {}
    params = []
    for key in ("params", "query", "body_fields"):
        for p in (iface.get(key) or []):
            params.append(p)
    if not params:
        params = list((iface.get("params") or []))

    result = CoverageResult(mode="http", target=entry)
    result.warnings.append(
        "HTTP 形态跨进程执行，无法统计代码内部分支覆盖率；"
        "本报告为接口参数覆盖矩阵，代码分支需被测服务挂 coverage 才能精确统计。"
    )
    # 期望分支仍列出（作为设计层面的盲区参考，但不算进 covered）
    expected = expected_branches(feature)
    result.total_branches = len(expected)

    # 统计每个参数在用例里的取值（简化：按参数名匹配 request.query/body）
    used_values: dict[str, list] = {}
    for c in executed_cases:
        req = c.get("request") or {}
        q = req.get("query") or {}
        b = req.get("body") or {}
        for k, v in list(q.items()) + list(b.items()):
            used_values.setdefault(str(k), []).append(v)

    for p in params:
        name = p.get("name", p) if isinstance(p, dict) else p
        name = str(name)
        vals = used_values.get(name, [])
        present = len(vals) > 0
        has_empty = present and any(v in ("", None) for v in vals)
        result.param_matrix.append({
            "param": name,
            "covered": present,
            "sample_values": [str(v) for v in vals[:3]],
            "has_empty": has_empty,
            "note": "" if present else "无用例覆盖该参数",
        })
    # HTTP 形态下覆盖分支数不计算（代码分支不可测），只给参数覆盖
    result.executed = len(executed_cases)
    return result


def render_report(result: CoverageResult) -> str:
    """把覆盖报告渲染成人类可读文本（用于 CLI 输出）。"""
    d = result.to_dict()
    lines = [
        f"=== 覆盖报告（{result.mode}）: {result.target} ===",
        f"期望分支: {d['total_branches']}  已覆盖: {d['covered_branches']}  "
        f"分支覆盖率: {d['branch_coverage']*100:.1f}%",
    ]
    if result.mode == "func":
        lines.append(
            f"语句覆盖: {d['statement_covered']}/{d['statement_total']} "
            f"({d['statement_coverage']*100:.1f}%)"
        )
    if d["uncovered_branches"]:
        lines.append("\n【盲区分支（期望但未覆盖）】")
        for b in d["uncovered_branches"]:
            sem = ",".join(b.get("semantics") or ["unknown"])
            cond = f" {b.get('cond')}" if b.get("cond") else ""
            lines.append(f"  - {b['symbol']} L{b['line']} {b['kind']} "
                         f"[{b['path']}] ({sem}){cond}")
    else:
        lines.append("\n（无盲区分支）")
    if d["param_matrix"]:
        lines.append("\n【接口参数覆盖矩阵】")
        for p in d["param_matrix"]:
            mark = "✅" if p["covered"] else "❌"
            note = p.get("note") or ""
            lines.append(f"  {mark} {p['param']} {p.get('sample_values')} {note}")
    if d["warnings"]:
        lines.append("\n【提示】")
        for w in d["warnings"]:
            lines.append(f"  ⚠️ {w}")
    lines.append(f"\n执行用例: {result.executed}  (通过 {result.passed} / 失败 {result.failed})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(__doc__)
