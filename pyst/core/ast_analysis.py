#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0：文件级 AST 结构分析器（原 ast_analyzer.py）
=====================================================
演示如何把 Python 标准库 ast 从"单句玩具"升级为"文件级 / 项目级"分析工具。

核心认知（回答"为什么要一次解析整个文件"）：
  1. ast.parse() 一次调用就能解析整个文件，得到一棵以 Module 为根的完整语法树。
     树里每个节点都自带行号 (lineno / end_lineno) 和父子嵌套关系。
     —— 这就是"整个文件"的全部结构信息，根本不需要逐句解析。
  2. 逐句解析（exec/compile 单句）会丢失：
        - 类 / 函数之间的嵌套层级（LoggingCallbackHandler 属于谁？）
        - import 与使用它们的代码的上下文
        - 每个符号的行号坐标（后面做 hunk→符号 映射全靠它）
  3. "分析整个项目" = for 每个 .py 文件: ast.parse + 遍历。
     跨文件的符号关系（谁 import 了谁、谁调用了谁）是更上层的问题，
     由后面的调用图模块负责 —— 但地基就是这里。

用法:
    python -m pyst <文件.py | 目录> --phase 0     # 包统一入口
    python phase0_file.py <文件.py | 目录>        # 模块内直接运行
    例: python -m pyst ../TestBrain/apps/llm/callbacks.py --phase 0
    例: python -m pyst ../TestBrain/apps/llm --phase 0   # 目录=批量统计所有 .py
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------- 数据结构（这就是"分析的产物"，也是后面调用图/映射的输入） ----------

@dataclass
class CallSite:
    """一次函数调用点：从哪一行、以什么表达式调用了谁"""
    line: int
    qualname: str       # 完整调用表达式，如 self._get_run_id(...) 里的 self._get_run_id
    name: str           # 最右侧名字，如 _get_run_id


_IF_BRANCH_ALIASES = {
    # 常见"空值 / 非空"条件 → 语义标签（供盲区覆盖推断）
    "empty":   ("is None", "== None", "== ''", "== \"\"", "not ", "len("),
    "nonempty": ("is not None", "!= None", "!= ''", "!= \"\"", "not None", " and "),
}


def _infer_cond_semantics(cond: str | None, kind: str) -> list[str]:
    """从条件表达式原文推断分支语义标签。

    目的：给"盲区覆盖"提供"应该测什么输入"的线索。
      - if x is None / == "" / not x       → empty（空值必测）
      - if x is not None / != ""           → nonempty（非空必测）
      - if x > n / x < n / >= / <=         → boundary（边界值必测）
      - if x == y                          → equality（相等性必测）
      - else / finally / 其他              → 兜底 unknown（诚实标注，不硬编）
    推断基于文本启发式，无法覆盖所有情况，故 unknown 作为兜底而不臆造语义。

    Args:
        cond: 条件表达式原文（if/elif/while/except 有；for 是迭代对象；else/finally 为 None）
        kind: 控制流节点类型

    Returns:
        语义标签列表（可多个，如 ['nonempty', 'boundary']）
    """
    if not cond:
        return []
    s = cond.strip()
    if not s:
        return []
    if kind == "except":
        return ["exception"]
    if kind == "for":
        return ["iteration"]
    tags: list[str] = []
    # 空值 / 非空
    lowered = s
    if any(tok in lowered for tok in _IF_BRANCH_ALIASES["empty"]):
        tags.append("empty")
    if any(tok in lowered for tok in _IF_BRANCH_ALIASES["nonempty"]):
        tags.append("nonempty")
    # 边界
    if any(op in s for op in ("<=", ">=", "<", ">")):
        tags.append("boundary")
    # 相等 / 不等
    if "==" in s:
        tags.append("equality")
    if "!=" in s:
        tags.append("inequality")
    # 异常分支通常意味着"非法输入"路径
    if kind == "elif" or kind == "else":
        tags.append("alternative")
    if not tags:
        tags.append("unknown")
    return tags


@dataclass
class ControlSite:
    """一个控制流节点：函数体里的分支/循环/异常处理/资源管理骨架。

    这是测试用例场景推断的关键线索（M2）：
      - if x is None: raise   → "传 None 必测"
      - for 循环              → "0 次 / 1 次 / N 次边界"
      - try/except            → "异常路径"
      - with                  → "资源管理 / 上下文进入退出"
    与行号区间配合，还能重建"哪个 if 包裹了哪个调用"的嵌套关系。

    M8 增强（盲区覆盖）：
      - start/end：该分支代码体的行区间，用于和 coverage.py 的语句行号对齐，
        从而判断"这条分支实际被执行到了没有"。
      - semantics：从 cond 推断的语义标签（empty/nonempty/boundary/...），
        用于生成"期望覆盖清单"和盲区报告。
      - path：分支路径标识（如 "if@12:then" / "if@12:else" / "for@15"），
        作为该分支在期望覆盖模型里的稳定 key。
    """
    kind: str           # if | elif | else | for | while | try | except | finally | with
    line: int
    cond: str | None    # 条件表达式原文（if/elif/while/except 有；for 是迭代对象；else/finally 为 None）
    branch: str | None  # 分支方向：then | else（if 有效）；None（for/while/try/with）
    start: int | None = None    # 该分支代码体起始行（覆盖对齐）
    end: int | None = None      # 该分支代码体结束行（覆盖对齐）
    semantics: list[str] = field(default_factory=list)  # 语义标签（盲区覆盖）
    path: str | None = None     # 分支路径标识 key

    def __post_init__(self):
        # 给每种控制节点生成稳定路径 key + 语义标签（若未显式传入）
        kind = self.kind
        if self.branch:
            self.path = f"{kind}@{self.line}:{self.branch}"
        elif kind in ("if", "elif"):
            self.path = f"{kind}@{self.line}:then"
        else:
            self.path = f"{kind}@{self.line}"
        if not self.semantics:
            self.semantics = _infer_cond_semantics(self.cond, kind)


@dataclass
class FunctionInfo:
    name: str
    kind: str               # "function" | "method" | "async method"
    args: list[str]         # 格式化后的参数列表，如 ["self", "value", "limit=500"]
    returns: str | None     # 返回类型注解（源码原样）
    start: int
    end: int
    docstring: str | None
    calls: list[CallSite]   # 函数体内直接出现的调用
    controls: list[ControlSite]   # 函数体内控制流骨架（M2）
    decorators: list[str]


@dataclass
class ClassInfo:
    name: str
    bases: list[str]        # 基类名，如 ["BaseCallbackHandler"]
    start: int
    end: int
    docstring: str | None
    methods: list[FunctionInfo]   # list[FunctionInfo]


@dataclass
class FileInfo:
    path: str
    module_doc: str | None
    imports: list[str] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    top_functions: list[FunctionInfo] = field(default_factory=list)


# ---------- 辅助函数 ----------

def _doc(node) -> str | None:
    """取 docstring 首行，超长截断"""
    d = ast.get_docstring(node)
    if not d:
        return None
    first = d.splitlines()[0]
    return first[:50] + "..." if len(first) > 50 else first


def _format_args(node) -> list[str]:
    """把函数的 args 节点格式化成人类可读的参数列表"""
    a = node.args
    out = []
    posonly = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(posonly) - len(a.defaults)) + [ast.unparse(d) for d in a.defaults]
    for arg, default in zip(posonly, defaults):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            s += f"={default}"
        out.append(s)
    for arg in a.kwonlyargs:
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        out.append(s + "=...")  # kwonly 必须有关键字形式
    if a.vararg:
        out.append(f"*{a.vararg.arg}")
    if a.kwarg:
        out.append(f"**{a.kwarg.arg}")
    return out


def _node_line_range(body_nodes) -> tuple[int, int]:
    """计算一组 AST 语句节点的 [起始行, 结束行] 区间（用于分支覆盖对齐）。

    若列表为空（如空 orelse/finalbody），返回 (0, 0) 表示"无代码体"。
    """
    if not body_nodes:
        return (0, 0)
    start = min(getattr(n, "lineno", 0) for n in body_nodes)
    end = max(getattr(n, "end_lineno", getattr(n, "lineno", 0)) for n in body_nodes)
    return (start, end)


def _collect_sites(fn_node) -> tuple[list[CallSite], list[ControlSite]]:
    """收集一个函数体里的调用点 + 控制流骨架（不含嵌套子函数内部）。

    用 ast.walk 从函数体的每个顶层语句向下展开，天然覆盖 if/for/while/try/with，
    且不会漏掉嵌套在分支/循环体内的 Call 与控制流节点。嵌套子函数（function body
    里的 FunctionDef）的 walk 子树会被跳过，因为它们属于另一个作用域。
    """
    calls: list[CallSite] = []
    controls: list[ControlSite] = []
    consumed: set[ast.If] = set()                # 已作为 elif 处理过的 If 节点（避免重复记录）
    for child in fn_node.body:
        for n in ast.walk(child):
            if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
                continue                       # 跳过嵌套子函数，避免跨作用域误收集
            if isinstance(n, ast.Call):
                qual = ast.unparse(n.func)
                calls.append(CallSite(n.lineno, qual, qual.split(".")[-1]))
            elif isinstance(n, ast.If):
                if n in consumed:               # 已作为某 if 的 elif 记录过，跳过
                    continue
                body_range = _node_line_range(n.body)
                controls.append(ControlSite("if", n.lineno, ast.unparse(n.test), "then",
                                            start=body_range[0], end=body_range[1]))
                for orelse in n.orelse:
                    # else 或 elif：orelse 里若又是 If，则是 elif（后续 walk 会再碰到它，先标记）
                    if isinstance(orelse, ast.If):
                        or_range = _node_line_range([orelse])
                        controls.append(ControlSite("elif", orelse.lineno, ast.unparse(orelse.test), "then",
                                                    start=or_range[0], end=or_range[1]))
                        consumed.add(orelse)
                    else:
                        or_range = _node_line_range([orelse])
                        controls.append(ControlSite("else", orelse.lineno, None, "else",
                                                    start=or_range[0], end=or_range[1]))
            elif isinstance(n, ast.For):
                body_range = _node_line_range(n.body)
                controls.append(ControlSite("for", n.lineno, ast.unparse(n.iter), None,
                                            start=body_range[0], end=body_range[1]))
            elif isinstance(n, ast.While):
                body_range = _node_line_range(n.body)
                controls.append(ControlSite("while", n.lineno, ast.unparse(n.test), None,
                                            start=body_range[0], end=body_range[1]))
            elif isinstance(n, ast.Try):
                controls.append(ControlSite("try", n.lineno, None, None,
                                            start=n.lineno, end=getattr(n, "end_lineno", n.lineno)))
                for handler in n.handlers:
                    t = ast.unparse(handler.type) if handler.type else None
                    h_range = _node_line_range(handler.body)
                    controls.append(ControlSite("except", handler.lineno, t, None,
                                                start=h_range[0], end=h_range[1]))
                if n.orelse:
                    or_range = _node_line_range(n.orelse)
                    controls.append(ControlSite("else", n.orelse[0].lineno, None, "else",
                                                start=or_range[0], end=or_range[1]))
                if n.finalbody:
                    fb_range = _node_line_range(n.finalbody)
                    controls.append(ControlSite("finally", n.finalbody[0].lineno, None, None,
                                                start=fb_range[0], end=fb_range[1]))
            elif isinstance(n, ast.With):
                # n.items 是 list[withitem]，取第一个的 context_expr（如 ThreadPoolExecutor(...)）
                expr = n.items[0].context_expr if n.items else None
                body_range = _node_line_range(n.body)
                controls.append(ControlSite("with", n.lineno, ast.unparse(expr) if expr else None, None,
                                            start=body_range[0], end=body_range[1]))
    return calls, controls


def _parse_function(node, kind: str) -> FunctionInfo:
    decorators = [ast.unparse(d) for d in node.decorator_list]
    returns = ast.unparse(node.returns) if node.returns else None
    calls, controls = _collect_sites(node)
    return FunctionInfo(
        name=node.name,
        kind=kind,
        args=_format_args(node),
        returns=returns,
        start=node.lineno,
        end=getattr(node, "end_lineno", node.lineno),
        docstring=_doc(node),
        calls=calls,
        controls=controls,
        decorators=decorators,
    )


# ---------- 核心：一次遍历收集整棵文件树 ----------

class FileVisitor(ast.NodeVisitor):
    """只遍历"表层结构"，方法体内部用 _collect_calls 单独处理，避免重复"""

    def __init__(self, info: FileInfo):
        self.info = info

    def visit_Import(self, node):
        for alias in node.names:
            s = f"import {alias.name}"
            if alias.asname:
                s += f" as {alias.asname}"
            self.info.imports.append(s)

    def visit_ImportFrom(self, node):
        base = "." * node.level + (node.module or "")
        for alias in node.names:
            s = f"from {base} import {alias.name}"
            if alias.asname:
                s += f" as {alias.asname}"
            self.info.imports.append(s)

    def visit_ClassDef(self, node):
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(_parse_function(child, "method"))
            elif isinstance(child, ast.AsyncFunctionDef):
                methods.append(_parse_function(child, "async method"))
        self.info.classes.append(ClassInfo(
            name=node.name,
            bases=[ast.unparse(b) for b in node.bases],
            start=node.lineno,
            end=getattr(node, "end_lineno", node.lineno),
            docstring=_doc(node),
            methods=methods,
        ))
        # 不调用 generic_visit：类内细节已在上面手动处理

    def visit_FunctionDef(self, node):
        self.info.top_functions.append(_parse_function(node, "function"))

    def visit_AsyncFunctionDef(self, node):
        self.info.top_functions.append(_parse_function(node, "async function"))


# ---------- 入口：文件级 / 项目级 ----------

def analyze_file(path: Path) -> FileInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))   # ← 一次解析整个文件
    info = FileInfo(str(path), _doc(tree))
    FileVisitor(info).visit(tree)
    return info


# 遍历时排除的目录（与 phase1_index 保持一致，避免把 venv 里的第三方库当被测代码）
_EXCLUDE_DIRS = {
    ".venv", "venv", "env", ".env", "node_modules", "site-packages", "dist-packages",
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", "build", "dist", ".eggs",
}


def iter_py_files(root: Path):
    """遍历 .py 文件，排除虚拟环境/第三方/版本控制目录（os.walk 剪枝，高效）。"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _EXCLUDE_DIRS and not d.endswith(".egg-info") and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def analyze_dir(path: Path):
    """项目级：遍历所有 .py（排除 venv/第三方），每个文件走同一条 analyze_file 管线"""
    results = []
    for py in sorted(iter_py_files(path)):
        try:
            results.append(analyze_file(py))
        except (SyntaxError, RecursionError, MemoryError) as e:
            print(f"  [跳过] {py}: {type(e).__name__}")
    return results


# ---------- 渲染 ----------

def render_file(info: FileInfo, indent: str = "    "):
    print(f"文件: {info.path}")
    print(f"{indent}├─ 模块docstring: {info.module_doc or '无'}")
    print(f"{indent}├─ imports ({len(info.imports)}):")
    for i, imp in enumerate(info.imports):
        branch = "├──" if i < len(info.imports) - 1 else "└──"
        print(f"{indent}│   {branch} {imp}")
    print(f"{indent}├─ 类 ({len(info.classes)}):")
    for ci, cls in enumerate(info.classes):
        cbranch = "├──" if ci < len(info.classes) - 1 else "└──"
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        print(f"{indent}│   {cbranch} 类 {cls.name}{bases}  [行 {cls.start}-{cls.end}]")
        if cls.docstring:
            print(f"{indent}│   │     说明: {cls.docstring}")
        for mi, m in enumerate(cls.methods):
            mbranch = "├──" if mi < len(cls.methods) - 1 else "└──"
            ret = f" -> {m.returns}" if m.returns else ""
            deco = f"  @{', @'.join(m.decorators)}" if m.decorators else ""
            print(f"{indent}│   │   {mbranch} {m.kind} {m.name}({', '.join(m.args)}){ret}  [行 {m.start}-{m.end}]{deco}")
            if m.controls:
                for cs in m.controls:
                    cond = f" {cs.cond}" if cs.cond else ""
                    branch = f" [{cs.branch}]" if cs.branch else ""
                    print(f"{indent}│   │   │   流 {cs.line}: {cs.kind}{cond}{branch}")
            if m.calls:
                for c in m.calls:
                    print(f"{indent}│   │   │   调 {c.line}: {c.qualname}()")
    if info.top_functions:
        print(f"{indent}├─ 模块级函数 ({len(info.top_functions)}):")
        for m in info.top_functions:
            print(f"{indent}│   └── {m.kind} {m.name}({', '.join(m.args)})  [行 {m.start}-{m.end}]")


def render_summary(results: list[FileInfo]):
    """目录模式的统计摘要"""
    print(f"\n{'='*60}\n项目级统计 ({len(results)} 个 .py 文件)\n{'='*60}")
    total = {"classes": 0, "methods": 0, "funcs": 0, "lines": 0}
    for info in results:
        methods = sum(len(c.methods) for c in info.classes)
        total["classes"] += len(info.classes)
        total["methods"] += methods
        total["funcs"] += len(info.top_functions)
        total["lines"] += sum(
            c.end - c.start + 1 for c in info.classes
        ) + sum(f.end - f.start + 1 for f in info.top_functions)
    print(f"  类: {total['classes']:<5} 方法: {total['methods']:<5} "
          f"模块级函数: {total['funcs']:<5} 符号覆盖行数: {total['lines']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_file():
        render_file(analyze_file(target))
    elif target.is_dir():
        infos = analyze_dir(target)
        for info in infos:
            render_file(info)
        render_summary(infos)
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
