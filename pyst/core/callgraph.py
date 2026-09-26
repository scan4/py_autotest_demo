#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2：跨文件调用图（原 call_graph.py）
==========================================
在 Phase 1 项目索引之上，把"调用点"配对成"调用边"。这是 AST 静态分析管线的最后一环。

【本阶段回答】"谁在哪个文件哪一行，调用了谁的哪个函数？"
【输入】ProjectIndex 的三个事实集：
  - 符号表  symbols          : qualname -> SymbolInfo   （知道项目里有哪些可调目标）
  - import 表 files[].imports : 绑定名 -> 来源模块       （知道名字从哪来）
  - 调用点  files[].calls     : 谁、在哪个函数、哪一行发生了调用
【输出】调用边列表，每条边记录 caller / callee / 行号 / 解析状态

调用点 -> 目标符号的解析策略（按优先级，每步都可能失败）：
  1. self.x / cls.x        -> owner 所属类的方法，沿基类链向上回溯
  2. 裸名 x                -> ① 本模块的同名模块级符号  ② import 绑定名指向的本地符号
  3. 带点路径 a.b.c.foo()  -> ① 最长前缀命中本地模块名（模块 a.b.c 的 foo）
                              ② 替换最左绑定名后再匹配（import a.b.c as x -> x.foo 指 a.b.c.foo）
  4. 兜底                  -> 第三方库 / 动态调用 / 属性链，如实标记 unresolved

【教学要点】解析不是 100% 精确：
  - 静态分析无法确定运行时 self 的真实类型（多态 / 鸭子类型）
  - 名字可能在函数内被重新绑定、被装饰器包装、通过 getattr 动态取用
  - 第三方库没有本地符号表，无法配对
  所以本工具输出"候选 + 状态"，而不是"唯一真相"。这也是真实静态分析工具（如 pylint / mypy
  的部分功能）的设计取舍。

用法:
    python -m pyst <项目根目录> --phase 2                     # 打印调用图 + 解析统计
    python -m pyst <项目根目录> --phase 2 --json callgraph.json
    python -m pyst <项目根目录> --phase 2 --query get_logger   # 反向查询：谁调用了它
    python -m pyst <项目根目录> --phase 2 --unresolved         # 只看本地该解未解的
    python phase2_callgraph.py <项目根目录> --query get_logger # 模块内直接运行

例:
    python -m pyst ../TestBrain --phase 2 --query get_logger
"""

import argparse
import builtins
import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .project_index import ProjectIndex, CallInfo, resolve_relative_module

_BUILTIN_NAMES = frozenset(dir(builtins))       # isinstance / int / open / getattr ...


# ---------- 数据结构 ----------

@dataclass
class CallEdge:
    """一条调用边：caller 在 line 行调用了 callee"""
    caller: str                     # 调用方限定名：module.Class.method / module.func
    callee: str                     # 目标符号限定名（未解析时为 "?"）
    line: int                       # 调用点行号
    module: str                     # 调用点所在模块
    expression: str                 # 调用表达式原文，如 self._get_run_id
    status: str                     # resolved | ambiguous | unresolved
    candidates: list[str] = field(default_factory=list)   # 歧义时列出全部候选


# ---------- 调用解析器：调用点 -> 目标符号 ----------

class CallResolver:
    """把 CallInfo（调用点）解析成候选目标限定名。

    依赖 Phase 1 的两张表：
      - index.symbols      限定名 -> SymbolInfo（全局符号表）
      - index.module_paths 模块名 -> 文件路径（判断某前缀是不是本地模块）
    """

    def __init__(self, index: ProjectIndex):
        self.index = index
        self._bind_cache: dict[str, dict[str, tuple[str, str, str]]] = {}

    # --- import 绑定表：某模块里，绑定名 -> (形式, 来源模块, 原名) ---
    def _bindings(self, module: str) -> dict[str, tuple[str, str, str]]:
        """把文件的 import 语句摊平成 {绑定名: (kind, src_module, orig_name)}。

        import a.b.c                -> a 绑定 ("import", "a.b.c", "a.b.c")
        import a.b.c as x           -> x 绑定 ("import", "a.b.c", "a.b.c")
        from x import y             -> y 绑定 ("from", "x", "y")
        .utils z        -> z 绑定 ("from", "apps.llm.utils", "z")
        """
        if module in self._bind_cache:
            return self._bind_cache[module]
        out: dict[str, tuple[str, str, str]] = {}
        fi = self.index.files.get(module)
        if fi:
            for imp in fi.imports:
                if imp.kind == "import":
                    for n in imp.names:
                        bound = n.asname or n.name.split(".")[0]
                        out[bound] = ("import", imp.module, n.name)
                else:  # from_import
                    src = resolve_relative_module(imp.level, imp.module, module)
                    if src:
                        for n in imp.names:
                            out[n.asname or n.name] = ("from", src, n.name)
        self._bind_cache[module] = out
        return out

    # --- re-export 解析：from pkg import name 但 name 定义在子模块 ---
    def _reexport_origin(self, src_module: str, name: str,
                         _seen: set[str] | None = None) -> str | None:
        """查 src_module 这个包里 name 的真实来源模块（处理 re-export）。

        典型场景：
            apps/llm/__init__.py:  .base LLMServiceFactory
            其它模块:               from apps.llm import LLMServiceFactory
        这时 src=apps.llm、name=LLMServiceFactory，apps.llm.LLMServiceFactory
        不在符号表，但 apps.llm.base.LLMServiceFactory 在 —— 递归经 __init__ 找到。

        返回真实限定名前缀（如 "apps.llm.base"），查不到返回 None。
        用 _seen 防循环（包 A re-export B、B re-export A）。
        """
        if _seen is None:
            _seen = set()
        if src_module in _seen:
            return None
        _seen.add(src_module)
        # 若该模块本身就是符号定义所在（非纯 re-export 包），直接用原名
        direct = f"{src_module}.{name}"
        if direct in self.index.symbols:
            return src_module
        fi = self.index.files.get(src_module)          # 包的 __init__ 的 imports
        if not fi:
            return None
        for imp in fi.imports:
            if imp.kind == "from_import":
                for n in imp.names:
                    if (n.asname or n.name) != name:
                        continue
                    # 相对导入（.base X / from ..pkg.base import X）：
                    # __init__.py 自身就是包 src_module，".base" 应解析为 src_module.base，
                    # 而非当成普通模块减级（resolve_relative_module 在这里会算错）。
                    # 注意 imp.module 已带前导点（如 ".base"），需先剥离。
                    if imp.level > 0:
                        rel = imp.module.lstrip(".") if imp.module else ""
                        if imp.level == 1:
                            sub = f"{src_module}.{rel}" if rel else src_module
                        else:
                            # level>1：往上跳级后拼子模块
                            parts = src_module.split(".")
                            base = parts[: len(parts) - (imp.level - 1)]
                            sub = ".".join(base + ([rel] if rel else []))
                    else:
                        sub = imp.module
                    if sub:
                        return self._reexport_origin(sub, n.name, _seen)
            else:  # import a.b.c as name 或 import name
                for n in imp.names:
                    if (n.asname or n.name.split(".")[0]) != name:
                        continue
                    return n.name.split(".")[0] if "." in n.name else src_module
        return None

    # --- 对外主入口：一个调用点 -> 一条调用边 ---
    def resolve(self, ci: CallInfo) -> CallEdge:
        expr, module, owner = ci.qualname, ci.module, ci.owner
        if expr.startswith(("self.", "cls.")):
            cands = self._resolve_self(expr, owner)
        elif expr.startswith("super()."):
            cands = []                              # 父类调用，多继承下静态不可确定
        elif "." not in expr:
            cands = self._resolve_bare(expr, module)
        else:
            cands = self._resolve_dotted(expr, module)
        uniq = list(dict.fromkeys(cands))           # 去重且保序
        status = self._classify(expr, module, uniq)
        return CallEdge(owner, uniq[0] if uniq else "?", ci.line,
                        module, expr, status, uniq)

    def _classify(self, expr: str, module: str, uniq: list[str]) -> str:
        """给一次解析结果定状态：本地命中 / 歧义 / 外部 / 未解。

        外部（external）= 内置函数、标准库、第三方库、对象属性链（full_case.get、
        logger.info、'\n'.join）—— 工具本就不该解析它们；
        未解（unresolved）= 本地符号表里没有匹配的裸名调用 —— 可能是本地漏解、
        模块级变量调用或函数内局部函数，是值得关注的真实信号。
        """
        if len(uniq) == 1:
            return "resolved"
        if len(uniq) > 1:
            return "ambiguous"
        first = expr.split(".")[0]
        if first in _BUILTIN_NAMES:                 # isinstance / int / open ...
            return "external"
        bindings = self._bindings(module)
        if first in bindings:
            _, src, _ = bindings[first]
            if src and src not in self.index.module_paths:   # json / os / langchain ...
                return "external"
        if "." in expr:
            # 带点但没解：对象属性链（静态无法确定对象类型）归外部；
            # 若 first 是本地符号（类/函数），则是它的成员调用漏解，值得关注
            if f"{module}.{first}" in self.index.symbols:
                return "unresolved"
            return "external"
        return "unresolved"                         # 裸名本地漏解 —— 真实信号

    # --- 策略 1：self.x / cls.x -> 类方法（含基类链回溯） ---
    def _resolve_self(self, expr: str, owner: str) -> list[str]:
        member = expr.split(".", 1)[1]
        if "." in member:
            return []                               # self.a.b 是属性链，近似不可解
        parts = owner.split(".")
        if len(parts) < 2:
            return []
        cls_qual = ".".join(parts[:-1])             # module.Class
        # BFS 沿基类链查找 member，优先命中本类
        visited: set[str] = set()
        queue: list[str] = [cls_qual]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            q = f"{cur}.{member}"
            if q in self.index.symbols:
                return [q]
            for base in self._bases_of(cur):
                bq = self._resolve_base(base, cur)
                if bq and bq not in visited:
                    queue.append(bq)
        return []

    def _bases_of(self, cls_qual: str) -> list[str]:
        s = self.index.symbols.get(cls_qual)
        return s.bases if s else []

    def _resolve_base(self, base_expr: str, cls_qual: str) -> str | None:
        """把基类表达式（'BaseCallbackHandler' / 'Generic[T]'）解析成限定名。取最左名字。"""
        base_name = base_expr.split("[")[0].split("(")[0].strip().split(".")[0]
        if not base_name or base_name in ("object", "ABC"):
            return None
        module = cls_qual.rsplit(".", 1)[0]         # 类的模块
        q = f"{module}.{base_name}"                 # ① 本模块同名符号
        if q in self.index.symbols:
            return q
        bindings = self._bindings(module)           # ② import 绑定
        if base_name in bindings:
            kind, src, orig = bindings[base_name]
            if kind == "from":
                q2 = f"{src}.{orig}"
                return q2 if q2 in self.index.symbols else None
            q2 = f"{src}.{base_name}"               # import a.b as Base 的近似
            return q2 if q2 in self.index.symbols else None
        return None

    # --- 策略 2：裸名 x -> 本模块符号 / import 绑定 ---
    def _resolve_bare(self, name: str, module: str) -> list[str]:
        out: list[str] = []
        q = f"{module}.{name}"                      # ① 本模块的模块级函数/类
        if q in self.index.symbols:
            out.append(q)
        bindings = self._bindings(module)           # ② import 来的名字
        if name in bindings:
            kind, src, orig = bindings[name]
            if kind == "from":
                q2 = f"{src}.{orig}"
                if q2 in self.index.symbols:
                    out.append(q2)
                else:
                    # re-export：from pkg import name，name 定义在 pkg 的子模块
                    origin = self._reexport_origin(src, orig)
                    if origin:
                        q3 = f"{origin}.{orig}"
                        if q3 in self.index.symbols:
                            out.append(q3)
        return out

    # --- 策略 3：带点路径 a.b.c.foo() ---
    def _resolve_dotted(self, expr: str, module: str) -> list[str]:
        out: list[str] = []
        q = f"{module}.{expr}"                      # ① 模块内符号链：LLMServiceFactory.create
        if q in self.index.symbols:
            out.append(q)
        hit = self._prefix_match(expr)              # ② 最长前缀命中本地模块
        if hit:
            out.append(hit)
        sub = self._substitute(expr, module)        # ③ 替换绑定名后再试
        if sub != expr:
            hit2 = self._prefix_match(sub)
            if hit2:
                out.append(hit2)
        return out

    def _prefix_match(self, qual: str) -> str | None:
        """qual 的最长点前缀是本地模块名，则返回 qual 本身（若它恰好在符号表里）"""
        parts = qual.split(".")
        for i in range(len(parts) - 1, 0, -1):      # 至少留一段符号名
            if ".".join(parts[:i]) in self.index.module_paths:
                return qual if qual in self.index.symbols else None
        # 后缀对齐（7.7.57）：root 不在包根时，import 语句写的模块名（如 app.crud.x）
        # 与索引模块名（backend.app.crud.x，root=项目根多了 backend 前缀）前缀不一致，
        # 上一段循环必然失败 → 本地跨文件调用被误标 external（实测 users.py 的
        # crud.create_user 漏配，全图 resolved 仅 29/843）。解法：点前缀若是某索引
        # 模块名的后缀 → 按该索引模块重写 qual，重写后在符号表命中才采信（自然过滤歧义）
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            for mp in sorted(self.index.module_paths):
                if mp == prefix or mp.endswith("." + prefix):
                    rewritten = mp + qual[len(prefix):]
                    if rewritten in self.index.symbols:
                        return rewritten
        return None

    def _substitute(self, expr: str, module: str) -> str:
        """把 expr 最左绑定名换成来源模块路径（近似启发式）。"""
        first, _, rest = expr.partition(".")
        bindings = self._bindings(module)
        if first not in bindings:
            return expr
        kind, src, orig = bindings[first]
        if kind == "import":
            # import a.b.c [as x]：无别名时原样（前缀匹配已覆盖）；有别名时 x.foo -> a.b.c.foo
            return expr if src == first else f"{src}.{rest}"
        # from x import y [as z]：y.z() -> x.y.z ；y() -> x.y
        base = src
        # re-export：from pkg import y，y 定义在 pkg 子模块时，换成真实来源模块
        if f"{src}.{orig}" not in self.index.symbols:
            origin = self._reexport_origin(src, orig)
            if origin:
                base = origin
        return f"{base}.{orig}" + (f".{rest}" if rest else "")


# ---------- 调用图构建 ----------

class CallGraph:
    def __init__(self, index: ProjectIndex):
        self.index = index
        self.resolver = CallResolver(index)
        self.edges: list[CallEdge] = []

    def build(self) -> "CallGraph":
        for fi in self.index.files.values():
            for ci in fi.calls:                     # CallInfo(module, owner, line, qualname, name)
                self.edges.append(self.resolver.resolve(ci))
        self.edges.sort(key=lambda e: (e.caller, e.line))
        return self

    def summary(self) -> str:
        counts = Counter(e.status for e in self.edges)
        total = len(self.edges)
        # 解析率只统计本地可解范围：external（内置/标准库/第三方）天然不属于我们的符号表
        local = counts["resolved"] + counts["ambiguous"] + counts["unresolved"]
        rate = counts["resolved"] / local * 100 if local else 0.0
        return (f"调用图: {total} 条调用边 | "
                f"本地命中 ✓{counts['resolved']} / 歧义 ?{counts['ambiguous']} / 未解 ✗{counts['unresolved']} "
                f"| 外部(内置·标准库·第三方) 外{counts['external']} | 本地解析率 {rate:.1f}%")

    def to_dict(self) -> dict[str, object]:
        return {
            "edges": [asdict(e) for e in self.edges],
            "summary": self.summary(),
        }


# ---------- 渲染 ----------

_MARK = {"resolved": "✓", "ambiguous": "?", "external": "外", "unresolved": "✗"}
_EXPLAIN = {
    "resolved": "解析成功",
    "ambiguous": "多个候选（静态无法确定运行时目标）",
    "external": "内置函数 / 标准库 / 第三方库，本工具不解析",
    "unresolved": "本地符号表无匹配（对象属性链 / 动态调用 / 本地漏解）",
}


def render_graph(graph: CallGraph, caller_limit: int = 0,
                 only_unresolved: bool = False, show_external: bool = False):
    print("=" * 72)
    print(graph.summary())
    print("=" * 72)

    by_caller: dict[str, list[CallEdge]] = {}
    for e in graph.edges:
        by_caller.setdefault(e.caller, []).append(e)

    for caller in sorted(by_caller):
        edges = by_caller[caller]
        external = [e for e in edges if e.status == "external"]
        core = [e for e in edges if e.status != "external"]
        if only_unresolved:                          # 只看本地该解未解的
            core = [e for e in core if e.status in ("ambiguous", "unresolved")]
        if not core and not (show_external and external):
            continue
        print(f"\n{caller}")
        shown = core + (external if show_external else [])
        for e in shown[:caller_limit] if caller_limit else shown:
            mark = _MARK[e.status]
            extra = ""
            if e.status == "ambiguous":
                extra = "  [候选: " + ", ".join(e.candidates) + "]"
            elif e.status in ("external", "unresolved"):
                extra = "  [" + _EXPLAIN[e.status] + "]"
            print(f"    L{e.line:<5} {e.expression:<46} -> {e.callee:<64} {mark}{extra}")


def render_query(graph: CallGraph, name: str):
    """反向查询：输入符号名（短名或限定名），列出所有调用它的位置"""
    hits = [
        e for e in graph.edges
        if e.callee == name or (e.status == "resolved" and e.callee.rsplit(".", 1)[-1] == name)
    ]
    print(f"\n符号 {name!r} 的调用者（{len(hits)} 处调用）:")
    for e in hits:
        print(f"  L{e.line:<5} {e.module:<45} {e.caller}   [{e.expression}]")
    if not hits:
        print("  （无。可能是第三方符号，或项目里没人调用它）")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="跨文件调用图（Phase 2）")
    ap.add_argument("root", help="项目源码根目录")
    ap.add_argument("--json", metavar="FILE", help="导出调用图为 JSON")
    ap.add_argument("--query", metavar="NAME", help="反向查询某个符号的调用者")
    ap.add_argument("--unresolved", action="store_true", help="只看未解析/歧义的调用")
    ap.add_argument("--external", action="store_true", help="同时显示内置/标准库/第三方调用")
    ap.add_argument("--limit", type=int, default=0, help="每个函数最多显示的边数（0=不限）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        sys.exit(1)

    index = ProjectIndex().build(root)
    graph = CallGraph(index).build()
    render_graph(graph, args.limit, args.unresolved, args.external)

    if args.query:
        render_query(graph, args.query)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n调用图已导出: {args.json}")


if __name__ == "__main__":
    main()
