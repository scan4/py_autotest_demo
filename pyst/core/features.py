#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3：功能点拆分器（开发计划 M3）
========================================
把"项目级事实"（Phase 1 索引 + Phase 2 调用图）聚合成"功能点档案"，
每个档案是后续喂给大模型生成测试用例的最小单元。

【本阶段回答】"这个项目有哪些功能点？每个功能点包含哪些代码事实？"

一个"功能点"在测试意义上的定义：
    一个业务入口（函数/方法）+ 它触发的一组内部调用 + 关键控制流 + 外部依赖。
    例如 Django 的 view、带 @route 装饰器的函数、类的 public 方法。

【入口识别（已验证结论，见 开发计划.md M3）】
    约定信号（路径含 views/api/路由装饰器）∧ 图结构信号（fan-in=0）组合最有效：
    TestBrain/apps 里路径含 views 的 caller，25/26 个 fan-in=0 —— 这些正是 Django
    view（由 URL 路由触发，图内无人调用）。fan-in>0 的是内部复用辅助函数，不算入口。

【输入】ProjectIndex（Phase 1，含 function_infos）+ CallGraph（Phase 2，含 edges）
【输出】list[FeaturePoint] —— 每个功能点一份"档案"，可直接序列化为 JSON 喂给 LLM。

【为什么不是喂完整源码】见 开发计划.md §风险与设计取舍：
    完整源码 token 装不下、注意力稀释、易诱发幻觉；
    结构化档案（签名 + 调用子图 + 控制流骨架）是"信息密度最高"的输入形态。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .project_index import ProjectIndex
    from .callgraph import CallGraph, CallEdge

# 约定信号：路径中含这些片段 → 强烈提示是入口（Web 框架约定）
_ENTRY_PATH_HINTS = ("views", "api", "routes", "endpoints", "handlers")
_ENTRY_PATH_HINT_SET = set(_ENTRY_PATH_HINTS)
# 约定信号：这些装饰器 → 是路由/HTTP 入口
_ENTRY_DECORATORS = ("route", "app.", "router.", "@api_view", "require_http_methods",
                     "@get", "@post", "@put", "@delete", "@patch")


def _has_entry_path_hint(module: str) -> bool:
    """路径信号：按【路径段】精确匹配，而非子串。

    子串匹配的坑（实测）：root 选在 full-stack-fastapi-template 的上级时，
    模块名带 "full-stack-fastapi-template." 前缀，而子串 "api" ⊂ "fastapi"，
    导致项目里所有 fan_in=0 的纯函数（如 alembic env 的 run_migrations_offline）
    都被误判为入口。按段匹配则 "full-stack-fastapi-template" ≠ "api"，不会误命中。
    """
    return bool(set(module.split(".")) & _ENTRY_PATH_HINT_SET)


# ---------- 功能点档案 ----------

@dataclass
class FeaturePoint:
    entry: str                      # 入口符号限定名：apps.users.views.create_user
    kind: str                       # function | method | async method
    signature: str                  # 人类可读签名，如 "def create_user(request, name: str) -> JsonResponse"
    docstring: str | None           # 入口 docstring 首行（业务语义线索）
    module: str                     # 入口所在模块
    line_range: tuple[int, int]     # 入口代码行号区间 [start, end]
    decorators: list[str] = field(default_factory=list)   # @require_http_methods / @route 等
    call_edges: list[dict[str, Any]] = field(default_factory=list)  # 本地调用子图：[{callee, line, status}]
    external_calls: list[str] = field(default_factory=list)  # 外部依赖调用表达式（requests.post 等）
    control_sites: list[dict[str, Any]] = field(default_factory=list)  # 控制流骨架：[{kind, line, cond, branch}]（M2 产物）
    interface: dict[str, Any] | None = None  # HTTP 接口信息（Phase 5）：{url, methods, params,...}
    interfaces: list[dict[str, Any]] = field(default_factory=list)  # 全部关联接口（OpenAPI 探测可 1:N）
    max_depth: int = 1        # 调用子图展开深度（1=仅入口，2=含直接下游，...）

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "kind": self.kind,
            "signature": self.signature,
            "docstring": self.docstring,
            "module": self.module,
            "line_range": list(self.line_range),
            "decorators": self.decorators,
            "call_edges": self.call_edges,
            "external_calls": self.external_calls,
            "control_sites": self.control_sites,
            "interface": self.interface,
            "interfaces": self.interfaces,
            "max_depth": self.max_depth,
        }

    # ---------- 代码指纹：检测"相关代码是否变化"（缓存失效判断） ----------
    def fingerprint(self, index=None) -> str:
        """计算功能点的代码指纹，覆盖入口自身 + 下游调用链。

        指纹基于会影响测试用例的信息：
          - 入口自身：签名、装饰器、接口、控制流骨架
          - 下游调用链：每个 call_edge 指向的本地符号的签名 + 控制流
            （能检测下游节点被删除 / 改逻辑 / 加分支）
        代码变了 → 指纹变 → 缓存失效，需重新生成用例。

        Args:
            index: ProjectIndex（提供 function_infos 取下游实现），可为 None
        """
        import hashlib
        import json as _json

        # 入口自身
        parts = [
            f"depth={self.max_depth}",
            self.signature,
            sorted(self.decorators),
            sorted(str(x) for x in self.control_sites),
            _json.dumps(self.interface, sort_keys=True, ensure_ascii=False) if self.interface else "",
        ]
        # 下游调用链：每个本地被调符号的实现（签名 + 控制流 + 该节点的下游）
        if index is not None:
            for edge in sorted(self.call_edges, key=lambda e: e.get("callee", "")):
                callee = edge.get("callee", "")
                depth = edge.get("depth", 1)
                fi = index.function_infos.get(callee)
                if fi is not None:
                    sig = _format_signature(fi)
                    parts.append(f"d{depth}|{callee}|{sig}|{sorted(str(c) for c in fi.controls)}")
        raw = ";;".join(str(p) for p in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def diff_with(self, index, old: "FeaturePoint") -> list[str]:
        """对比新旧功能点，返回变更描述（用于喂给 LLM 重新生成）。"""
        changes: list[str] = []
        # 签名
        if self.signature != old.signature:
            changes.append(f"签名变化: {old.signature} -> {self.signature}")
        # 控制流
        new_controls = {c.get("line"): c for c in self.control_sites}
        old_controls = {c.get("line"): c for c in old.control_sites}
        if new_controls.keys() != old_controls.keys():
            added = [new_controls[k] for k in new_controls if k not in old_controls]
            removed = [old_controls[k] for k in old_controls if k not in new_controls]
            if added:
                changes.append(f"新增控制流分支: {[c.get('kind')+' '+str(c.get('cond','')) for c in added]}")
            if removed:
                changes.append(f"删除控制流分支: {[c.get('kind')+' '+str(c.get('cond','')) for c in removed]}")
        else:
            # 行号相同但条件变了
            for k in new_controls:
                if new_controls[k].get("cond") != old_controls[k].get("cond"):
                    changes.append(f"控制流条件变化: {old_controls[k].get('kind')} '{old_controls[k].get('cond','')}' -> '{new_controls[k].get('cond','')}'")
        # 接口
        if self.interface != old.interface:
            changes.append(f"接口变化: {old.interface} -> {self.interface}")
        # 下游调用链
        new_callees = {e.get("callee") for e in self.call_edges}
        old_callees = {e.get("callee") for e in old.call_edges}
        removed_callees = old_callees - new_callees
        added_callees = new_callees - old_callees
        if removed_callees:
            changes.append(f"下游被调函数被删除: {sorted(removed_callees)}")
        if added_callees:
            changes.append(f"新增下游调用: {sorted(added_callees)}")
        return changes


# ---------- 功能点入口识别 ----------

def _is_entry_symbol(qualname: str, kind: str, decorators: list[str],
                     fan_in: int, fan_out: int) -> bool:
    """判断一个符号是否适合作为功能点入口。

    策略（M3 已验证的组合）：
      A. 约定信号：路径含 views/api/routes（入口强信号）
      B. 图结构：fan-in=0（无人调用 → 外部/URL 触达）
      C. 装饰器信号：带路由/HTTP 装饰器
    命中 A 且 fan-in=0 是强入口；命中 C 也是入口；A 但不满足 fan-in=0 是辅助函数。
    """
    module = qualname.rsplit(".", 1)[0] if "." in qualname else ""
    path_hint = _has_entry_path_hint(module)
    deco_hint = any(any(d in dk for dk in decorators) for d in _ENTRY_DECORATORS)
    if deco_hint:
        return True                                # 明确的路由/HTTP 装饰器，必是入口
    if path_hint and fan_in == 0:
        return True                                # views/api 且无人调用 → 入口
    return False


# ---------- 主入口 ----------

def split_features(index: "ProjectIndex", graph: "CallGraph",
                   max_depth: int = 2,
                   entries: Iterable[str] = ()) -> list[FeaturePoint]:
    """把项目事实聚合成功能点档案列表。

    Args:
        index:     Phase 1 项目索引（符号表 / function_infos / 调用点）
        graph:     Phase 2 调用图（调用边）
        max_depth: 本地调用子图沿调用边的展开深度（控制档案大小 / token 预算）
        entries:   可选白名单限定名，显式指定入口（人工兜底，覆盖自动识别）

    Returns:
        按入口限定的功能点档案列表。
    """
    # 1. 统计每个符号的 fan-in（谁调用了它）——入口判定需要
    fan_in: dict[str, int] = {}
    for e in graph.edges:
        if e.status == "resolved" and e.callee in index.symbols:
            fan_in[e.callee] = fan_in.get(e.callee, 0) + 1

    # 1.5 接口信息准备（Phase 5）：收集 urls.py 路由 + include 前缀
    from .interface import collect_urls, build_interface, _clean_url_pattern
    from .interface import collect_decorator_routes, build_decorator_interface
    import ast as _ast
    url_mapping, include_prefixes = _collect_interface_info(index)
    # 通用装饰器路由探测（Django 之外的框架：FastAPI/Flask/Sanic/Bottle...），
    # 服务未运行拿不到 OpenAPI 时的静态兜底。只在 Django 提取不到时启用。
    deco_routes = collect_decorator_routes(index)

    # 2. 按 caller 聚合调用边，方便取"某入口直接调用了谁"
    by_caller: dict[str, list[CallEdge]] = {}
    for e in graph.edges:
        by_caller.setdefault(e.caller, []).append(e)

    # 3. 候选入口 = 白名单 ∪ 自动识别（符号表里非 class 的符号）
    whitelist = set(entries)
    features: list[FeaturePoint] = []

    # 测试模块里的符号不参与"自动识别"：
    # 测试函数天然 fan-in=0，且路径常含 api/routes（如 tests.api.routes.test_items），
    # 会同时命中 _ENTRY_PATH_HINTS 和 fan-in=0 两条规则，被误当成功能点入口。
    # 测试模块判定见 project_index.is_test_module（文件位置 + 模块级 import 双信号，不看函数名）。
    # 注意：白名单（用户显式指定 entries）不受此限制，保留人工兜底能力。
    test_modules = {m for m, fi in index.files.items() if fi.is_test}

    # 只遍历"可执行符号"（function/method，排除 class 本身）
    candidates = [
        s for s in index.symbols.values()
        if s.kind in ("function", "method") and s.module not in test_modules
    ]
    # 去重（同名符号可能来自不同模块，按限定名保留）
    seen = set()

    def _expand_calls(caller_qualname: str, current_depth: int,
                      visited: set[str], depth_limit: int) -> list[dict]:
        """递归展开调用链，直到 max_depth。

        Args:
            caller_qualname: 当前函数限定名
            current_depth: 当前深度（入口=1）
            visited: 已访问的符号（防环）
            depth_limit: 最大深度

        Returns:
            call_edges 列表，每个含 callee/line/status/depth
        """
        edges = by_caller.get(caller_qualname, [])
        result = []
        for e in edges:
            if e.status != "resolved":
                continue
            node = {"callee": e.callee, "line": e.line, "status": e.status, "depth": current_depth}
            # 带上该下游函数自身的控制逻辑（改动2：让 LLM 看到下游分支/异常）
            d_fi = index.function_infos.get(e.callee)
            if d_fi is not None:
                node["signature"] = _format_signature(d_fi)
                node["controls"] = [
                    {"kind": c.kind, "line": c.line, "cond": c.cond, "branch": c.branch,
                     "start": c.start, "end": c.end, "semantics": c.semantics, "path": c.path}
                    for c in d_fi.controls
                ]
            result.append(node)
            # 递归展开下一层（若未超深且未访问过该下游函数）
            if current_depth < depth_limit and e.callee not in visited \
                    and e.callee in index.function_infos:
                visited.add(e.callee)
                result.extend(_expand_calls(e.callee, current_depth + 1, visited, depth_limit))
        return result

    def _build_feature(qualname: str, depth_limit: int = 1) -> FeaturePoint | None:
        if qualname in seen or qualname not in index.symbols:
            return None
        sym = index.symbols[qualname]
        fi = index.function_infos.get(qualname)
        if fi is None:                              # 无函数体信息（异常），跳过
            return None
        seen.add(qualname)

        # 调用子图（可配置深度展开）
        call_edges = _expand_calls(qualname, 1, {qualname}, depth_limit)
        # 外部/未解调用（入口的直接调用）
        external = set()
        for e in by_caller.get(qualname, []):
            if e.status != "resolved":
                external.add(e.expression)          # 外部/未解调用按表达式记录

        # 控制流骨架（Phase 0 已收集，含语义标签+行区间用于盲区覆盖）
        control_sites = [
            {"kind": c.kind, "line": c.line, "cond": c.cond, "branch": c.branch,
             "start": c.start, "end": c.end, "semantics": c.semantics, "path": c.path}
            for c in fi.controls
        ]

        # HTTP 接口信息（Phase 5）：优先 Django urls.py 提取，失败则用通用装饰器探测兜底
        interface = None
        if url_mapping:
            iface = _build_feature_interface(index, sym.module, qualname, url_mapping, include_prefixes)
            if iface:
                interface = iface.to_dict()
        if interface is None and deco_routes.get(qualname):
            d_iface = build_decorator_interface(deco_routes[qualname])
            if d_iface:
                interface = d_iface.to_dict()

        return FeaturePoint(
            entry=qualname,
            kind=fi.kind,
            signature=_format_signature(fi),
            docstring=fi.docstring,
            module=sym.module,
            line_range=(sym.start, sym.end),
            decorators=fi.decorators,
            call_edges=call_edges,
            external_calls=sorted(external),
            control_sites=control_sites,
            interface=interface,
            max_depth=depth_limit,
        )

    # 3a. 白名单显式入口
    for q in whitelist:
        fp = _build_feature(q, max_depth)
        if fp:
            features.append(fp)

    # 3b. 自动识别
    for s in candidates:
        q = s.qualname
        if q in seen:
            continue
        f_info = index.function_infos.get(q)
        if f_info is None:
            continue
        fi = fan_in.get(q, 0)
        fo = len(by_caller.get(q, []))
        if _is_entry_symbol(q, s.kind, f_info.decorators, fi, fo):
            fp = _build_feature(q, max_depth)
            if fp:
                features.append(fp)

    # 4. 排序：白名单优先，其余按模块分组、fan-in 小的（更像入口）在前
    features.sort(key=lambda f: (0 if f.entry in whitelist else 1, f.module, f.entry))
    return features


def _collect_interface_info(index):
    """收集项目 urls.py 的路由映射和 include 前缀。

    Returns:
        (url_mapping, include_prefixes)
        - url_mapping: {view_name: [url_pattern]}  来自所有 urls.py 的 path() 调用
        - include_prefixes: {模块名: include前缀}  从 urlpatterns 里的 include('...')
          推断每个 app 挂在哪个前缀下（如 iface_case_generator -> iface_case_generator）
    """
    from .interface import collect_urls
    import ast as _ast

    # 找所有 urls.py 文件（通过 module_paths 里的模块名含 urls 的）
    urls_files = []
    for mod, path in index.module_paths.items():
        if mod.endswith(".urls"):
            urls_files.append(Path(path))
    if not urls_files:
        # 兜底：按目录名找
        urls_files = [Path(p) for p in index.module_paths.values()]
    url_mapping = collect_urls(urls_files)

    # 推断 include 前缀：urlpatterns 里 include('apps.xxx.urls') -> xxx
    include_prefixes: dict[str, str] = {}
    for mod, path in index.module_paths.items():
        if not mod.endswith(".urls"):
            continue
        try:
            tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                    and node.func.id == "include" and node.args:
                arg = node.args[0]
                if isinstance(arg, _ast.Constant) and isinstance(arg.value, str):
                    # include('apps.iface_case_generator.urls') -> 前缀 iface_case_generator
                    parts = arg.value.split(".")
                    if len(parts) >= 2:
                        prefix = parts[-2]          # xxx.urls 的 xxx
                        include_prefixes[prefix] = prefix
    return url_mapping, include_prefixes


def _build_feature_interface(index, module: str, qualname: str,
                             url_mapping: dict[str, list[str]],
                             include_prefixes: dict[str, str]):
    """为入口符号构建 InterfaceInfo（从 view 函数体提取 method/参数）。

    Args:
        index: ProjectIndex（用 module_paths 定位模块文件）
        module: 入口所在模块（apps.xxx.views）
        qualname: 入口限定名
        url_mapping: collect_urls 结果
        include_prefixes: 模块 -> include 前缀

    Returns:
        InterfaceInfo | None（无 URL 映射时返回 None，即非 HTTP 入口）
    """
    from .interface import build_interface
    import ast as _ast

    view_name = qualname.rsplit(".", 1)[-1]
    # 模块文件路径：module 可能是 apps.xxx.views，找它的 .py
    file_path = index.module_paths.get(module)
    if not file_path:
        return None
    try:
        tree = _ast.parse(Path(file_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    # 找 view 函数定义，提取 body
    fn_body = None
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == view_name:
            fn_body = node.body
            break
    if fn_body is None:
        return None
    # include 前缀：入口模块名倒数第二段是 app 名
    parts = module.split(".")
    app_name = parts[-2] if len(parts) >= 2 else ""
    prefix = include_prefixes.get(app_name, app_name)
    return build_interface(module, view_name, fn_body, url_mapping, include_prefix=prefix)


def _format_signature(fi) -> str:
    """把 FunctionInfo 渲染成人类可读签名：def name(args) -> returns"""
    prefix = "async def" if fi.kind == "async method" or fi.kind == "async function" else "def"
    ret = f" -> {fi.returns}" if fi.returns else ""
    return f"{prefix} {fi.name}({', '.join(fi.args)}){ret}"


# ---------- 渲染（验证用） ----------

def render_feature(fp: FeaturePoint):
    print(f"功能点: {fp.entry}  [行 {fp.line_range[0]}-{fp.line_range[1]}]")
    print(f"  签名: {fp.signature}")
    if fp.docstring:
        print(f"  说明: {fp.docstring}")
    if fp.decorators:
        print(f"  装饰器: {', '.join(fp.decorators)}")
    print(f"  本地调用 ({len(fp.call_edges)}):")
    for e in fp.call_edges:
        print(f"    L{e['line']:<5} -> {e['callee']:<60} [{e['status']}]")
    print(f"  外部依赖 ({len(fp.external_calls)}): {', '.join(fp.external_calls)}")
    print(f"  控制流 ({len(fp.control_sites)}):")
    for c in fp.control_sites:
        cond = f" {c['cond']}" if c.get("cond") else ""
        branch = f" [{c['branch']}]" if c.get("branch") else ""
        print(f"    L{c['line']:<5} {c['kind']}{cond}{branch}")


def render_features(features: list[FeaturePoint]):
    print(f"\n{'='*72}\n功能点拆分结果：共 {len(features)} 个功能点\n{'='*72}")
    for i, fp in enumerate(features):
        print(f"\n[{i+1}] {fp.entry}  [行 {fp.line_range[0]}-{fp.line_range[1]}]")
        print(f"    签名: {fp.signature}")
        if fp.docstring:
            print(f"    说明: {fp.docstring}")
        print(f"    本地调用 ({len(fp.call_edges)}):")
        for e in fp.call_edges:
            print(f"      L{e['line']:<5} -> {e['callee'][:70]}")
        print(f"    外部依赖 ({len(fp.external_calls)}): {', '.join(fp.external_calls[:8])}")
        print(f"    控制流 ({len(fp.control_sites)}):")
        for c in fp.control_sites[:8]:
            cond = f" {c['cond']}" if c.get("cond") else ""
            print(f"      L{c['line']:<5} {c['kind']}{cond}")


def main():
    import argparse
    import json
    import sys
    from pathlib import Path
    from .project_index import ProjectIndex
    from .callgraph import CallGraph

    ap = argparse.ArgumentParser(description="功能点拆分（Phase 3）")
    ap.add_argument("root", help="项目源码根目录")
    ap.add_argument("--json", metavar="FILE", help="导出功能点档案 JSON")
    ap.add_argument("--depth", type=int, default=2, help="本地调用子图展开深度（默认 2）")
    ap.add_argument("--entry", metavar="QUALNAME", action="append",
                    help="显式指定入口（可多次），覆盖自动识别")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        sys.exit(1)

    index = ProjectIndex().build(root)
    graph = CallGraph(index).build()
    features = split_features(index, graph, max_depth=args.depth, entries=args.entry or [])
    render_features(features)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([fp.to_dict() for fp in features], f, ensure_ascii=False, indent=2)
        print(f"\n功能点档案已导出: {args.json}")


if __name__ == "__main__":
    main()
