#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5：接口定义提取（让测试用例"可直接执行"）
====================================================
从 urls.py 路由 + view 函数体提取真实 HTTP 接口信息，让生成的测试用例
能直接通过 HTTP 请求执行（而非自然语言描述）。

【动机】之前模型返回的用例是"构造一个 GET 请求，URL 中 file_path 参数..."
这种描述性文字，不能直接用。因为档案里只有代码事实，没有 HTTP 层信息。
本阶段从代码里提取真实的 method / url / 参数，喂给模型生成可执行 request。

【两个事实来源】
1. urls.py 路由：path('download_file/', views.download_file) + include 前缀
   → 得到完整 URL 和 view→接口的映射
2. view 函数体：request.method == 'GET' / request.POST.get('file_path')
   → 得到 method 和参数名/来源

【输出】InterfaceInfo：method / url / 参数列表 / 请求体提示
       挂到 FeaturePoint.interface，供 prompt 生成可执行 request。
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InterfaceParam:
    name: str                       # 参数名：file_path / single_file
    location: str                   # query | form | json | file | path
    type_hint: str | None = None    # 类型提示：int/str/文件（如能推断）


@dataclass
class InterfaceInfo:
    url: str                        # 完整 URL（含 include 前缀）：/iface_case_generator/download_file/
    methods: list[str]              # ['GET'] / ['POST'] / ['GET','POST']
    params: list[InterfaceParam] = field(default_factory=list)   # 请求参数
    body_type: str | None = None    # form | json | None（POST 时的请求体类型）
    file_field: str | None = None   # 文件上传字段名（如有）
    source: str = "urls"            # 来源说明

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "methods": self.methods,
            "params": [{"name": p.name, "location": p.location, "type_hint": p.type_hint}
                       for p in self.params],
            "body_type": self.body_type,
            "file_field": self.file_field,
        }


# ---------- urls.py 路由解析 ----------

def _parse_path_call(node: ast.Call) -> tuple[str, str] | None:
    """解析 path('download_file/', views.download_file) -> (url_pattern, view_name)。

    兼容 path / re_path；view 可能是 views.xxx 或直接符号名。
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.id if isinstance(func, ast.Name) else None
    if name not in ("path", "re_path"):
        return None
    if len(node.args) < 2:
        return None
    # 第一个参数：URL pattern
    url_arg = node.args[0]
    if not isinstance(url_arg, ast.Constant) or not isinstance(url_arg.value, str):
        return None
    url_pattern = url_arg.value
    # 第二个参数：view（views.download_file 或 download_file）
    view_arg = node.args[1]
    view_name = None
    if isinstance(view_arg, ast.Attribute):
        view_name = view_arg.attr          # views.download_file -> download_file
    elif isinstance(view_arg, ast.Name):
        view_name = view_arg.id            # download_file
    elif isinstance(view_arg, ast.Call):   # include('...') 之类，跳过
        return None
    return url_pattern, view_name


def _clean_url_pattern(pattern: str) -> str:
    """把 Django path 模式转成可访问 URL（去掉 <int:x>/<str:y> 参数占位）。

    download_file/            -> download_file/
    api/<int:user_id>/        -> api/{user_id}/
    """
    return re.sub(r"<[^>]*>", "{}", pattern)


def collect_urls(urls_files: list[Path]) -> dict[str, list[str]]:
    """解析一组 urls.py，返回 {view_name: [完整URL]} 映射。

    注意：本实现不处理 include 递归（跨文件前缀拼接），只提取每个 urls.py
    内 view 的 URL pattern。完整前缀由 _build_interface 用 include 映射补全。
    """
    mapping: dict[str, list[str]] = {}
    for f in urls_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        # 找 urlpatterns 列表里的 path() 调用
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) > 0:
                for elt in node.elts:
                    parsed = _parse_path_call(elt)
                    if parsed:
                        pattern, view_name = parsed
                        if view_name:
                            mapping.setdefault(view_name, []).append(_clean_url_pattern(pattern))
    return mapping


# ---------- view 函数体 method/参数 提取 ----------

def _extract_methods(fn_body: list[ast.stmt]) -> list[str]:
    """从 view 函数体找 request.method == 'GET'/'POST' 判断，得到允许的 method"""
    methods: list[str] = []
    for n in ast.walk(ast.Module(body=fn_body, type_ignores=[])):
        if isinstance(n, ast.Compare):
            # request.method == 'GET'
            if (isinstance(n.left, ast.Attribute) and n.left.attr == "method"
                    and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)):
                comp = n.comparators[0]
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    methods.append(comp.value)
    return methods or ["GET", "POST"]   # 无法判断时假设 GET+POST


def _extract_params(fn_body: list[ast.stmt]) -> list[InterfaceParam]:
    """从 view 函数体找 request.POST/GET/FILES 的参数用法。

    request.POST.get('file_path')     -> form/query 参数 file_path
    request.GET.get('task_id')        -> query 参数
    request.FILES['single_file']      -> 文件上传
    """
    params: list[InterfaceParam] = []
    seen: set[tuple[str, str]] = set()
    for n in ast.walk(ast.Module(body=fn_body, type_ignores=[])):
        if isinstance(n, ast.Call):
            # request.POST.get('x') / request.GET.get('x')
            if isinstance(n.func, ast.Attribute) and n.func.attr == "get":
                obj = n.func.value
                if isinstance(obj, ast.Attribute) and obj.attr in ("POST", "GET", "FILES"):
                    if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                        name = n.args[0].value
                        loc = "form" if obj.attr == "POST" else ("query" if obj.attr == "GET" else "file")
                        if (name, loc) not in seen:
                            seen.add((name, loc))
                            params.append(InterfaceParam(name=name, location=loc))
        elif isinstance(n, ast.Subscript):
            # request.FILES['single_file'] / request.POST['x']
            if isinstance(n.value, ast.Attribute) and n.value.attr in ("POST", "GET", "FILES"):
                if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                    name = n.slice.value
                    loc = "form" if n.value.attr == "POST" else ("query" if n.value.attr == "GET" else "file")
                    if (name, loc) not in seen:
                        seen.add((name, loc))
                        params.append(InterfaceParam(name=name, location=loc))
    return params


# ---------- 组装 InterfaceInfo ----------

def build_interface(module: str, view_name: str,
                    fn_body: list[ast.stmt],
                    url_mapping: dict[str, list[str]],
                    include_prefix: str = "") -> InterfaceInfo | None:
    """为一个 view 函数构建接口信息。

    Args:
        module: 入口所在模块（apps.xxx.views）
        view_name: 函数名（download_file）
        fn_body: view 函数体 AST（用于提取 method/参数）
        url_mapping: collect_urls 得到的 {view_name: [url_pattern]}
        include_prefix: 该模块挂在 config/urls.py 下的 include 前缀

    Returns:
        InterfaceInfo（若在 url_mapping 里找到了 URL）
    """
    urls = url_mapping.get(view_name)
    if not urls:
        return None
    # 取第一个 URL（多路由取第一个）
    url = f"{include_prefix.rstrip('/')}/{urls[0].lstrip('/')}" if include_prefix else urls[0]
    if not url.startswith("/"):
        url = "/" + url
    methods = _extract_methods(fn_body)
    params = _extract_params(fn_body)

    # 判断 body_type / file_field
    file_fields = [p.name for p in params if p.location == "file"]
    has_form = any(p.location == "form" for p in params)
    return InterfaceInfo(
        url=url,
        methods=methods,
        params=params,
        body_type="file" if file_fields else ("form" if has_form else None),
        file_field=file_fields[0] if file_fields else None,
    )


# ===========================================================================
# 通用装饰器路由探测（服务未运行、拿不到 OpenAPI 时的静态兜底）
# ===========================================================================
# 【动机】上面的 Django 提取对 FastAPI/Flask/Sanic/Bottle 等"装饰器注册"框架
# 完全失效（实测 FastAPI 项目 0/23 有 URL）。而运行时 OpenAPI 探测要求服务在跑。
# 本节从源码装饰器直接提取路由，覆盖"声明式"框架，不依赖服务运行。
#
# 【支持形态】（target=装饰器作用对象，如 app/router/bp，也可能裸写）
#   router.get('/items/')                              → method 在装饰器名里
#   app.route('/user/<int:id>', methods=['GET','POST'])→ method 在参数里（Flask/Bottle）
#   app.api_view(['GET','POST'])                       → DRF 函数视图
#   get('/path') / route('/path')                      → 裸装饰器（Litestar/Bottle）
#   websocket 装饰器跳过（非 REST）
#
# 【前缀】解析同模块的 APIRouter(prefix=)/Blueprint(url_prefix=) 赋值，
#          以及任意模块的 include_router(router, prefix=)/register_blueprint(bp, prefix=)。
#          前缀不是字符串常量（如 settings.API_V1_STR）时，以表达式占位并在 note 里说明。

import ast as _ast

_ROUTER_FACTORIES = ("APIRouter", "Blueprint", "Router", "BlueprintApi")
_INCLUDE_FUNCS = ("include_router", "register_blueprint")
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
_METHOD_KW_FUNCS = {"route", "api_route", "add_api_route", "api_view", "methods"}
_SKIP_PARAM_NAMES = {
    "request", "response", "session", "db", "current_user", "credentials",
    "background_tasks", "settings", "self", "cls",
}
_DIY_ANN_MARKERS = ("Depends", "Session", "CurrentUser", "Request", "Annotated",
                    "SessionDep", "Generator", "Iterator")
_SCALAR_TYPES = {"str", "int", "float", "bool", "EmailStr", "Path", "Query"}


def _parse_decorator_route(deco: str) -> tuple[str, list[str]] | None:
    """解析一条装饰器表达式 → (路径, methods)。非路由装饰器返回 None。

    Args:
        deco: ast.unparse 后的装饰器原文，如 router.get('/items/', response_model=X)
    """
    m = re.match(r"^\s*([\w\.]+)\s*\(\s*(['\"])(.*?)\2", deco)
    if not m:
        return None
    target = m.group(1)
    verb = target.rsplit(".", 1)[-1]
    path = m.group(3)
    rest = deco[m.end() - 1:]
    if verb == "websocket" or verb.startswith("ws"):
        return None
    if verb in _HTTP_VERBS:
        return path, [verb.upper()]
    if verb in _METHOD_KW_FUNCS or verb.endswith("route") or verb == "api_view":
        mm = re.search(r"methods\s*=\s*\[([^\]]*)\]", rest)
        if mm:
            methods = [x.strip().strip("'\"").upper()
                       for x in mm.group(1).split(",") if x.strip()]
            if methods:
                return path, methods
        return path, ["GET"]                   # Flask @route 缺省 GET
    return None


def _clean_path_params(pattern: str) -> str:
    """统一路径参数为 {name}：FastAPI {id} 保留；Flask <int:id> 转 {id}。"""
    def _sub(mm: re.Match) -> str:
        inner = mm.group(1)
        return "{" + (inner.split(":")[-1].strip() if ":" in inner else inner) + "}"
    s = re.sub(r"<([^>]+)>", _sub, pattern)
    return s


def _join_prefix(a: str, b: str) -> str:
    """拼接两段 URL 前缀，处理斜杠与空段：_join_prefix('/api', '/items') -> '/api/items'。"""
    if not a:
        return b
    if not b:
        return a
    return a.rstrip("/") + "/" + b.lstrip("/")


def _module_route_context(path: Path) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """解析一个模块文件的路由上下文。

    Returns:
        (var_prefixes, includes)
        - var_prefixes: {router变量名: 前缀}  来自 APIRouter(prefix=...)/Blueprint(url_prefix=...)
        - includes: [(父挂载点表达式, 子router表达式, 前缀)]
          来自 include_router(parent).include(child, prefix=...)
    """
    var_prefixes: dict[str, str] = {}
    includes: list[tuple[str, str]] = []
    try:
        tree = _ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return var_prefixes, includes

    def _const_or_expr(node) -> str:
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            return node.value
        return "{" + _ast.unparse(node) + "}"      # 非常量（settings.API_V1_STR）以表达式占位

    for node in _ast.walk(tree):
        # router = APIRouter(prefix='/items') / bp = Blueprint(..., url_prefix='/blog')
        # 无 prefix 的工厂赋值也记录（记空串）：后续 include 边的变量归属解析依赖它
        if isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call):
            f = node.value.func
            fname = f.id if isinstance(f, _ast.Name) else (f.attr if isinstance(f, _ast.Attribute) else "")
            if fname in _ROUTER_FACTORIES:
                var = node.targets[0].id if isinstance(node.targets[0], _ast.Name) else None
                if var:
                    var_prefixes[var] = ""
                    for kw in node.value.keywords:
                        if kw.arg in ("prefix", "url_prefix"):
                            var_prefixes[var] = _const_or_expr(kw.value)
        # app.include_router(items.router, prefix='/api/v1')
        # 记录 (父挂载点表达式, 子router表达式, prefix)，保留完整表达式供 import 别名解析
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute) \
                and node.func.attr in _INCLUDE_FUNCS and node.args:
            arg = node.args[0]
            child = None
            if isinstance(arg, _ast.Name):
                child = arg.id
            elif isinstance(arg, _ast.Attribute):
                child = _ast.unparse(arg)          # items.router -> "items.router"
            parent = _ast.unparse(node.func.value)  # api_router.include_router -> "api_router"
            prefix = ""
            for kw in node.keywords:
                if kw.arg in ("prefix", "url_prefix"):
                    prefix = _const_or_expr(kw.value)
            if child:
                includes.append((parent, child, prefix))
    return var_prefixes, includes


def _signature_params(fi) -> tuple[list[dict[str, Any]], list[str]]:
    """从函数签名提取参数（声明式框架的参数都在签名里，不在函数体）。

    Returns:
        (params, path_names)
        - params: [{"name","location","type_hint"}]
        - path_names: 路径模板里的参数名（供 location=path 判定）
    """
    path_names: list[str] = []
    params: list[dict[str, Any]] = []
    for raw in fi.args:
        name = raw.split(":")[0].split("=")[0].strip()
        ann = ""
        if ":" in raw:
            ann = raw.split(":", 1)[1].split("=")[0].strip()
        if name in ("self", "cls") or name in _SKIP_PARAM_NAMES:
            continue
        if any(mk in ann for mk in _DIY_ANN_MARKERS):   # 框架注入的依赖，非请求参数
            continue
        base = ann.split("[")[0].strip()
        if name in path_names:
            params.append({"name": name, "location": "path", "type_hint": ann or None})
        elif base in _SCALAR_TYPES or not ann:
            params.append({"name": name, "location": "query", "type_hint": ann or None})
        else:                                            # Pydantic 模型 / dict → 请求体
            params.append({"name": name, "location": "json", "type_hint": ann or None})
    return params, path_names


def _resolve_include_module(index, consumer: str, expr: str) -> tuple[str, str] | None:
    """把 include_router 的变量表达式解析回「定义方模块 + 变量名」。

    'items.router' 里的 items 是 consumer 模块里的 import 别名
    （from app.api.routes import items）→ 定义方模块 app.api.routes.items。
    裸名 'router' 则尝试 from ... import router 形式。
    解析失败返回 None（前缀无法归并，宁可不算也不算错）。
    """
    fi = index.files.get(consumer)
    if fi is None:
        return None
    if "." in expr:
        alias, var = expr.rsplit(".", 1)
    else:
        alias, var = None, expr
    for imp in fi.imports:
        if imp.kind != "from_import" or imp.level:
            continue
        for n in imp.names:
            bound = n.asname or n.name
            if alias is not None:
                if bound == alias:
                    return f"{imp.module}.{n.name}", var
            else:
                if n.name == expr:
                    return imp.module, var
    return None


def collect_decorator_routes(index) -> dict[str, list[dict[str, Any]]]:
    """扫描项目全部函数装饰器，提取通用路由注册（不依赖服务运行）。

    Returns:
        {qualname: [{"url", "methods", "params", "note", "source": "decorator"}]}
        一个函数叠多个路由装饰器（或多处挂载）时会有多条。
    """
    # 1. 找出带路由装饰器的函数（按模块分组，解析前缀用）
    decorated: dict[str, list[tuple[str, str, list[str], str]]] = {}  # module -> [(q, path, methods, target_var)]
    for q, fi in index.function_infos.items():
        for deco in fi.decorators:
            parsed = _parse_decorator_route(deco)
            if parsed:
                module = q.rsplit(".", 1)[0] if "." in q else ""
                tm = re.match(r"^\s*([\w\.]+)\s*\(", deco)
                target_var = tm.group(1).rsplit(".", 1)[0] if "." in tm.group(1) else ""
                decorated.setdefault(module, []).append((q, parsed[0], parsed[1], target_var))
    if not decorated:
        return {}

    # 2. 逐模块解析：router 自身前缀 + include 挂载边（父→子），再经 import 别名
    #    把变量表达式解析回「定义方模块」，构建前缀传递链
    var_prefixes_by_mod: dict[str, dict[str, str]] = {}
    raw_edges: list[tuple[str, str, str, str]] = []   # (consumer, parent_expr, child_expr, prefix)
    for module in set(decorated) | set(index.module_paths):
        p = index.module_paths.get(module)
        if not p:
            continue
        vp, incs = _module_route_context(Path(p))
        if vp:
            var_prefixes_by_mod[module] = vp
        for parent_expr, child_expr, prefix in incs:
            raw_edges.append((module, parent_expr, child_expr, prefix))

    def _resolve_var(consumer: str, expr: str) -> tuple[str, str] | None:
        """变量表达式 → (定义方模块, 变量名)。本地定义优先，其次 import 别名。"""
        if expr in var_prefixes_by_mod.get(consumer, {}):
            return (consumer, expr)
        return _resolve_include_module(index, consumer, expr)

    # 挂载边：child (defmod, var) -> [(parent (defmod, var) 或 ("","") 表示 app 根, 边前缀)]
    edges: dict[tuple[str, str], list[tuple[tuple[str, str], str]]] = {}
    for consumer, parent_expr, child_expr, prefix in raw_edges:
        child = _resolve_var(consumer, child_expr)
        if child is None:
            continue                                    # 解析不到归属，宁可不算也不算错
        parent = _resolve_var(consumer, parent_expr) or ("", "")
        edges.setdefault(child, []).append((parent, prefix))

    # 3. 前缀链传递：URL 前缀 = 上层挂载链 ⊕ 边前缀 ⊕ 自身前缀
    #    （如 items.router ← api_router(prefix=API_V1_STR) ← app → "{API_V1_STR}/items"）
    _MAX_CHAINS = 8

    def _chains(module: str, var: str, seen: frozenset) -> list[str]:
        if (module, var) in seen:
            return []                                   # include 环保护
        own = var_prefixes_by_mod.get(module, {}).get(var, "")
        incoming = edges.get((module, var), [])
        if not incoming:
            return [own] if own else [""]
        out: list[str] = []
        for (pm, pv), edge_prefix in incoming:
            bases = [""] if (pm, pv) == ("", "") else _chains(pm, pv, seen | {(module, var)})
            for base in bases:
                full = _join_prefix(_join_prefix(base, edge_prefix), own)
                if full not in out:
                    out.append(full)
                    if len(out) >= _MAX_CHAINS:
                        return out
        return out

    # 4. 拼 URL + 签名参数
    routes: dict[str, list[dict[str, Any]]] = {}
    for module, items in decorated.items():
        for q, pattern, methods, target_var in items:
            fi = index.function_infos.get(q)
            path = _clean_path_params(pattern)
            path_names = set(re.findall(r"\{([^}:]+)\}", path))
            params: list[dict[str, Any]] = []
            if fi is not None:
                cand, _ = _signature_params(fi)
                for p in cand:
                    if p["name"] in path_names:
                        p = {**p, "location": "path"}
                    params.append(p)
            # 同一函数/router 可能挂多个前缀（include 多次）→ 每个前缀一条
            prefixes = _chains(module, target_var, frozenset()) if target_var else [""]
            for prefix in prefixes:
                url = _join_prefix(prefix, path) or path
                if not url.startswith("/"):
                    url = "/" + url
                note = ""
                if re.search(r"\{\w+\.\w+", url):
                    note = "前缀含非常量表达式（如 settings.API_V1_STR），请以运行时实际 URL 为准"
                routes.setdefault(q, []).append({
                    "url": url, "methods": methods, "params": params,
                    "note": note, "source": "decorator",
                })
    return routes


def build_decorator_interface(routes: list[dict[str, Any]], fi=None) -> "InterfaceInfo | None":
    """把装饰器提取的路由转成 InterfaceInfo（与 Django 提取产物同构）。"""
    if not routes:
        return None
    primary = routes[0]
    params = list(primary.get("params") or [])
    file_fields = [p["name"] for p in params if p.get("location") == "file"]
    has_form = any(p.get("location") == "form" for p in params)
    info = InterfaceInfo(
        url=primary["url"],
        methods=primary["methods"],
        params=[InterfaceParam(name=p["name"], location=p["location"],
                               type_hint=p.get("type_hint")) for p in params],
        body_type="file" if file_fields else ("form" if has_form else None),
        file_field=file_fields[0] if file_fields else None,
        source="decorator",
    )
    # 多路由时附带其余路由信息（notes），主 URL 取第一条
    if len(routes) > 1:
        info.source = f"decorator({len(routes)} routes)"
    return info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    print(__doc__)
