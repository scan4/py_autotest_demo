#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAPI 运行时接口探测与「符号 ↔ endpoint」匹配器
==================================================
【动机】interface.py 的 URL/method/参数提取是 Django 范式（解析 urls.py 的
path() + 从 view 函数体读 request.method/request.POST）。对 FastAPI 这类
"声明式"框架完全失效（实测 full-stack-fastapi-template 23 个功能点 0 个有 URL）。

【原理】运行中的服务会由框架自己生成机器可读契约（OpenAPI/Swagger）：
  GET {base_url}/openapi.json
它有三个天然优势：
  1. URL 已含全部前缀（API_V1_STR + APIRouter prefix 真实生效），不受代码风格影响
  2. 参数 / 请求体 schema / 状态码都准确
  3. 能捞到静态分析原理上看不到的接口（路由由第三方库命令式注册、handler 动态生成）

【与静态分析的关系】不是替代，是两层模型：
  - 功能点（代码层，静态分析产出）：一个函数符号 + 内部结构（调用链/控制流）
  - endpoint（契约层，OpenAPI 产出）：一个 (method, path) + 参数契约
  join key 是 operationId（FastAPI 默认格式 "{tag}-{函数名}"，内含函数名）。

【匹配策略（从强到弱）】
  1. operation_id 精确匹配：operationId 的函数名段 == 符号函数名，且 tag 与模块名吻合
     —— 同名函数靠 tag 消歧（实测项目里 users.create_user / private.create_user 两个同名）
  2. 函数名单独匹配：无 tag 可比对时，若唯一命中则采用
  3. path 包含函数名：spec 无 operationId 时的兜底（如 /download_file/）
  4. 人工确认：0 个或多个且无法消歧 → 交给前端让用户选
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

# 探测候选路径（按常见度排序；只支持 JSON spec，yaml 需额外解析器，暂不支持）
_SPEC_CANDIDATES = (
    "/openapi.json",
    "/api/openapi.json",
    "/api/v1/openapi.json",
    "/swagger.json",
    "/api/swagger.json",
    "/api/v1/swagger.json",
    "/?format=openapi",          # DRF
)

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}


# ---------------------------------------------------------------------------
# 1. 探测：从运行中的服务拉取 spec
# ---------------------------------------------------------------------------

def probe_openapi(base_url: str, timeout: tuple[float, float] = (5, 30)) -> dict[str, Any]:
    """探测运行中服务的 OpenAPI spec。

    Args:
        base_url: 服务根地址，如 http://localhost:8100（scheme+host+port，不带路径）
        timeout: (连接超时, 读取超时)——连接超时设短，避免卡在不可达地址

    Returns:
        {"ok": bool, "spec_url": str|None, "spec": dict|None, "error": str|None,
         "tried": [尝试过的路径]}
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return {"ok": False, "spec_url": None, "spec": None,
                "error": "base_url 为空", "tried": []}
    if not base.startswith(("http://", "https://")):
        base = "http://" + base

    tried: list[str] = []
    for cand in _SPEC_CANDIDATES:
        url = base + cand
        tried.append(url)
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"Accept": "application/json"})
        except requests.exceptions.RequestException:
            continue
        if resp.status_code != 200:
            continue
        ctype = resp.headers.get("content-type", "")
        if "json" not in ctype and not resp.text.lstrip().startswith("{"):
            continue
        try:
            spec = resp.json()
        except ValueError:
            continue
        if not isinstance(spec, dict) or "paths" not in spec:
            continue                      # 不是 OpenAPI 结构（可能是别的 JSON）
        return {"ok": True, "spec_url": url, "spec": spec, "base_url": base,
                "error": None, "tried": tried}

    return {"ok": False, "spec_url": None, "spec": None,
            "error": f"未在 {base} 探测到 OpenAPI spec（尝试了 {len(tried)} 个候选路径）",
            "tried": tried}


# ---------------------------------------------------------------------------
# 2. 解析：spec → endpoint 列表
# ---------------------------------------------------------------------------

def parse_endpoints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """把 OpenAPI spec 解析成 endpoint 列表。

    Returns:
        [{"url": "/api/v1/items/", "methods": ["GET"], "operation_id": "items-read_items",
          "tag": "items", "params": [{"name","location","type_hint"}], "source": "openapi"}]
        一个 path 挂多个 method 时拆成多条（每个 (method,path) 一个 endpoint）。
    """
    out: list[dict[str, Any]] = []
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            tags = op.get("tags") or []
            params = _extract_params(op)
            out.append({
                "url": path,
                "methods": [method.upper()],
                "operation_id": op.get("operationId") or "",
                "tag": tags[0] if tags else "",
                "params": params,
                "summary": op.get("summary") or "",
                "source": "openapi",
            })
    return out


# 登录端点自动识别：url/operationId 命中关键词 + username/password 表单参数加权
_LOGIN_KEYWORDS = ("login", "access-token", "token", "auth", "signin", "sign-in", "session", "jwt")


def find_login_endpoint(endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 OpenAPI endpoint 列表自动识别登录端点（用于执行前自动换 token）。

    打分信号：POST 方法 + url/operationId 命中关键词 + 参数表含 username/password
    （OAuth2 password flow 的强特征）。取最高分，无命中返回 None。
    """
    best: dict[str, Any] | None = None
    best_score = 0
    for e in endpoints or []:
        if "POST" not in [str(m).upper() for m in (e.get("methods") or [])]:
            continue
        text = (str(e.get("url", "")) + " " + str(e.get("operation_id", ""))).lower()
        score = sum(2 for k in _LOGIN_KEYWORDS if k in text)
        for p in e.get("params") or []:
            if str(p.get("name", "")).lower() in ("username", "password"):
                score += 3
        if score > best_score:
            best, best_score = e, score
    return best


def _extract_params(op: dict[str, Any]) -> list[dict[str, Any]]:
    """从 operation 提取参数：路径/查询/头参数 + 请求体字段。"""
    params: list[dict[str, Any]] = []
    for p in op.get("parameters") or []:
        if not isinstance(p, dict):
            continue
        loc = p.get("in", "query")
        if loc not in ("path", "query", "header", "cookie"):
            continue
        schema = p.get("schema") or {}
        params.append({
            "name": p.get("name", "?"),
            "location": loc,
            "type_hint": schema.get("type") or p.get("required") and "required" or None,
        })
    rb = op.get("requestBody")
    if isinstance(rb, dict):
        content = rb.get("content") or {}
        for ctype, media in content.items():
            loc = "json" if "json" in ctype else ("file" if "form-data" in ctype else "form")
            schema = (media or {}).get("schema") or {}
            props = schema.get("properties")
            if isinstance(props, dict) and props:
                for name, s in props.items():
                    params.append({"name": name, "location": loc,
                                   "type_hint": (s or {}).get("type")})
            else:
                params.append({"name": "(请求体)", "location": loc, "type_hint": ctype})
            break                          # 取第一个 content 类型即可
    return params


# ---------------------------------------------------------------------------
# 3. 匹配：功能点符号 → endpoint 列表
# ---------------------------------------------------------------------------

def _split_operation_id(op_id: str) -> tuple[str, str]:
    """拆 operationId：'items-read_items' -> ('items', 'read_items')。

    Python 标识符不含 '-'，所以按最后一个 '-' 切是安全的。
    无 '-' 时 tag 为空。
    """
    if "-" in op_id:
        tag, name = op_id.rsplit("-", 1)
        return tag, name
    return "", op_id


def match_endpoints(endpoints: list[dict[str, Any]],
                    entry: str, module: str = "") -> dict[str, Any]:
    """为一个功能点符号匹配它对应的 endpoint 列表。

    Args:
        endpoints: parse_endpoints() 产物
        entry: 功能点限定名，如 app.api.routes.items.read_items
        module: 功能点所在模块（用于与 operationId 的 tag 消歧）

    Returns:
        {"endpoints": [匹配到的 endpoint（按置信度排序）],
         "match": "operation_id" | "name" | "path" | "none",
         "ambiguous": bool}      # True = 多个候选且无法消歧，需人工确认
    """
    func_name = entry.rsplit(".", 1)[-1] if "." in entry else entry
    module_last = module.rsplit(".", 1)[-1] if module and "." in module else (module or "")
    module_parts = set((module or "").split("."))

    strong: list[dict[str, Any]] = []      # 策略1：operationId 函数名 + tag 消歧
    by_name: list[dict[str, Any]] = []     # 策略2：仅函数名
    by_path: list[dict[str, Any]] = []     # 策略3：path 含函数名

    for ep in endpoints:
        op_id = ep.get("operation_id") or ""
        tag, op_name = _split_operation_id(op_id)
        if op_name == func_name:
            # tag 与模块名吻合（tag 是模块最后一段，或出现在模块名里）→ 强匹配
            if not tag or tag == module_last or tag in module_parts:
                strong.append(ep)
            else:
                by_name.append(ep)
        elif func_name and op_id and ep.get("url") and ep.get("methods"):
            # 原生 FastAPI 默认格式（无自定义 generate_unique_id）：op_id 是确定性构造——
            # {函数名}{URL 非词字符转下划线}_{method 小写}，如 url=/api/v1/users/ method=POST
            # → create_user_api_v1_users__post。用本 endpoint 的 url/method 反算期望值精确比对：
            # 同名函数（users/private 两处 create_user）因 path 不同各自唯一命中，天然消歧
            import re as _re
            pw = _re.sub(r"\W", "_", str(ep["url"]))
            method0 = str((ep.get("methods") or [""])[0]).lower()
            if op_id == f"{func_name}{pw}_{method0}":
                strong.append(ep)
        elif not op_id and func_name and func_name in (ep.get("url") or ""):
            # spec 无 operationId：靠 path 字段兜底（如 /download_file/）
            by_path.append(ep)

    if strong:
        return {"endpoints": strong, "match": "operation_id", "ambiguous": len(strong) > 1}
    if by_name:
        # 仅函数名命中且唯一 → 可信；多个 → 歧义（同名函数不同模块），交给人工
        urls = {e.get("url") for e in by_name}
        return {"endpoints": by_name, "match": "name", "ambiguous": len(urls) > 1}
    if by_path:
        return {"endpoints": by_path, "match": "path", "ambiguous": len(by_path) > 1}
    return {"endpoints": [], "match": "none", "ambiguous": False}


def build_symbol_map(endpoints: list[dict[str, Any]],
                     features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """批量建立「功能点符号 → endpoint 匹配结果」映射。

    Returns:
        {entry: match_endpoints() 的返回结构}
    """
    out: dict[str, dict[str, Any]] = {}
    for f in features:
        entry = f.get("entry") if isinstance(f, dict) else getattr(f, "entry", "")
        module = f.get("module") if isinstance(f, dict) else getattr(f, "module", "")
        if entry:
            out[entry] = match_endpoints(endpoints, entry, module or "")
    return out


def summarize_probe(endpoints: list[dict[str, Any]],
                    symbol_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """探测结果的统计摘要（给前端/日志展示）。"""
    matched = {e: m for e, m in symbol_map.items() if m["endpoints"]}
    ambiguous = {e: m for e, m in matched.items() if m["ambiguous"]}
    return {
        "endpoint_count": len(endpoints),
        "feature_count": len(symbol_map),
        "matched_count": len(matched),
        "unmatched_features": sorted(e for e, m in symbol_map.items() if not m["endpoints"]),
        "ambiguous_features": sorted(ambiguous),
        # OpenAPI 有、但没有任何功能点匹配上的接口（如第三方库注册的路由）
        "unreferenced_endpoints": [
            {"url": ep["url"], "methods": ep["methods"], "operation_id": ep["operation_id"]}
            for ep in endpoints
            if ep["operation_id"] and not any(
                ep in m["endpoints"] for m in symbol_map.values())
        ],
    }


def render_endpoint_hint(ep: dict[str, Any]) -> str:
    """把一个 endpoint 渲染成一行提示（日志/前端用）。"""
    methods = "/".join(ep.get("methods") or ["?"])
    return f"{methods} {ep.get('url')}  (operationId={ep.get('operation_id') or '-'})"


if __name__ == "__main__":
    print(__doc__)
