#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pythonTest Web 服务（独立，不依赖 TestBrain）
================================================
可视化测试用例平台：文件夹浏览 → 代码分析 → LLM 生成用例 → AI 评审 → 执行 → 结果展示。

技术栈：FastAPI + uvicorn + 内嵌静态前端。LLM 走独立 llm_client（OpenAI 兼容）。

【接口】
    GET  /                    # 前端页面
    GET  /api/dirs            # 列出服务器根目录下的文件夹
    POST /api/dirs/children   # 列出某目录的子目录
    POST /api/analyze         # 分析代码 → 功能点
    POST /api/generate        # 生成测试用例
    POST /api/review          # AI 评审用例
    POST /api/execute         # 执行用例（人工/AI mock）
    GET  /api/health          # 健康检查

【启动】
    pip install -r requirements.txt
    export DEEPSEEK_API_KEY=xxx   # 或配置 pyst/.config
    uvicorn pyst.webapp:app --port 8001
    浏览器访问 http://localhost:8001
"""

import os
import re
import uuid
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core.project_index import ProjectIndex
from .core.callgraph import CallGraph
from .core.features import FeaturePoint, split_features
from .core.generate import generate_test_cases
from .core.openapi import (probe_openapi, parse_endpoints, build_symbol_map,
                           summarize_probe, match_endpoints)
from .eval.executor import execute_suite_with_resources

# ---------------- 静态资源 ----------------
_STATIC_DIR = Path(__file__).parent / "static"

# ---------------- 用例落库 ----------------
_DB_PATH = Path(__file__).parent / "test_cases.db"

# 重新生成时的用例条数上限：评审驱动的动态扩容不能无限涨
# （输出 8192 token ÷ 每条用例约 400~600 token，留余量取 15）
_REGEN_MAX_CASES = 15

app = FastAPI(
    title="pythonTest 测试用例生成平台",
    description="AST 静态分析 → 功能点拆分 → LLM 生成 → AI 评审 → 执行",
    version="0.8.0",
)


# ---------------- 请求模型 ----------------
class DirChildrenRequest(BaseModel):
    path: str


class AnalyzeRequest(BaseModel):
    source_dir: str | None = None
    source: str | None = None
    entries: list[str] = Field(default_factory=list)


class GenerateRequest(AnalyzeRequest):
    provider: str = "deepseek"
    case_count: int = 5
    case_design_methods: str = ""
    case_categories: str = ""
    force: bool = False   # True 时强制重新生成并覆盖，不走缓存
    max_depth: int = 1    # 调用子图搜索深度（1=仅入口，2=含直接下游，...）
    test_level: str = "api"   # "api"=HTTP 用例；"func"=纯函数 call_args 用例
    cover_module: str = ""    # func 模式：被测纯函数模块 .py 路径


class ReviewRequest(AnalyzeRequest):
    provider: str = "deepseek"


class RegenerateRequest(AnalyzeRequest):
    provider: str = "deepseek"
    case_count: int = 10
    entry: str = ""                      # 要重新生成的功能点入口
    review: dict = Field(default_factory=dict)   # 评审意见


class ExecuteRequest(AnalyzeRequest):
    base_url: str = ""     # 被测服务根地址；缺省用会话探测过的 base_url
    auth: dict = Field(default_factory=dict)  # 测试凭证 {login_path, username, password}；可选


class FeedbackRequest(AnalyzeRequest):
    """失败回灌：执行失败结果结构化后回灌 LLM（单 Agent 诊断+修正）。"""
    provider: str = "deepseek"
    entry: str = ""        # 可选：只处理该功能点；空 = 处理全部有失败的功能点


class CoverageRequest(AnalyzeRequest):
    """盲区覆盖检测：被测纯函数模块 + 入口 + 调用参数集。"""
    cover_module: str = ""                          # 被测纯函数模块 .py 绝对路径
    entry: str = ""                                 # 被测函数限定名（对应功能点入口）
    calls: list[dict] = Field(default_factory=list)  # [{"args": [...], "kwargs": {...}}, ...]
    provider: str = "deepseek"


class ProbeRequest(AnalyzeRequest):
    """OpenAPI 运行时接口探测：从运行中的服务拉取接口契约并与功能点建立映射。"""
    base_url: str = ""   # 被测服务根地址，如 http://localhost:8100


# ---------------- 会话状态（内存） ----------------
class Session:
    """一个分析任务的状态：功能点、生成的用例、评审、执行结果。"""
    def __init__(self, source_dir: str):
        self.id = uuid.uuid4().hex[:8]
        self.source_dir = source_dir
        self.created_at = time.time()
        self.features: list[dict] = []
        self.feature_objs: list[FeaturePoint] = []
        self.test_cases: dict[str, list[dict]] = {}
        self.reviews: dict[str, dict] = {}
        self.refined_cases: dict[str, list[dict]] = {}   # 按评审意见改进后的用例（对比用，不覆盖上一步）
        self.results: list[dict] = []
        # OpenAPI 运行时探测（契约层，与静态分析的功能点层互补）
        self.base_url: str = ""                          # 被测服务根地址（探测/执行共用）
        self.openapi_endpoints: list[dict] = []          # 解析出的 endpoint 列表
        self.symbol_map: dict[str, dict] = {}            # 功能点符号 → endpoint 匹配结果
        self.probe_summary: dict = {}                    # 探测统计摘要
        # 测试凭证缓存（执行前用账号密码登录换 token，替换用例里的 <valid_jwt> 占位符）
        self.auth_token: str = ""
        self.auth_time: float = 0.0
        self.auth_config: dict = {}   # {login_path, username, password}——记住凭证，同会话只填一次


_sessions: dict[str, Session] = {}


def _get_or_create_session(source_dir: str) -> Session:
    for s in _sessions.values():
        if s.source_dir == source_dir:
            return s
    s = Session(source_dir)
    _sessions[s.id] = s
    return s


def _cleanup_sessions(max_age: float = 3600):
    """清理超过 max_age 秒的会话。"""
    now = time.time()
    stale = [k for k, s in _sessions.items() if now - s.created_at > max_age]
    for k in stale:
        _sessions.pop(k, None)


# ---------------- 文件夹浏览 ----------------
_ROOT_HINT = "."


def _filter_entries(features: list[FeaturePoint], entries: list[str]) -> list[FeaturePoint]:
    """按 entries 过滤功能点。容忍模块前缀不一致（如 apps.xxx vs xxx）。

    前端浏览目录时模块名不带 apps. 前缀，但用户可能传带前缀的完整名，
    这里用"限定名以 entry 结尾 / entry 以限定名结尾"做宽松匹配。
    """
    if not entries:
        return features
    wanted = set(entries)
    out = []
    for f in features:
        if f.entry in wanted:
            out.append(f)
            continue
        # 后缀/前缀宽松匹配：apps.xxx.views.download_file vs ai_agents.xxx.views.download_file
        if any(f.entry.endswith(w) or w.endswith(f.entry) for w in wanted):
            out.append(f)
    return out


def _resolve_session_entry(session, entry: str) -> str | None:
    """在 session 的 test_cases / features 里定位真实的 entry key。

    兼容 apps.xxx 与 xxx 前缀差异：精确匹配失败时做后缀/前缀宽松匹配。
    """
    if entry in session.test_cases:
        return entry
    # 在 features 里找匹配的 entry（其 entry 一定在 test_cases 里）
    for f in session.features:
        if f["entry"] == entry:
            return entry
        if f["entry"].endswith(entry) or entry.endswith(f["entry"]):
            return f["entry"]
    # 直接在 test_cases 的 key 里宽松匹配
    for k in session.test_cases:
        if k.endswith(entry) or entry.endswith(k):
            return k
    return None


# 排除的目录名（隐藏/缓存/虚拟等，避免列表过长和权限错误）
_SKIP_DIRS = {
    ".", "__pycache__", "node_modules", "venv", ".git", ".idea", ".vscode",
    ".cache", ".config", ".local", ".m2", ".npm", ".gradle",
    # 系统/虚拟文件系统（根目录下的不可用目录）
    "proc", "sys", "dev", "run", "boot", "lost+found", "snap",
}

# 默认根目录：HOME 优先，其次 /root 或 /
_DEFAULT_ROOT = os.environ.get("HOME") or "/root"
if not os.path.isdir(_DEFAULT_ROOT):
    _DEFAULT_ROOT = "/"


def _list_dir_contents(path: str) -> dict:
    """列出某目录下的子目录，附带父目录路径（用于向上导航）。

    Returns:
        {"dirs": [...], "parent": <上级目录或None>, "current": path}
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {path}")
    dirs = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir and entry.name not in _SKIP_DIRS and not entry.name.startswith("."):
                dirs.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        pass
    # 父目录：除非已在根 /
    parent = None
    try:
        resolved = p.resolve()
        parent_path = resolved.parent
        parent = str(parent_path) if parent_path != resolved else None
    except OSError:
        parent = None
    return {"dirs": dirs, "parent": parent, "current": str(p)}


@app.get("/")
def index():
    idx_file = _STATIC_DIR / "index.html"
    if idx_file.exists():
        return HTMLResponse(idx_file.read_text(encoding="utf-8"))
    return {"message": "前端页面缺失"}


@app.get("/api/dirs")
def list_root_dirs(path: str | None = None):
    """列出根目录下的顶层文件夹。path 不传时从 HOME 目录开始。"""
    start = path if path else _DEFAULT_ROOT
    return _list_dir_contents(start)


@app.post("/api/dirs/children")
def list_children(req: DirChildrenRequest):
    return _list_dir_contents(req.path)


# ---------------- 代码分析 ----------------
def _build_func_features(cover_module: str, entries: list[str]):
    """为纯函数入口构造 FeaturePoint 档案（func 模式，绕过 split_features）。

    纯函数入口（如工具/校验函数）没有 views/api/路由装饰器，不会被 split_features
    自动识别。这里用 analyze_file 直接分析被测模块，对每个指定入口取同名函数，
    构造含 control_sites（带行区间+语义）的档案，供生成 call_args 用例 + 覆盖分析。

    Returns:
        list[FeaturePoint]
    """
    from .core.ast_analysis import analyze_file as _analyze_file
    from .core.features import FeaturePoint

    finfo = _analyze_file(Path(cover_module))
    module = finfo.path

    def _find_func(func_name: str):
        for f in finfo.top_functions:
            if f.name == func_name:
                return f
        for cls in finfo.classes:
            for m in cls.methods:
                if m.name == func_name:
                    return m
        return None

    features: list[FeaturePoint] = []
    for entry in entries:
        func_name = entry.rsplit(".", 1)[-1]
        f = _find_func(func_name)
        if f is None:
            raise HTTPException(status_code=404,
                                detail=f"在 {cover_module} 中找不到函数 {func_name}")
        fp = FeaturePoint(
            entry=entry,
            kind=f.kind,
            signature=f"def {f.name}({', '.join(f.args)})" + (f" -> {f.returns}" if f.returns else ""),
            docstring=f.docstring,
            module=module,
            line_range=(f.start, f.end),
            decorators=f.decorators,
            call_edges=[],       # func 模式不展开调用子图（覆盖聚焦入口自身控制流）
            external_calls=[],
            control_sites=[
                {"kind": c.kind, "line": c.line, "cond": c.cond, "branch": c.branch,
                 "start": c.start, "end": c.end, "semantics": c.semantics, "path": c.path}
                for c in f.controls
            ],
            interface=None,
            max_depth=1,
        )
        features.append(fp)
    return features


def _build_index(source_dir: str | None, source: str | None):
    from .core.ast_analysis import analyze_file
    from .core.project_index import FileIndex, ImportVisitor
    import ast, tempfile

    if source_dir:
        root = Path(source_dir)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"目录不存在: {source_dir}")
        index = ProjectIndex().build(root)
        graph = CallGraph(index).build()
        return index, graph

    if source:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source)
            tmp_path = f.name
        try:
            info = analyze_file(Path(tmp_path))
            module = "source"
            index = ProjectIndex()
            index.module_paths[module] = tmp_path
            tree = ast.parse(source)
            fi = FileIndex(path=tmp_path, module_name=module)
            visitor = ImportVisitor()
            visitor.visit(tree)
            fi.imports = visitor.imports
            fi.symbols = index._collect_symbols(info, module)
            fi.calls = index._collect_calls(info, module)
            index.files[module] = fi
            for s in fi.symbols:
                index.symbols[s.qualname] = s
            graph = CallGraph(index).build()
            return index, graph
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"代码语法错误: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise HTTPException(status_code=400, detail="请提供 source_dir 或 source")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pythonTest", "version": "0.8.0"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """分析代码 → 返回功能点档案（含接口信息）。

    顺序无关：若用户在分析前做过 OpenAPI 探测（_probe_cache 有缓存），
    这里自动把接口映射到功能点——先探测后分析、先分析后探测皆可。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    _cleanup_sessions()
    index, graph = _build_index(req.source_dir, None)
    features = split_features(index, graph, entries=req.entries)

    # 存会话
    session = _get_or_create_session(req.source_dir)
    session.features = [f.to_dict() for f in features]
    session.feature_objs = features

    # 自动补映射（用户先探测过后分析的顺序）
    probed = None
    if _probe_cache:
        probed = max(_probe_cache.values(), key=lambda c: c.get("ts", 0))
        session.base_url = probed["base_url"]
        session.openapi_endpoints = probed["endpoints"]
        session.symbol_map = build_symbol_map(probed["endpoints"], session.features)
        session.probe_summary = summarize_probe(probed["endpoints"], session.symbol_map)
        for fd in session.features:
            m = session.symbol_map.get(fd.get("entry")) or {}
            if m.get("endpoints"):
                fd["interfaces"] = m["endpoints"]
                fd["interface"] = m["endpoints"][0]
                fd["interface_source"] = "openapi"
                fd["interface_ambiguous"] = bool(m.get("ambiguous"))

    return {"session_id": session.id, "count": len(features), "features": session.features,
            "probed": probed["base_url"] if probed else None,
            "probe_summary": session.probe_summary or None}


def _enrich_fp_with_openapi(fp: "FeaturePoint", session: Session) -> None:
    """用 OpenAPI 探测结果为功能点补充接口契约（供生成用例的 prompt 使用）。

    两层模型：功能点（静态，代码结构）+ endpoint（运行时，HTTP 契约）。
    只在探测命中时覆盖 Django 静态提取的 interface；未命中保持原样，
    这样 Django 项目不受影响，FastAPI 项目拿到真实 URL。
    """
    if not session.openapi_endpoints:
        return
    m = match_endpoints(session.openapi_endpoints, fp.entry, fp.module)
    if m.get("endpoints"):
        fp.interfaces = m["endpoints"]
        fp.interface = m["endpoints"][0]     # 主接口（多路由时取第一个，全部在 interfaces）


# OpenAPI 探测缓存（base_url → 探测结果）：探测独立于源码分析，
# 用户可以"先探测后分析"或"先分析后探测"，两侧顺序无关，分析时自动补映射
_probe_cache: dict[str, dict] = {}


@app.post("/api/probe")
def probe(req: ProbeRequest):
    """探测运行中服务的 OpenAPI 接口契约。

    base_url 是唯一必需项——探测本身是纯网络操作，与源码无关。
    source_dir 可选：提供时把探测结果映射到该会话的功能点（符号 ↔ endpoint）；
    不提供则只返回接口清单（缓存在服务端，之后分析代码时自动补上映射）。
    """
    if not req.base_url:
        raise HTTPException(status_code=400, detail="请提供被测服务根地址 base_url（如 http://localhost:8100）")

    result = probe_openapi(req.base_url)
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error") or "探测失败")

    endpoints = parse_endpoints(result["spec"])
    normalized = result.get("base_url") or req.base_url
    _probe_cache[normalized] = {"endpoints": endpoints, "spec_url": result.get("spec_url"),
                                "base_url": normalized, "ts": time.time()}

    resp: dict[str, Any] = {
        "base_url": normalized,          # 归一化后的地址（补了 http://）
        "spec_url": result.get("spec_url"),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
    }

    # 提供了 source_dir → 顺带做符号映射（探测与映射解耦，映射是可选增强）
    if req.source_dir:
        _cleanup_sessions()
        session = _get_or_create_session(req.source_dir)
        if not session.features:
            index, graph = _build_index(req.source_dir, None)
            features = split_features(index, graph, entries=req.entries)
            session.features = [f.to_dict() for f in features]
            session.feature_objs = features

        session.base_url = normalized
        session.openapi_endpoints = endpoints
        session.symbol_map = build_symbol_map(endpoints, session.features)
        session.probe_summary = summarize_probe(endpoints, session.symbol_map)
        for fd in session.features:
            m = session.symbol_map.get(fd.get("entry")) or {}
            if m.get("endpoints"):
                fd["interfaces"] = m["endpoints"]
                fd["interface"] = m["endpoints"][0]
                fd["interface_source"] = "openapi"
                fd["interface_ambiguous"] = bool(m.get("ambiguous"))
        resp.update({"session_id": session.id, "summary": session.probe_summary,
                     "features": session.features})
    return resp


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """分析 + 生成测试用例（按入口分组）。重复生成命中 SQLite 缓存，不重复调 LLM。

    test_level="api"：走 HTTP 入口识别（split_features），输出 request 用例。
    test_level="func"：纯函数入口（无 views/api/路由装饰器，不会被 split_features 识别），
        用 analyze_file 从 cover_module 直接构造 feature，输出 call_args 用例。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")

    # ---- func 模式：纯函数入口用 analyze_file 构造 feature ----
    if req.test_level == "func":
        if not req.cover_module:
            raise HTTPException(status_code=400, detail="func 模式需要提供被测纯函数模块路径 cover_module")
        if not req.entries:
            raise HTTPException(status_code=400, detail="func 模式需要指定被测函数入口 entries")
        features = _build_func_features(req.cover_module, req.entries)
        index = None
    else:
        index, graph = _build_index(req.source_dir, None)
        features = split_features(index, graph, entries=req.entries, max_depth=req.max_depth)
        features = _filter_entries(features, req.entries)
    if not features:
        raise HTTPException(status_code=404, detail="未识别到任何功能点")

    # OpenAPI 契约注入（仅 api 模式）：功能点（静态层）+ endpoint（运行时层），
    # 让 prompt 拿到真实 URL/method/参数，而不是 LLM 猜测。func 模式是纯函数，不注入。
    session = _get_or_create_session(req.source_dir)
    if req.test_level != "func" and session.openapi_endpoints:
        for fp in features:
            _enrich_fp_with_openapi(fp, session)

    result: dict[str, list[dict]] = {}
    cached_count = 0
    stale_count = 0
    changed_details: dict[str, list[str]] = {}
    warnings: dict[str, str] = {}   # 生成预警（如 api 模式遇非 HTTP 入口）
    from .storage.db import TestCaseStore
    store = TestCaseStore(_DB_PATH)
    for fp in features:
        # 单个功能点整体容错：指纹/缓存/生成/落库任一步失败，只记录该功能点，不中断整组
        try:
            # 算当前代码指纹（覆盖下游调用链）；func 模式 index 为 None，跳过指纹缓存
            fp_fingerprint = fp.fingerprint(index) if index is not None else ""
            cached = None

            # 缓存判断：同 source_dir + entry 已有落库用例（仅 api 模式走缓存）
            if index is not None and not req.force:
                cached = store.get_by_entry_source(fp.entry, req.source_dir)
                if cached:
                    # 对比指纹：相同 → 代码没变，命中缓存
                    old_fp = cached[-1].fingerprint if cached else ""
                    if old_fp and old_fp == fp_fingerprint:
                        result[fp.entry] = [c.to_dict() for c in cached]
                        cached_count += 1
                        continue
                    # 指纹不同 → 相关代码变了，缓存失效，需要重新生成
                    stale_count += 1
                    changed_details[fp.entry] = ["相关代码已变化（下游调用链/控制流/接口/签名），缓存失效，已重新生成"]
            # 目标条数：用户输入只约束"首次生成"；库里已有批次（可能是回灌/重新生成
            # 扩容过的）不应被更小的输入缩水——重新生成时取 max(输入, 现有条数)，封顶 15
            # （与评审驱动的动态扩容同语义，防止回灌扩出来的用例集被低条数生成覆盖掉）
            target_count = req.case_count
            if index is not None and cached:
                target_count = min(_REGEN_MAX_CASES, max(target_count, len(cached)))
            cases = generate_test_cases(
                fp, provider=req.provider, case_count=target_count,
                case_design_methods=req.case_design_methods,
                case_categories=req.case_categories,
                test_level=req.test_level,
            )
            case_dicts = [c.to_dict() for c in cases]
            result[fp.entry] = case_dicts
            # api 模式 + 非 HTTP 入口 → 预警（该入口无接口信息，request 可能不可执行）
            if req.test_level == "api" and not fp.interface:
                warnings[fp.entry] = ("该入口未提取到 HTTP 接口（可能是纯函数/非 HTTP 入口），"
                                      "API 模式生成的 request 可能不可执行；建议改用「纯函数用例」模式重新生成")
            # 落库：**走了生成就无条件覆盖**（删旧再存新）——旧条件
            # `any(s.fingerprint and s.fingerprint != 当前)` 有漏洞：回灌/regenerate 落库
            # 的批次无指纹，导致既不能证明"代码没变"（缓存 miss），又不触发覆盖，
            # 每次生成都往库里追加（实测累积出 15 条）。现在语义收敛为：
            # 缓存命中 → 保留旧批；走到生成 → 旧批必然作废，完整替换
            store.delete_by_entry_source(fp.entry, req.source_dir)
            store.save_test_cases(fp.entry, fp.module, case_dicts,
                                  source_dir=req.source_dir, fingerprint=fp_fingerprint)
        except Exception as e:
            import traceback as _tb
            result[fp.entry] = [{"description": f"[生成失败] {e}", "test_steps": [], "expected_results": []}]
            changed_details[fp.entry] = [_tb.format_exc(limit=2)]
    store.close()

    session = _get_or_create_session(req.source_dir)
    session.test_cases = result
    return {"feature_count": len(features), "test_cases": result,
            "cached_count": cached_count, "stale_count": stale_count,
            "cached": cached_count > 0, "changed": changed_details, "warnings": warnings}


@app.post("/api/review")
def review(req: ReviewRequest):
    """AI 评审生成的测试用例。"""
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    from .eval.review import review_test_cases

    session = _get_or_create_session(req.source_dir)
    if not session.test_cases:
        # 若还没生成过，先生成
        index, graph = _build_index(req.source_dir, None)
        features = split_features(index, graph, entries=req.entries)
        for fp in features:
            try:
                cases = generate_test_cases(fp, provider=req.provider, case_count=5)
                session.test_cases[fp.entry] = [c.to_dict() for c in cases]
            except Exception:
                continue

    reviews: dict[str, dict] = {}
    for entry, cases in session.test_cases.items():
        if not cases:
            continue
        # 找对应功能点档案
        fp = next((f for f in session.features if f["entry"] == entry), None)
        if not fp:
            fp = {"entry": entry, "signature": entry, "docstring": "", "control_sites": []}
        try:
            reviews[entry] = review_test_cases(fp, cases, provider=req.provider)
        except Exception as e:
            reviews[entry] = {"score": 0, "strengths": [], "weaknesses": [str(e)],
                              "suggestions": [], "missing_scenarios": [], "recommendation": "评审失败", "comments": str(e)}
    session.reviews = reviews
    # 评审结果挂到落库的用例上
    try:
        from .storage.db import TestCaseStore
        store = TestCaseStore(_DB_PATH)
        for entry, rv in reviews.items():
            store.attach_review(entry, rv)
        store.close()
    except Exception:
        pass   # 落库失败不影响评审返回
    return {"reviews": reviews}


@app.post("/api/regenerate")
def regenerate(req: RegenerateRequest):
    """根据评审意见重新生成测试用例（修正 + 补充缺失场景）。

    输入：功能点档案 + 原用例 + 评审意见 → 发给 LLM → 返回改进后的用例。
    """
    from .core.generate import refine_test_cases

    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    if not req.entry:
        raise HTTPException(status_code=400, detail="请指定要重新生成的功能点 entry")

    session = _get_or_create_session(req.source_dir)

    # 定位真实的 entry key（兼容 apps.xxx vs xxx 前缀差异）
    real_entry = _resolve_session_entry(session, req.entry)
    if not real_entry:
        raise HTTPException(status_code=404, detail=f"该功能点 {req.entry} 没有已生成的用例")

    # 取原用例
    original_cases = session.test_cases.get(real_entry, [])
    if not original_cases:
        raise HTTPException(status_code=404, detail=f"该功能点 {real_entry} 没有已生成的用例")

    # 取功能点档案
    fp = next((f for f in session.features if f["entry"] == real_entry), None)
    if not fp:
        fp = {"entry": real_entry, "signature": real_entry, "docstring": "", "control_sites": [], "interface": None}

    # 取评审意见（前端传入，或从 session 取）
    review = req.review or session.reviews.get(real_entry, {})

    # 评审驱动的动态扩容：缺失场景数 = 需要补充的用例信号。
    # 用户初设条数只是"下限"——评审发现场景不足时按 原条数+缺失场景数 放大，
    # 但封顶 _REGEN_MAX_CASES，防止输出再次打爆 max_tokens（问题 15.4）。
    missing_count = len(review.get("missing_scenarios") or [])
    base_count = req.case_count or 5
    effective_case_count = min(_REGEN_MAX_CASES,
                               max(base_count, len(original_cases) + missing_count))

    try:
        refined = refine_test_cases(
            fp, original_cases, review, provider=req.provider, case_count=effective_case_count)
        refined_dicts = [c.to_dict() for c in refined]

        # 计算 diff：按 description 匹配，标记新增/保留/删除
        #   原用例 -> 被删除的（红）；新用例 -> 新增的（绿）
        orig_descs = {c.get("description", ""): c for c in original_cases}
        new_descs = {c.get("description", ""): c for c in refined_dicts}
        removed = [c for desc, c in orig_descs.items() if desc and desc not in new_descs]
        added = [c for desc, c in new_descs.items() if desc and desc not in orig_descs]

        # 改进后的用例【提升为当前版本】：后续的重新评审 / AI 执行都使用它。
        # （修复：之前只存 refined_cases 历史版本，session.test_cases 仍是第一次生成的用例，
        #   导致重新生成后"重新评审/AI 执行"用的都是旧用例）
        session.refined_cases[real_entry] = refined_dicts   # 历史版本保留（供对比）
        session.test_cases[real_entry] = refined_dicts      # 当前版本
        # 落库改进后的用例（覆盖旧批次：重新生成语义上是"替换当前版本"，
        # 不删旧会导致库里同 entry 累积多批全部显示，前端出现"重复"用例）
        try:
            from .storage.db import TestCaseStore
            store = TestCaseStore(_DB_PATH)
            store.delete_by_entry_source(real_entry, req.source_dir)
            store.save_test_cases(real_entry, fp.get("module", ""), refined_dicts, source_dir=req.source_dir)
            store.close()
        except Exception:
            pass

        return {
            "entry": real_entry,
            "original_cases": original_cases,   # 左边：上次结果
            "refined_cases": refined_dicts,     # 右边：这次结果
            "removed_count": len(removed),
            "added_count": len(added),
            "removed": removed,
            "added": added,
            "missing_count": missing_count,             # 评审发现的缺失场景数
            "effective_case_count": effective_case_count,  # 本轮目标条数（动态扩容后）
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新生成失败: {e}")


# ---------------- 多轮迭代（Agent 闭环第③步前置：执行→回灌→再执行自动循环） ----------------
def _resolve_token(session, base: str, auth: dict) -> str:
    """测试凭证 → 真实 token（执行与多轮迭代共用）。

    凭证记住到会话（只填一次）；登录路径可留空——从 OpenAPI 探测结果自动识别；
    token 缓存 50 分钟复用；登录失败显式报错（不静默产生误导性 401/403）。
    """
    auth = auth or {}
    if auth.get("username") and auth.get("password"):
        session.auth_config = auth          # 同会话记住，后续执行不再填
    elif session.auth_config:
        auth = session.auth_config          # 本次没传 → 复用记住的凭证
    if not (auth.get("username") and auth.get("password")):
        return ""
    login_path = (auth.get("login_path") or "").strip()
    if not login_path:
        from .core.openapi import find_login_endpoint
        ep = find_login_endpoint(session.openapi_endpoints)
        if not ep:
            raise HTTPException(
                status_code=400,
                detail="无法自动识别登录端点（请先在步骤1做接口探测，或手动填写登录路径）")
        login_path = ep["url"]
    fresh = (time.time() - session.auth_time) < 50 * 60
    if fresh and session.auth_token:
        return session.auth_token
    from .eval.executor import fetch_login_token
    try:
        token = fetch_login_token(base, login_path, auth["username"], auth["password"])
        session.auth_token = token
        session.auth_time = time.time()
        return token
    except ValueError as e:
        raise HTTPException(status_code=400,
                            detail=f"测试凭证登录失败（token 未替换，占位符用例将 401/403）: {e}")


def _record_bug_findings(session, source_dir: str, results: list[dict]) -> None:
    """同步 bug 注册表：①potential_bug 失败登记（幂等）；②注册表命中的失败
    强制覆盖分类为 potential_bug——已知缺陷的复现用例，其分类由注册表说了算，
    不随规则措辞/LLM 诊断漂移（实测空白用例曾被标成"用例缺陷"）。"""
    from .eval.feedback import CAT_BUG
    try:
        from .storage.db import TestCaseStore
        store = TestCaseStore(_DB_PATH)
    except Exception:
        return
    try:
        for r in results:
            if r.get("category") == CAT_BUG:
                store.upsert_bug_finding(
                    r.get("entry", ""), source_dir, r.get("description", ""),
                    r.get("request"), r.get("expected_status"),
                    {"actual_status": r.get("status"),
                     "response_snippet": (r.get("response_snippet") or "")[:300],
                     "recorded_at_round": time.strftime("%Y-%m-%d %H:%M:%S")})
        # 注册表命中 → 覆盖分类（已知缺陷复现用例持续失败是预期行为）
        findings = {(f["entry"], f["description"]): f
                    for f in store.list_bug_findings(source_dir=source_dir)}
        for r in results:
            if r.get("verdict") not in ("FAIL", "ERROR", "SKIPPED"):
                continue
            f = findings.get((r.get("entry", ""), r.get("description", "")))
            if f:
                r["category"] = CAT_BUG
                r["category_label"] = "疑似被测系统缺陷"
                r["category_hint"] = (f"已在缺陷注册表中（#{f['id']}，登记于 {f['created_at']}）——"
                                      "该用例是复现用例，持续失败为预期行为；"
                                      "服务端修复或确认误报时请解除登记")
    finally:
        store.close()


def _execute_all(session, base: str, token: str, source_dir: str = "",
                 log: Callable[[str], None] | None = None) -> tuple[list[dict], dict[str, int]]:
    """执行会话全部用例（真实 HTTP）+ 失败条目附规则初分类 + potential_bug 登记。"""
    from .eval.feedback import CAT_LABEL, classify_failure
    cases: list[dict] = []
    for entry, cs in session.test_cases.items():
        for c in cs:
            cases.append({**c, "_entry": entry})
    results, stats = execute_suite_with_resources(cases, base, token=token, log=log)
    for r in results:
        if r.get("verdict") in ("FAIL", "ERROR", "SKIPPED"):
            cat, hint = classify_failure(r)
            r["category"] = cat
            r["category_label"] = CAT_LABEL.get(cat, cat)
            r["category_hint"] = hint
    session.results = results
    if source_dir:
        _record_bug_findings(session, source_dir, results)
    return results, stats


def _failed_count(stats: dict[str, int]) -> int:
    return sum(stats.get(v, 0) for v in ("FAIL", "ERROR", "SKIPPED"))


def _feedback_core(session, source_dir: str, provider: str, entry_filter: str = "",
                   previous_attempts: list[dict] | None = None) -> dict:
    """失败回灌核心（/api/feedback 与多轮迭代共用）。

    previous_attempts：前几轮已尝试未成功的修正摘要（轮次间记忆，防 LLM 重复同方案）。
    """
    from .eval.feedback import (structure_failures, feedback_refine_cases,
                                CAT_BUG)

    structured_all = structure_failures(session.results)
    failures_by_entry: dict[str, list[dict]] = {}
    for f in structured_all["failures"]:
        failures_by_entry.setdefault(f["entry"], []).append(f)
    if entry_filter:
        real = _resolve_session_entry(session, entry_filter)
        if not real:
            raise HTTPException(status_code=404, detail=f"功能点 {entry_filter} 不在当前会话中")
        failures_by_entry = {k: v for k, v in failures_by_entry.items()
                             if k == real or k.endswith(real) or real.endswith(k)}
    if not failures_by_entry:
        return {"entries": {}, "global_stats": structured_all["stats"],
                "aggregate_hints": structured_all["aggregate_hints"]}

    # 把执行结果挂回原用例（按 description 匹配），供 prompt 展示"执行结果"上下文
    results_by_key: dict[tuple, dict] = {}
    for r in session.results:
        if r.get("verdict") in ("FAIL", "ERROR", "SKIPPED"):
            results_by_key[(r.get("entry", ""), r.get("description", ""))] = r

    out: dict[str, dict] = {}
    store = None
    try:
        from .storage.db import TestCaseStore
        store = TestCaseStore(_DB_PATH)
    except Exception:
        pass

    for entry, failures in failures_by_entry.items():
        real_entry = _resolve_session_entry(session, entry) or entry
        cases = session.test_cases.get(real_entry, [])
        if not cases:
            out[real_entry] = {"error": "该功能点没有已生成的用例，无法回灌"}
            continue
        cases_with_exec = []
        for c in cases:
            r = results_by_key.get((entry, c.get("description", "")))
            cases_with_exec.append({**c, "_exec": r} if r else dict(c))
        structured = {"failures": failures, "stats": structured_all["stats"],
                      "aggregate_hints": structured_all["aggregate_hints"],
                      "previous_attempts": previous_attempts or []}
        # bug 注册表查询提前（bug_repro 与回填都要用）：已被删光、当轮无执行痕迹的
        # potential_bug 用例也能从持久化档案恢复
        registry_cases: dict[str, dict] = {}
        if store is not None:
            try:
                for f in store.list_bug_findings(source_dir=source_dir):
                    if f["entry"] == real_entry:
                        registry_cases[f["description"]] = f
            except Exception:
                pass
        # 已知缺陷复现用例单列（prompt 层防护）：禁止 LLM 修改/改预期（假修复）
        bug_repro: list[dict] = []
        for f in failures:
            if f.get("category") == CAT_BUG:
                orig = next((c for c in cases if c.get("description") == f["description"]), None)
                if orig:
                    bug_repro.append({"description": orig.get("description", ""),
                                      "expected_status": orig.get("expected_status"),
                                      "request": orig.get("request"),
                                      "evidence": f.get("response_snippet", "")[:150]})
        for desc, reg in registry_cases.items():
            if all(b["description"] != desc for b in bug_repro):
                bug_repro.append({"description": desc, "expected_status": reg.get("expected_status"),
                                  "request": reg.get("request"),
                                  "evidence": (reg.get("evidence") or {}).get("response_snippet", "")[:150]})
        structured["bug_repro"] = bug_repro
        fp = next((f for f in session.features if f["entry"] == real_entry), None)
        if not fp:
            fp = {"entry": real_entry, "signature": real_entry, "docstring": "",
                  "control_sites": [], "interface": None}
        case_count = min(_REGEN_MAX_CASES, max(len(cases), 8))
        try:
            diagnosis, refined_dicts = feedback_refine_cases(
                fp, cases_with_exec, structured, provider=provider, case_count=case_count)
        except Exception as e:
            import traceback as _tb
            out[real_entry] = {"error": f"回灌失败: {e}", "traceback": _tb.format_exc(limit=2)}
            continue

        orig_descs = {c.get("description", "") for c in cases}
        new_descs = {c.get("description", "") for c in refined_dicts}
        added = [c for c in refined_dicts if c.get("description") and c["description"] not in orig_descs]
        removed = [c for c in cases if c.get("description") and c["description"] not in new_descs]

        # potential_bug 用例强制回填（确定性保护，不信任 prompt 的"原样保留"承诺）：
        # 规则判为疑似系统缺陷的失败用例，若 LLM 修正集中消失（被删/被替换）→ 原样回填。
        # 背景（实测激励错位）：无改进判定以失败数为准，LLM 删掉修不好的 bug 复现用例
        # 反而能"降失败数"——bug 的证据就是这么丢的。
        # 此外从【bug 注册表】复活：已被删光、当轮无执行痕迹的 potential_bug 用例，
        # 也能从持久化档案恢复（注册表在每次执行时登记，跨迭代会话）
        bug_descs = {f["description"] for f in failures if f.get("category") == CAT_BUG}
        bug_descs |= set(registry_cases)
        restored: list[str] = []
        if bug_descs:
            have = {c.get("description") for c in refined_dicts}
            for desc in bug_descs:
                if desc in have:
                    continue
                orig = next((c for c in cases if c.get("description") == desc), None)
                if orig is not None:
                    refined_dicts.append(dict(orig))
                    restored.append(desc)
                    continue
                reg = registry_cases.get(desc)
                if reg:
                    refined_dicts.append({
                        "description": reg["description"],
                        "request": reg.get("request"),
                        "expected_status": reg.get("expected_status"),
                        "test_steps": [f"1. 复现已知缺陷（bug 注册表 #{reg['id']}，登记于 {reg['created_at']}）"],
                        "expected_results": [f"1. 观察到与缺陷记录一致的响应（{ (reg.get('evidence') or {}).get('response_snippet', '')[:80] }…）"],
                    })
                    restored.append(desc)
            if restored:
                added = [c for c in refined_dicts if c.get("description") and c["description"] not in orig_descs]
                removed = [c for c in cases if c.get("description") and c["description"] not in
                           {x.get("description") for x in refined_dicts}]
        # 复现用例字段强制恢复（最后防线）：LLM 若把 potential_bug 判成 case_defect 并
        # 改预期（假修复），这里按注册表/原用例把 request/expected_status 拨回原状
        for b in bug_repro:
            for c in refined_dicts:
                if c.get("description") == b["description"]:
                    if c.get("expected_status") != b.get("expected_status"):
                        c["expected_status"] = b.get("expected_status")
                    if b.get("request") and c.get("request") != b.get("request"):
                        c["request"] = b.get("request")
                    c["_bug_repro_restored"] = True

        # 复现用例变体收敛：同一缺陷只保留一条复现用例。
        # 实测 LLM 会给已知缺陷"复制"变体用例（如"无 Authorization 头"变体）——描述
        # 带后缀逃过精确去重，导致 failed_bug 虚高与诊断重复输出
        import re as _re

        def _norm_desc(s: str) -> str:
            return _re.sub(r"（[^）]*）|\[[^\]]*\]|\s+", "", s or "")

        def _same_repro(desc: str, base_desc: str) -> bool:
            a, b = _norm_desc(desc), _norm_desc(base_desc)
            return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))

        variant_pruned: list[str] = []
        for reg_desc, reg in registry_cases.items():
            anchor = next((c for c in refined_dicts if c.get("description") == reg_desc), None)
            if anchor is None:
                continue
            variants = [c for c in refined_dicts
                        if c is not anchor and _same_repro(c.get("description", ""), reg_desc)]
            for v in variants:
                refined_dicts.remove(v)
            variant_pruned.extend(v.get("description", "") for v in variants)

        session.test_cases[real_entry] = refined_dicts
        if store is not None:
            try:
                store.delete_by_entry_source(real_entry, source_dir)
                store.save_test_cases(real_entry, fp.get("module", ""), refined_dicts,
                                      source_dir=source_dir)
            except Exception:
                pass
        out[real_entry] = {
            "diagnosis": diagnosis,
            "failure_stats": {"failed": len(failures),
                              "by_category": {f["category"]: sum(1 for x in failures if x["category"] == f["category"])
                                              for f in failures}},
            "restored_potential_bug": restored,
            "variant_pruned": variant_pruned,
            "original_cases": cases,
            "refined_cases": refined_dicts,
            "added": added, "removed": removed,
            "added_count": len(added), "removed_count": len(removed),
        }
    if store is not None:
        store.close()

    return {"entries": out,
            "global_stats": structured_all["stats"],
            "aggregate_hints": structured_all["aggregate_hints"]}


@app.post("/api/execute")
def execute(req: ExecuteRequest):
    """执行测试用例：**真实 HTTP 执行**（复用 pyst/eval/executor）。

    需要 base_url（会话探测过或请求传入）；带断言判定 PASS/FAIL/WARN；
    非 HTTP 用例（CALL 伪请求/无 URL/纯函数）自动 SKIPPED，单条异常不中断整批。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    session = _get_or_create_session(req.source_dir)
    if not session.test_cases:
        raise HTTPException(status_code=400, detail="请先生成测试用例")

    base = (req.base_url or session.base_url).strip()
    if not base:
        raise HTTPException(status_code=400,
                            detail="执行需要被测服务地址 base_url（请在步骤1填写并探测，或请求传入）")
    token = _resolve_token(session, base, req.auth)
    results, stats = _execute_all(session, base, token)
    return {"results": results, "stats": stats, "base_url": base}



@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """失败回灌（Agent 闭环第②步）：执行失败结果结构化 → 单 Agent 诊断 + 修正用例。

    流程：取 session.results 里的失败条目（FAIL/ERROR/SKIPPED）→ 按功能点分组 →
    规则初分类（case_defect / potential_bug / env）→ 回灌 LLM 输出诊断 + 修正后完整用例集 →
    修正用例提升为当前版本（后续评审/执行使用），并落库。
    铁律：疑似被测系统缺陷（potential_bug）的用例原样保留，不做"改预期让它变绿"的假修复。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    session = _get_or_create_session(req.source_dir)
    if not session.results:
        raise HTTPException(status_code=400, detail="请先执行测试用例（失败回灌需要真实执行结果）")
    result = _feedback_core(session, req.source_dir, req.provider, entry_filter=req.entry)
    if not result["entries"]:
        raise HTTPException(status_code=400, detail="没有需要回灌的失败用例（FAIL/ERROR/SKIPPED）")
    return result


class IterateRequest(ExecuteRequest):
    """多轮迭代：执行 → 失败回灌 → 再执行，自动循环直到收敛/无改进/达到轮次上限。"""
    provider: str = "deepseek"
    max_rounds: int = 3


@app.post("/api/iterate")
def iterate(req: IterateRequest):
    """多轮迭代（Agent 闭环第③步第一小节）：一键"执行→回灌→再执行"自动循环。

    停止条件（按序）：
      1. 全部通过 → 收敛
      2. 本轮失败数 >= 上轮（回灌未产生改进）→ 提前停止，防死循环烧 token
      3. 达到 max_rounds（回灌修正次数上限）
    轮次间记忆：每轮回灌产生的诊断摘要累积传给下一轮 prompt（"上轮已试过 X 方案仍失败，请换思路"）。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    session = _get_or_create_session(req.source_dir)
    if not session.test_cases:
        raise HTTPException(status_code=400, detail="请先生成测试用例")
    base = (req.base_url or session.base_url).strip()
    if not base:
        raise HTTPException(status_code=400,
                            detail="执行需要被测服务地址 base_url（请在步骤1填写并探测，或请求传入）")
    max_rounds = max(1, min(req.max_rounds or 3, 8))   # 轮次预算上限防失控
    token = _resolve_token(session, base, req.auth)

    rounds: list[dict] = []
    attempts: list[dict] = []          # 轮次间记忆：已尝试未成功的修正
    prev_other_failed: int | None = None
    converged = False
    stopped_reason = ""

    for rd in range(1, max_rounds + 1):
        res_log: list[str] = []
        results, stats = _execute_all(session, base, token, source_dir=req.source_dir,
                                      log=res_log.append)
        failed = _failed_count(stats)
        # 判定口径：potential_bug 用例（已知缺陷复现，预期就是 FAIL）单独计数——
        # 它不参与收敛/无改进判定，否则 bug 用例的稳定失败会让循环永不收敛，
        # 还会激励 LLM 删除修不好的用例来"降失败数"（实测 bug 证据就是这么丢的）
        failed_bug = sum(1 for r in results if r.get("verdict") in ("FAIL", "ERROR", "SKIPPED")
                         and r.get("category") == "potential_bug")
        failed_other = failed - failed_bug
        rounds.append({"round": rd, "stats": stats, "failed": failed,
                       "failed_bug": failed_bug, "failed_other": failed_other,
                       "resource_log": res_log,
                       "total": sum(stats.values())})
        if failed == 0:
            converged = True
            stopped_reason = f"第 {rd} 轮全部通过，收敛"
            break
        if failed_other == 0:
            converged = True
            stopped_reason = (f"第 {rd} 轮可修复失败已清零；剩余 {failed_bug} 条为已知缺陷复现用例"
                              "（potential_bug——发现的是被测系统问题，应报告而非迭代修复）")
            break
        if prev_other_failed is not None and failed_other >= prev_other_failed:
            stopped_reason = (f"连续无改进（可修复失败数 {prev_other_failed} → {failed_other}），"
                              "停止迭代以避免死循环烧 token")
            break
        prev_other_failed = failed_other

        # 轮次间记忆补全：上轮修正的执行结果回填到记忆条目（"已证伪"的客观证据），
        # 修正后已通过的条目解除负记忆（不再占用 prompt）
        import json as _json
        still_failed = {(r.get("entry", ""), r.get("description", "")): r for r in results
                        if r.get("verdict") in ("FAIL", "ERROR", "SKIPPED")}
        for a in attempts:
            r = still_failed.get((a.get("entry", ""), a.get("description", "")))
            if r is None:
                a["_resolved"] = True     # 本轮没再失败 → 修正生效
            else:
                a["next_result"] = f"{r.get('verdict')}（实际 {r.get('status')}）"
        attempts = [a for a in attempts if not a.get("_resolved")]

        # 回灌（带前几轮诊断记忆）——修正用例直接提升为当前版本
        fb = _feedback_core(session, req.source_dir, req.provider,
                            previous_attempts=attempts or None)
        for entry, info in (fb.get("entries") or {}).items():
            # 记忆条目附"修正后用例快照"：回灌覆盖后上轮旧版本已不在下轮 prompt 里，
            # 不带快照的话 LLM 看不到"上轮改成了什么样"，无法判断该往哪换思路
            refined_by_desc = {c.get("description"): c for c in info.get("refined_cases") or []}
            for d in info.get("diagnosis") or []:
                c = refined_by_desc.get(d.get("description", "")) or {}
                req_snap = c.get("request") or {}
                body_snap = _json.dumps(req_snap.get("body"), ensure_ascii=False)[:200]
                attempts.append({"entry": entry, "description": d.get("description", ""),
                                 "category": d.get("category", ""),
                                 "cause": d.get("cause", ""), "action": d.get("action", ""),
                                 "fixed_expected": c.get("expected_status"),
                                 "fixed_request": {"method": req_snap.get("method"),
                                                   "url": req_snap.get("url"),
                                                   "body": body_snap}})
        rounds[-1]["feedback_summary"] = {
            "entries": {k: {"added_count": v.get("added_count", 0),
                            "removed_count": v.get("removed_count", 0),
                            "diagnosis": v.get("diagnosis", []),
                            "error": v.get("error", "")}
                        for k, v in (fb.get("entries") or {}).items()},
            "aggregate_hints": fb.get("aggregate_hints", []),
        }
    else:
        stopped_reason = f"达到最大轮次 {max_rounds}（最后一轮回灌的修正未再执行验证）"

    return {"converged": converged, "stopped_reason": stopped_reason,
            "rounds": rounds, "max_rounds": max_rounds,
            "final_results": session.results}


@app.post("/api/coverage")
def coverage(req: CoverageRequest):
    """盲区覆盖检测（纯函数同进程 + coverage.py）。

    输入：被测纯函数模块路径 + 被测函数限定名（entry）+ 每组调用参数 calls。
    服务端从会话取该 entry 的功能点档案（含 control_sites 行区间），
    用 coverage.py 采集实际语句/分支覆盖，与期望分支比对 → 盲区清单 + 覆盖率。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    if not req.cover_module:
        raise HTTPException(status_code=400, detail="请提供被测纯函数模块路径 cover_module")
    if not req.entry:
        raise HTTPException(status_code=400, detail="请提供被测函数限定名 entry")
    if not req.calls:
        raise HTTPException(status_code=400, detail="请提供至少一组调用参数 calls")

    # 取功能点档案（含 control_sites 行区间）作为期望覆盖模型
    from .eval.coverage import analyze_function, render_report
    from pathlib import Path as _P

    session = _get_or_create_session(req.source_dir)
    feature = next((f for f in session.features if f.get("entry") == req.entry), None)
    if feature is None:
        # 优先从被测模块直接构造档案（覆盖检测针对单个函数，用 analyze_file 最准）。
        # 注意：不能用 split_features 白名单重建索引，因为白名单 miss 时会退回自动识别，
        # 把项目里所有 HTTP 入口都拉进来，污染期望分支。
        from .core.ast_analysis import analyze_file as _analyze_file
        try:
            finfo = _analyze_file(_P(req.cover_module))
            func_name = req.entry.rsplit(".", 1)[-1]
            found = None
            for f in finfo.top_functions:
                if f.name == func_name:
                    found = f
                    break
            if found is None:
                for cls in finfo.classes:
                    for m in cls.methods:
                        if m.name == func_name:
                            found = m
                            break
            if found is None:
                raise HTTPException(status_code=404,
                                    detail=f"在 {req.cover_module} 中找不到函数 {func_name}")
            feature = {
                "entry": req.entry,
                "signature": f"def {found.name}({', '.join(found.args)})",
                "control_sites": [
                    {"kind": c.kind, "line": c.line, "cond": c.cond, "branch": c.branch,
                     "start": c.start, "end": c.end, "semantics": c.semantics, "path": c.path}
                    for c in found.controls
                ],
                "call_edges": [],
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"分析被测模块失败: {e}")

    args_list = [c.get("args") or [] for c in req.calls]
    kwargs_list = [c.get("kwargs") or {} for c in req.calls]

    try:
        result = analyze_function(feature, req.cover_module,
                                  args_list, kwargs_list, entry_qualname=req.entry)
        return {
            "coverage": result.to_dict(),
            "report_text": render_report(result),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"覆盖分析失败: {e}")


# ---------------- 缺陷登记面板（bug 注册表查询/重放/删除） ----------------
def _get_bug_store():
    from .storage.db import TestCaseStore
    return TestCaseStore(_DB_PATH)


@app.get("/api/bugs")
def list_bugs(source_dir: str):
    """列出缺陷登记（按 source_dir 过滤），前端按功能点分组展示。"""
    if not source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    store = _get_bug_store()
    try:
        findings = store.list_bug_findings(source_dir=source_dir)
    finally:
        store.close()
    return {"count": len(findings), "bugs": findings}


@app.delete("/api/bugs/{finding_id}")
def delete_bug(finding_id: int):
    """删除一条缺陷登记（人工修复缺陷/确认误报时解除）。"""
    store = _get_bug_store()
    try:
        ok = store.delete_bug_finding(finding_id)
    finally:
        store.close()
    if not ok:
        raise HTTPException(status_code=404, detail=f"缺陷登记不存在: {finding_id}")
    return {"deleted": finding_id}


class BugReplayRequest(BaseModel):
    """手动重放缺陷复现请求（验证缺陷是否仍存在）/ 应用修复分支。"""
    source_dir: str
    base_url: str = ""
    auth: dict = Field(default_factory=dict)
    service_cmd: str = ""   # 被测服务重启命令（应用修复分支后重启用），{project}/{port} 占位


# ---------------- AI 修复子 Agent（方案 A：pythonTest 内置，完全全自动） ----------------
_fix_tasks: dict[str, dict] = {}

# 默认重启命令模板：适配 full-stack-fastapi-template（uv + backend 子目录）
_DEFAULT_SERVICE_CMD = ("cd {project}/backend && nohup uv run uvicorn app.main:app "
                        "--host 0.0.0.0 --port {port} > /tmp/pyst-fix-svc.log 2>&1 &")


class FixStartRequest(BaseModel):
    """启动一次 AI 修复（后台执行，/api/fix/status 轮询）。"""
    source_dir: str
    finding_id: int
    base_url: str = ""
    auth: dict = Field(default_factory=dict)
    service_cmd: str = ""       # 被测服务重启命令，{project}/{port} 占位；空 = 默认模板
    provider: str = "deepseek"
    max_turns: int = 10
    confirm: bool = False       # 显式授权开关：AI 修复为高权限操作，必须确认


@app.post("/api/fix/start")
def fix_start(req: FixStartRequest):
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    if not req.confirm:
        raise HTTPException(status_code=400,
                            detail="AI 修复为高权限操作（会修改被测项目代码并重启服务），需显式确认")
    session = _get_or_create_session(req.source_dir)
    base = (req.base_url or session.base_url).strip()
    if not base:
        raise HTTPException(status_code=400, detail="请提供被测服务地址 base_url")
    store = _get_bug_store()
    try:
        finding = store.get_bug_finding(req.finding_id)
    finally:
        store.close()
    if not finding:
        raise HTTPException(status_code=404, detail=f"缺陷登记不存在: {req.finding_id}")
    if finding.get("source_dir") != req.source_dir:
        raise HTTPException(status_code=400, detail="缺陷登记与 source_dir 不匹配")
    if not finding.get("request"):
        raise HTTPException(status_code=400, detail="该登记缺少复现请求，无法修复")

    port = (urllib.parse.urlparse(
        base if "//" in base else "http://" + base).port) or 80
    service_cmd = (req.service_cmd or _DEFAULT_SERVICE_CMD).format(
        project=req.source_dir, port=port)
    token = _resolve_token(session, base, req.auth)

    task_id = uuid.uuid4().hex[:8]
    from .eval.fixer import FixAgent
    agent = FixAgent(task_id, session, req.source_dir, finding, base, token,
                     service_cmd, req.provider, max_turns=min(req.max_turns, 20),
                     log_fn=lambda m: _fix_tasks[task_id]["log"].append(m))
    _fix_tasks[task_id] = agent.task
    import threading
    threading.Thread(target=agent.run, daemon=True).start()
    return {"task_id": task_id, "status": "running"}


@app.get("/api/fix/status/{task_id}")
def fix_status(task_id: str):
    task = _fix_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"修复任务不存在: {task_id}")
    return task


@app.post("/api/bugs/{finding_id}/replay")
def replay_bug(finding_id: int, req: BugReplayRequest):
    """手动重复测试：真实发送注册表里的复现请求，对照登记时的实际行为判定。

    - 实际状态码与登记时一致 → 缺陷仍复现
    - 不一致 → 服务端行为已变化（可能已修复），建议解除登记
    自动重放**不会**解除登记（判定交给人工），也不会回填结果。
    """
    store = _get_bug_store()
    try:
        finding = store.get_bug_finding(finding_id)
    finally:
        store.close()
    if not finding:
        raise HTTPException(status_code=404, detail=f"缺陷登记不存在: {finding_id}")
    if not finding.get("request"):
        raise HTTPException(status_code=400, detail="该登记缺少复现请求，无法重放")
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    from .eval.fixer import _git, find_git_root
    session = _get_or_create_session(req.source_dir)
    base = (req.base_url or session.base_url).strip()
    if not base:
        raise HTTPException(status_code=400, detail="请提供被测服务地址 base_url")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    token = _resolve_token(session, base, req.auth)
    from .eval.executor import execute_http_case
    result = execute_http_case(
        {"_entry": finding["entry"], "description": finding["description"],
         "request": finding["request"], "expected_status": finding.get("expected_status")},
        base, token=token)
    orig_status = (finding.get("evidence") or {}).get("actual_status")
    now_status = result.get("status")
    still = None
    suggestion = ""
    fix_branch = f"ai-fix/bug-{finding_id}"
    project = find_git_root(req.source_dir)
    if orig_status and now_status:
        still = (now_status == orig_status)
        if still:
            suggestion = f"缺陷仍复现（实际 {now_status}，与登记时一致）——保留登记"
            # AI 修复分支存在但未应用时，说明重放的是"未应用修复"的主工作区代码——
            # 这是预期状态（修复待人工审阅 merge），提示用户操作路径
            project = find_git_root(req.source_dir)
            if project:
                branches = _git(project, "branch", "--list", fix_branch, check=False)
                if fix_branch in branches:
                    suggestion += (f"（注意：AI 修复分支 {fix_branch} 已存在但尚未应用——"
                                   "当前重放的是未应用修复的主工作区代码，可点「应用修复分支」后再验证）")
        else:
            suggestion = (f"服务端行为已变化（登记时 {orig_status} → 现在 {now_status}），"
                          "缺陷可能已被修复或环境不同——建议核对后解除登记")
    return {"finding_id": finding_id, "replay": result,
            "recorded_status": orig_status, "still_reproduces": still,
            "fix_branch": fix_branch if _git(project, "branch", "--list", fix_branch, check=False) else "",
            "suggestion": suggestion}


@app.post("/api/bugs/{finding_id}/apply")
def apply_bug_fix(finding_id: int, req: BugReplayRequest):
    """应用 AI 修复分支（merge ai-fix/bug-N 到当前分支）并重启被测服务。

    这是"AI 修复成功 → 人工确认"之间的操作：先在前端审阅 diff，再应用，
    应用后重放验证缺陷是否消失，确认无误后解除登记。merge 冲突时报 409 交人工处理。
    """
    if not req.source_dir:
        raise HTTPException(status_code=400, detail="请提供 source_dir")
    from .eval.fixer import _git, find_git_root, _restart_service
    store = _get_bug_store()
    try:
        finding = store.get_bug_finding(finding_id)
    finally:
        store.close()
    if not finding:
        raise HTTPException(status_code=404, detail=f"缺陷登记不存在: {finding_id}")
    project = find_git_root(req.source_dir)
    if not project:
        raise HTTPException(status_code=400, detail=f"{req.source_dir} 不在 git 仓库中")
    branch = f"ai-fix/bug-{finding_id}"
    branches = _git(project, "branch", "--list", branch, check=False)
    if branch not in branches:
        raise HTTPException(status_code=404,
                            detail=f"修复分支 {branch} 不存在（可能已被应用/删除，或 AI 修复未成功）")
    if _git(project, "status", "--porcelain").strip():
        raise HTTPException(status_code=400,
                            detail="被测项目工作区不干净，拒绝 merge（防止误伤未提交的更改）")
    try:
        merge_out = _git(project, "merge", "--no-ff", "-m",
                         f"merge: 应用 AI 修复（bug #{finding_id}）", branch, check=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"merge 执行失败: {e}")
    if "CONFLICT" in merge_out:
        _git(project, "merge", "--abort", check=False)
        raise HTTPException(status_code=409,
                            detail=f"merge 存在冲突（已中止，工作区未变），请人工处理分支 {branch}")
    # merge 成功：重启服务加载修复后代码，并清理已应用的修复分支
    session = _get_or_create_session(req.source_dir)
    base = (req.base_url or session.base_url).strip() or req.base_url
    token = _resolve_token(session, base, req.auth)
    port = urllib.parse.urlparse(base if "//" in base else "http://" + base).port or 80
    service_cmd = (req.service_cmd or _DEFAULT_SERVICE_CMD).format(project=req.source_dir, port=port)
    _restart_service(base, service_cmd, log=lambda m: None)
    _git(project, "branch", "-d", branch, check=False)
    return {"applied": True, "branch": branch,
            "merge": merge_out[:300],
            "next": "服务已重启（运行修复后代码）——请点「重放验证」确认缺陷行为已消失，再解除登记"}


# ---------------- PRD 需求文档分析 ----------------
_PRD_UPLOAD_DIR = Path(__file__).parent / "uploads"
_PRD_UPLOAD_DIR.mkdir(exist_ok=True)

_UPLOAD_EXTS = {".docx", ".md", ".txt", ".markdown", ".pdf"}


@app.post("/api/prd/upload")
async def prd_upload(file: UploadFile = File(...)):
    """上传 PRD 文档（.docx/.md/.txt/.pdf），预处理提取功能点列表（确定性，不调 LLM）。"""
    from .prd.analyzer import read_document, extract_prd_features

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {suffix}，请上传 .docx/.md/.txt/.pdf")

    # 保存上传文件
    save_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = _PRD_UPLOAD_DIR / save_name
    content = await file.read()
    save_path.write_bytes(content)

    try:
        # 读取文档内容（返回文本 + 格式）
        text, fmt = read_document(save_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空")
        # 预处理：提取功能点列表（不调 LLM）
        features = extract_prd_features(text)
        return {
            "filename": file.filename,
            "format": fmt,
            "feature_count": len(features),
            "features": [f.to_dict() for f in features],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PRD 预处理失败: {e}")
    finally:
        save_path.unlink(missing_ok=True)


class PrdGenerateRequest(BaseModel):
    features: list[dict] = Field(default_factory=list)   # 选中的功能点列表
    provider: str = "deepseek"
    case_count: int = 5


@app.post("/api/prd/generate")
async def prd_generate(req: PrdGenerateRequest):
    """为选中的 PRD 功能点生成测试用例。"""
    from .prd.analyzer import generate_prd_cases

    if not req.features:
        raise HTTPException(status_code=400, detail="请至少选择一个功能点")

    result: dict[str, list[dict]] = {}
    for feature in req.features:
        title = feature.get("title", "未命名功能点")
        try:
            cases = generate_prd_cases(feature, provider=req.provider, case_count=req.case_count)
            result[title] = cases
        except Exception as e:
            result[title] = [{"description": f"[生成失败] {e}", "test_steps": [], "expected_results": []}]
    return {"feature_count": len(result), "test_cases": result}


# ---------------- 用例落库查询 ----------------
@app.get("/api/cases")
def list_stored_cases(entry: str | None = None, module: str | None = None):
    """查询落库的测试用例。可按 entry 或 module 过滤。

    先清理历史遗留的完全重复记录（deduplicate），再返回去重后的用例。
    """
    from .storage.db import TestCaseStore
    store = TestCaseStore(_DB_PATH)
    try:
        removed = store.deduplicate()   # 清理历史重复
        if entry:
            cases = store.list_by_entry(entry)
        elif module:
            cases = store.list_by_module(module)
        else:
            cases = store.all()
        return {"count": len(cases), "cases": [c.to_dict() for c in cases],
                "entries": store.entries(), "deduplicated": removed}
    finally:
        store.close()


@app.delete("/api/cases/{case_id}")
def delete_stored_case(case_id: int):
    """删除一条落库的用例。"""
    from .storage.db import TestCaseStore
    store = TestCaseStore(_DB_PATH)
    try:
        ok = store.delete(case_id)
    finally:
        store.close()
    if not ok:
        raise HTTPException(status_code=404, detail=f"用例不存在: {case_id}")
    return {"deleted": case_id}


# ---------------- 启动 ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
