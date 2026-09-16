#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1：项目级符号索引（原 project_index.py）
==============================================
遍历目录下所有 .py 文件，产出全局可查询的符号索引 —— 这是调用图（Phase 2）的地基。

【本阶段回答】"项目里有哪些符号？每个符号定义在哪一行？每个文件 import 了什么？
             谁依赖了哪些本地模块？"
【本阶段不回答】"符号级别的调用关系"（A.py 的 f() 到底调用了 B.py 的哪个 g()）
              —— 那是 Phase 2 phase2_callgraph.py 的职责，本模块是它的输入。

产出结构（可 JSON 导出）：
  - module_paths : 模块名 -> 文件路径          （import 解析的查表基础）
  - symbols      : 全局符号表 限定名 -> SymbolInfo（类/方法/函数 + 行号区间）
  - files        : 每个文件的 imports / calls / local_deps（文件级依赖：谁 import 谁）

用法:
    python -m pyst <项目根目录> --phase 1                # 打印摘要
    python -m pyst <项目根目录> --phase 1 --json index.json
    python -m pyst <项目根目录> --phase 1 --query get_logger
    python phase1_index.py <项目根目录> --query get_logger   # 模块内直接运行

例:
    python -m pyst ../TestBrain --phase 1 --query get_logger
"""

import ast
import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .ast_analysis import analyze_file, FileInfo, ClassInfo, FunctionInfo, CallSite


# ---------- 数据结构 ----------

@dataclass
class ImportedName:
    name: str                       # 导入的名字，如 get_logger / os
    asname: str | None = None       # 别名


@dataclass
class ImportInfo:
    line: int                       # 语句所在行
    kind: str                       # "import" | "from_import"
    module: str                     # 导入来源：import a.b.c -> "a.b.c"；from x import y -> "x"；相对导入带 . 前缀
    level: int                      # 相对导入层级（0 = 绝对导入，1 = 当前包，2 = 父包...）
    names: list[ImportedName] = field(default_factory=list)


@dataclass
class SymbolInfo:
    qualname: str                   # 全局唯一限定名：apps.llm.callbacks.LoggingCallbackHandler.on_llm_start
    name: str                       # 短名：on_llm_start
    kind: str                       # class | method | function
    module: str                     # 所属模块：apps.llm.callbacks
    parent: str | None              # 所属类限定名（模块级函数/类为 None）
    start: int
    end: int
    bases: list[str] = field(default_factory=list)   # 类符号：基类原始表达式（继承链解析用）


@dataclass
class CallInfo:
    module: str                     # 调用点所在模块
    owner: str                      # 调用点所在函数（限定名）
    line: int
    qualname: str                   # 调用表达式原文，如 self._get_run_id
    name: str                       # 最右名字，如 _get_run_id


@dataclass
class FileIndex:
    path: str
    module_name: str
    imports: list[ImportInfo] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    local_deps: list[str] = field(default_factory=list)   # 依赖的本地模块名（Phase 1.5 产物）
    is_test: bool = False   # 是否为测试模块（供功能点入口识别排除测试函数）


# ---------- 模块名推导：文件路径 -> Python 模块名 ----------

def derive_module_name(file_path: Path, root: Path) -> str:
    """把文件路径换算成模块名。

    TestBrain/apps/llm/callbacks.py       -> "apps.llm.callbacks"
    TestBrain/apps/llm/__init__.py        -> "apps.llm"     （__init__.py 代表包本身）
    """
    rel = file_path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        return ".".join(parts) if parts else root.name
    return ".".join(parts[:-1] + [parts[-1][:-3]])   # 去掉 ".py" 后缀


def resolve_relative_module(level: int, module: str, current_module: str) -> str | None:
    """把相对导入解析成绝对模块名。level=0 是绝对导入，直接返回 module。

    from . import x     (level=1, module="")   当前模块 apps.llm.callbacks -> "apps.llm"
    .utils x (level=1, module="utils") -> "apps.llm.utils"
    from ..utils import x (level=2, module="utils") -> "apps.utils"
    from x import y      (level=0, module="x")  -> "x"
    """
    if level == 0:
        return module
    parts = current_module.split(".")
    if level > len(parts):
        return None
    base = parts[: len(parts) - level]
    if module:
        base = base + [module]
    return ".".join(base)


# ---------- 结构化 import 收集（FileInfo.imports 只是展示字符串，这里重新收集） ----------

class ImportVisitor(ast.NodeVisitor):
    """只收集 import 语句的结构化信息（来源模块、层级、名字、别名）"""

    def __init__(self):
        self.imports: list[ImportInfo] = []

    def visit_Import(self, node):
        # import os            -> module="os"
        # import a.b.c         -> module="a.b.c"，绑定名是 a
        # import os, sys       -> 两条记录
        for alias in node.names:
            bound = alias.name.split(".")[0]
            self.imports.append(ImportInfo(
                line=node.lineno,
                kind="import",
                module=alias.name,
                level=0,
                names=[ImportedName(bound, alias.asname)],
            ))

    def visit_ImportFrom(self, node):
        # from x import y        -> module="x", level=0
        # from . import y        -> module=".", level=1
        # from ..x import y      -> module="..x", level=2
        names = [ImportedName(a.name, a.asname) for a in node.names]
        module = "." * node.level + (node.module or "")
        self.imports.append(ImportInfo(
            line=node.lineno,
            kind="from_import",
            module=module,
            level=node.level,
            names=names,
        ))


# ---------- 全局索引 ----------

# 遍历时排除的目录（虚拟环境/第三方依赖/版本控制等，避免把 venv 里的库当被测代码）
_EXCLUDE_DIRS = {
    ".venv", "venv", "env", ".env",
    "node_modules", "site-packages", "dist-packages",
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__",
    "build", "dist", ".eggs", "*.egg-info",
}


def _is_excluded_dir(path: Path) -> bool:
    """判断路径的某部分是否是需要排除的目录。"""
    return any(part in _EXCLUDE_DIRS or part.endswith(".egg-info") for part in path.parts)


def iter_python_files(root: Path):
    """遍历项目下所有 .py 文件，排除虚拟环境/第三方/版本控制目录。

    用 os.walk + 剪枝（不进入被排除的目录），比 rglob 高效，避免遍历 venv 里
    成千上万的第三方文件（之前 in-re-agent 有 1.2 万个 .py 全是 .venv 的）。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # 剪枝：剔除需要排除的子目录（原地修改 dirnames 使 os.walk 不进入）
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDE_DIRS and not d.endswith(".egg-info") and not d.startswith(".")
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


# ---------- 测试模块识别（用于排除测试函数被误识别为功能点入口） ----------
# 【设计原则】以"结构性信号"为主判据（文件位置 + 模块级 import），
# 函数名前缀 test_ 只作兜底（在 features 层），避免误伤名字里碰巧带 test 的业务函数。
_TEST_DIR_NAMES = {"tests", "test", "testing", "unit_tests", "integration_tests"}
_TEST_IMPORT_ROOTS = {"pytest", "unittest", "nose", "nose2", "hypothesis", "ddt", "mock"}
_TEST_IMPORT_PREFIXES = (
    "fastapi.testclient",      # FastAPI TestClient
    "starlette.testclient",
    "django.test",             # Django TestCase / Client
    "flask.testing",
    "sqlalchemy.testing",
    "werkzeug.test",
)
_TEST_PACKAGE_ROOTS = {"tests", "test", "testing"}


def _is_test_path(path: str) -> bool:
    """文件位置信号：所在目录名为测试目录，或文件名符合测试命名约定。"""
    p = Path(path)
    for part in p.parts[:-1]:
        if part.lower() in _TEST_DIR_NAMES:
            return True
    name = p.name.lower()
    if name == "conftest.py":
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return False


def _imports_test_framework(imports: list[ImportInfo]) -> bool:
    """模块级 import 信号：引入了测试框架 / 测试包。

    相对导入（level>0）不参与判断（无法可靠还原绝对模块名，且测试内部相对导入
    对"这个文件是不是测试"没有增量信息）。
    """
    for imp in imports:
        if imp.level and imp.level > 0:
            continue
        mod = (imp.module or "").strip()
        if not mod:
            continue
        root = mod.split(".")[0]
        if root in _TEST_IMPORT_ROOTS:                     # import pytest / unittest
            return True
        if mod in _TEST_IMPORT_PREFIXES or \
                any(mod.startswith(pfx + ".") for pfx in _TEST_IMPORT_PREFIXES):
            return True                                    # from fastapi.testclient import TestClient
        if root in _TEST_PACKAGE_ROOTS:                    # from tests.utils import xxx
            return True
        for n in (imp.names or []):
            if getattr(n, "name", "") in ("TestCase", "TestClient", "APITestCase"):
                return True                                # from django.test import TestCase
    return False


def is_test_module(path: str, imports: list[ImportInfo]) -> bool:
    """判断一个模块是否为测试模块：文件位置信号 ∨ 模块级 import 信号。

    注意与"函数名是否以 test_ 开头"解耦——本函数完全不看函数名，因此：
      - 业务模块里名为 test_connection() 的函数 → 不会被误判
      - tests/ 里名为 check_create_item() 的测试函数 → 仍能被正确识别
    """
    if _is_test_path(path):
        return True
    if imports and _imports_test_framework(imports):
        return True
    return False


class ProjectIndex:
    def __init__(self):
        self.files: dict[str, FileIndex] = {}      # module_name -> FileIndex
        self.symbols: dict[str, SymbolInfo] = {}   # qualname -> SymbolInfo（全局符号表）
        self.module_paths: dict[str, str] = {}     # module_name -> 文件路径
        self.function_infos: dict[str, FunctionInfo] = {}  # qualname -> FunctionInfo（含签名/控制流/装饰器，M3 用）

    def build(self, root: Path) -> "ProjectIndex":
        """两遍扫描：第一遍收集全部事实，第二遍解析本地依赖（需要完整索引）。"""
        for py in sorted(iter_python_files(root)):
            self._index_file(py, root)
        for fi in self.files.values():
            fi.local_deps = self._resolve_local_deps(fi)
        return self

    def _index_file(self, path: Path, root: Path):
        try:
            info: FileInfo = analyze_file(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, RecursionError):
            return  # 跳过解析失败或嵌套过深的文件
        except MemoryError:
            return  # 跳过超大文件

        module = derive_module_name(path, root)
        self.module_paths[module] = str(path)

        fi = FileIndex(path=str(path), module_name=module)
        visitor = ImportVisitor()
        visitor.visit(tree)
        fi.imports = visitor.imports
        fi.is_test = is_test_module(str(path), visitor.imports)   # 供入口识别排除测试函数
        fi.symbols = self._collect_symbols(info, module)
        fi.calls = self._collect_calls(info, module)

        self.files[module] = fi
        for s in fi.symbols:
            self.symbols[s.qualname] = s

    # --- 符号收集：FileInfo -> 全局限定名符号表 ---
    def _collect_symbols(self, info: FileInfo, module: str) -> list[SymbolInfo]:
        out = []
        for cls in info.classes:                       # ClassInfo
            q = f"{module}.{cls.name}"
            out.append(SymbolInfo(q, cls.name, "class", module, None, cls.start, cls.end, bases=cls.bases))
            for m in cls.methods:                      # FunctionInfo
                mq = f"{q}.{m.name}"
                out.append(SymbolInfo(mq, m.name, "method", module, q, m.start, m.end))
                self.function_infos[mq] = m            # M3：登记函数体信息（控制流/签名/装饰器）
        for f in info.top_functions:
            q = f"{module}.{f.name}"
            out.append(SymbolInfo(q, f.name, "function", module, None, f.start, f.end))
            self.function_infos[q] = f                # M3：登记模块级函数体信息
        return out

    # --- 调用点收集：方法内的 CallSite -> 带 owner 的 CallInfo ---
    def _collect_calls(self, info: FileInfo, module: str) -> list[CallInfo]:
        out = []
        for cls in info.classes:
            for m in cls.methods:
                owner = f"{module}.{cls.name}.{m.name}"
                for c in m.calls:                      # CallSite(line, qualname, name)
                    out.append(CallInfo(module, owner, c.line, c.qualname, c.name))
        for f in info.top_functions:
            owner = f"{module}.{f.name}"
            for c in f.calls:
                out.append(CallInfo(module, owner, c.line, c.qualname, c.name))
        return out

    # --- 文件级依赖解析：import 语句 -> 本地模块名（Phase 1.5） ---
    def _resolve_local_deps(self, fi: FileIndex) -> list[str]:
        deps = set()
        for imp in fi.imports:
            if imp.kind == "import":
                # import a.b.c：可能是 a.b.c、a.b、a 中的任何一个本地模块
                parts = imp.module.split(".")
                candidates = [".".join(parts[: i + 1]) for i in range(len(parts))]
            else:
                if imp.level > 0:
                    m = resolve_relative_module(imp.level, imp.module, fi.module_name)
                    candidates = [m] if m else []
                else:
                    candidates = [imp.module]
            for c in candidates:
                if c in self.module_paths:             # 只算本地模块，标准库/第三方自动排除
                    deps.add(c)
        return sorted(deps)

    # --- 查询接口（Phase 2 会用） ---
    def find_symbol(self, qualname: str) -> SymbolInfo | None:
        return self.symbols.get(qualname)

    def symbols_by_name(self, name: str) -> list[SymbolInfo]:
        """同名符号查询：项目里有多个 get_logger 时，返回全部候选"""
        return [s for s in self.symbols.values() if s.name == name]

    def to_dict(self) -> dict[str, object]:
        return {
            "module_paths": self.module_paths,
            "symbols": {k: asdict(v) for k, v in self.symbols.items()},
            "files": {k: asdict(v) for k, v in self.files.items()},
        }


# ---------- 渲染 ----------

def render_summary(index: ProjectIndex, limit: int = 50):
    print(f"项目索引: {len(index.files)} 个文件, {len(index.symbols)} 个符号\n")
    for i, (module, fi) in enumerate(sorted(index.files.items())):
        if i >= limit:
            print(f"... 其余 {len(index.files) - limit} 个文件省略，可用 --json 导出完整索引")
            break
        deps = ", ".join(fi.local_deps) if fi.local_deps else "无"
        print(f"  {module}")
        print(f"      [{fi.path}]")
        print(f"      依赖本地模块: {deps}")


def render_query(index: ProjectIndex, name: str):
    hits = index.symbols_by_name(name)
    print(f"\n同名符号查询: {name!r} 共 {len(hits)} 个定义")
    for s in hits:
        print(f"  {s.qualname}  [{s.kind}] 行 {s.start}-{s.end}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="项目级符号索引（Phase 1）")
    ap.add_argument("root", help="项目源码根目录")
    ap.add_argument("--json", metavar="FILE", help="导出完整索引为 JSON 文件")
    ap.add_argument("--query", metavar="NAME", help="查询同名符号的定义位置")
    ap.add_argument("--limit", type=int, default=50, help="摘要显示的文件数上限（默认 50）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        sys.exit(1)

    index = ProjectIndex().build(root)
    render_summary(index, args.limit)

    if args.query:
        render_query(index, args.query)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n索引已导出: {args.json}")


if __name__ == "__main__":
    main()
