# -*- coding: utf-8 -*-
"""
pyst：Python AST 静态分析管线（pythonTest 主包）
=================================================
从源码到可执行测试用例的完整管线：
    Phase 0  phase0_file     文件级 AST 结构分析
    Phase 1  phase1_index    项目级符号/依赖索引
    Phase 2  phase2_callgraph 跨文件调用图
    Phase 3  phase3_features 功能点拆分
    Phase 4  phase4_generate 基于功能点档案生成测试用例
    Phase 5  phase5_interface 接口定义提取（method/url/参数）
    review    AI 测试用例评审
    storage   用例落库（SQLite）

统一入口：python -m pyst <root> --phase 0|1|2|3
可视化平台：uvicorn pyst.webapp:app（见 开发计划.md）
"""

from .core.ast_analysis import analyze_file, analyze_dir, FileInfo, ClassInfo, FunctionInfo, CallSite, ControlSite
from .core.project_index import ProjectIndex, SymbolInfo, CallInfo, resolve_relative_module
from .core.callgraph import CallGraph, CallEdge, CallResolver
from .core.features import FeaturePoint, split_features
from .core.generate import TestCase, build_prompt, generate_test_cases, build_refine_prompt, refine_test_cases
from .eval.review import review_test_cases, build_review_prompt
from .storage.db import TestCaseStore, StoredCase
from .prd.analyzer import analyze_prd, read_document

__all__ = [
    "analyze_file", "analyze_dir", "FileInfo", "ClassInfo", "FunctionInfo", "CallSite", "ControlSite",
    "ProjectIndex", "SymbolInfo", "CallInfo", "resolve_relative_module",
    "CallGraph", "CallEdge", "CallResolver",
    "FeaturePoint", "split_features",
    "TestCase", "build_prompt", "generate_test_cases",
    "build_refine_prompt", "refine_test_cases",
    "review_test_cases", "build_review_prompt",
    "TestCaseStore", "StoredCase",
    "analyze_prd", "read_document",
]

__version__ = "0.7.0"
