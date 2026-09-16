#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyst 统一命令行入口
====================
一个命令跑通 Phase 0 → 1 → 2，后续 Phase 3/4 也在此挂载。

用法:
    python -m pyst <目标> --phase 0              # 文件/目录结构分析
    python -m pyst <项目根目录> --phase 1         # 项目符号索引
    python -m pyst <项目根目录> --phase 2         # 跨文件调用图（默认）

通用参数:
    --json FILE      导出 JSON（Phase 1: 索引；Phase 2: 调用图）
    --query NAME     Phase 1: 同名符号查询；Phase 2: 反向调用查询
    --unresolved     Phase 2: 只看未解/歧义调用
    --external       Phase 2: 同时显示外部调用
    --limit N        Phase 0: 无；Phase 1: 摘要文件数；Phase 2: 每函数边数

例:
    python -m pyst ../TestBrain/apps/llm/callbacks.py --phase 0
    python -m pyst ../TestBrain --phase 1 --query get_logger
    python -m pyst ../TestBrain --phase 2 --unresolved
"""

import argparse
import json
import sys
from pathlib import Path


def _dump(obj: dict, path: str, label: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"\n{label}已导出: {path}")


def main():
    ap = argparse.ArgumentParser(prog="python -m pyst", description="Python AST 静态分析管线（pythonTest）")
    ap.add_argument("root", help="目标 .py 文件（Phase 0）或项目根目录（Phase 1/2）")
    ap.add_argument("--phase", type=int, default=2, choices=[0, 1, 2, 3],
                    help="运行阶段：0 文件结构 / 1 项目索引 / 2 调用图 / 3 功能点拆分（默认 2）")
    ap.add_argument("--json", metavar="FILE", help="导出 JSON")
    ap.add_argument("--query", metavar="NAME", help="同名符号 / 反向调用查询")
    ap.add_argument("--unresolved", action="store_true", help="（Phase 2）只看未解/歧义调用")
    ap.add_argument("--external", action="store_true", help="（Phase 2）同时显示外部调用")
    ap.add_argument("--limit", type=int, default=0, help="显示上限（0=不限）")
    ap.add_argument("--depth", type=int, default=2, help="（Phase 3）本地调用子图展开深度（默认 2）")
    ap.add_argument("--entry", metavar="QUALNAME", action="append",
                    help="（Phase 3）显式指定功能点入口（可多次）")
    args = ap.parse_args()

    target = Path(args.root)
    if not target.exists():
        print(f"路径不存在: {target}")
        sys.exit(1)

    # ---------- Phase 0：文件级结构 ----------
    if args.phase == 0:
        from .core.ast_analysis import analyze_file, analyze_dir, render_file, render_summary
        if target.is_file():
            render_file(analyze_file(target))
        else:
            infos = analyze_dir(target)
            for info in infos:
                render_file(info)
            render_summary(infos)
        return

    if not target.is_dir():
        print("Phase 1/2 需要项目根目录（目录）；单文件请用 --phase 0。")
        sys.exit(1)

    # ---------- Phase 1：项目索引 ----------
    if args.phase == 1:
        from .core.project_index import ProjectIndex, render_summary, render_query
        index = ProjectIndex().build(target)
        render_summary(index, args.limit or 50)
        if args.query:
            render_query(index, args.query)
        if args.json:
            _dump(index.to_dict(), args.json, "索引")
        return

    # ---------- Phase 3：功能点拆分 ----------
    if args.phase == 3:
        from .core.project_index import ProjectIndex
        from .core.callgraph import CallGraph
        from .core.features import split_features, render_features

        index = ProjectIndex().build(target)
        graph = CallGraph(index).build()
        features = split_features(index, graph, max_depth=args.depth, entries=args.entry or [])
        render_features(features)
        if args.json:
            _dump({"features": [fp.to_dict() for fp in features]}, args.json, "功能点档案")
        return

    # ---------- Phase 2：调用图 ----------
    from .core.project_index import ProjectIndex
    from .core.callgraph import CallGraph, render_graph, render_query as render_cg_query

    index = ProjectIndex().build(target)
    graph = CallGraph(index).build()
    render_graph(graph, args.limit, args.unresolved, args.external)
    if args.query:
        render_cg_query(graph, args.query)
    if args.json:
        _dump(graph.to_dict(), args.json, "调用图")


if __name__ == "__main__":
    main()
