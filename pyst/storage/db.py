#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例落库（SQLite 持久化）
================================
让生成的测试用例可持久化、可复用、可追溯。

【设计】轻量 SQLite，不引入重型 ORM。单表存储用例，按功能点 + 入口维度组织。

【表结构】test_cases
    id           INTEGER PK        自增主键
    entry        TEXT               功能点入口（如 apps.xxx.views.download_file）
    module       TEXT               所在模块
    description  TEXT               用例描述
    request      TEXT(JSON)         可执行请求参数
    expected_status  INTEGER        预期状态码
    expected_results TEXT(JSON)     预期结果
    test_steps   TEXT(JSON)         步骤
    review_score INTEGER            AI 评审评分
    review_data  TEXT(JSON)         评审详情
    created_at   TEXT               创建时间
    source_dir   TEXT               来源代码目录

【用法】
    from pyst.storage import TestCaseStore
    store = TestCaseStore("cases.db")
    store.save_test_cases(entry, module, cases)      # 批量保存
    store.list_by_entry(entry)                       # 按入口查询
    store.all()                                      # 全部
    store.attach_review(entry, review)               # 挂上评审
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class StoredCase:
    id: int
    entry: str
    module: str
    description: str
    request: dict[str, Any] | None
    expected_status: int | None
    expected_results: list[Any]
    test_steps: list[Any]
    review_score: int | None = None
    review_data: dict[str, Any] | None = None
    created_at: str = ""
    source_dir: str = ""
    fingerprint: str = ""   # 生成时的代码指纹，用于缓存失效判断

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entry": self.entry,
            "module": self.module,
            "description": self.description,
            "request": self.request,
            "expected_status": self.expected_status,
            "expected_results": self.expected_results,
            "test_steps": self.test_steps,
            "review_score": self.review_score,
            "review_data": self.review_data,
            "created_at": self.created_at,
            "source_dir": self.source_dir,
            "fingerprint": self.fingerprint,
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class TestCaseStore:
    def __init__(self, db_path: str | Path = "test_cases.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry TEXT NOT NULL,
                module TEXT,
                description TEXT,
                request TEXT,
                expected_status INTEGER,
                expected_results TEXT,
                test_steps TEXT,
                review_score INTEGER,
                review_data TEXT,
                created_at TEXT,
                source_dir TEXT,
                fingerprint TEXT
            )
        """)
        # bug 发现注册表：potential_bug 用例的持久化档案——用例可被 LLM 重写/删除，
        # 但"系统有这个缺陷"的发现与复现方式必须独立存续（跨迭代会话）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bug_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry TEXT NOT NULL,
                source_dir TEXT,
                description TEXT,
                request TEXT,
                expected_status INTEGER,
                evidence TEXT,
                created_at TEXT
            )
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self):
        """老库迁移：若缺 fingerprint 列则补上。"""
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(test_cases)").fetchall()]
        if "fingerprint" not in cols:
            self._conn.execute("ALTER TABLE test_cases ADD COLUMN fingerprint TEXT")
            self._conn.commit()

    @staticmethod
    def _normalize_value(v: Any) -> Any:
        """递归归一化值，消除"语义相同但字面不同"的差异。

        规则：
          - None / "" / 空串 / [] / {} → 统一标记 "<empty>"（空值语义归一）
          - dict：递归归一化每个值，键保持（键名本身是接口参数，不归一）
          - list：递归归一化每个元素
          - 数字 / 布尔 / 非空字符串：保持原样
        目的：让"id 为空(None)"和"id 为空值(\"\")"归一化后相同，从而正确判重；
        同时保留正常值/边界值/非法值的差异（数字、非空串不归一）。
        """
        if v is None:
            return "<empty>"
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            return "<empty>" if v.strip() == "" else v
        if isinstance(v, dict):
            return {k: TestCaseStore._normalize_value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [TestCaseStore._normalize_value(x) for x in v]
        return v

    def _case_signature(self, c: dict[str, Any]) -> str:
        """计算一条用例的去重签名。

        判重依据（只考虑"确定了测试功能"的硬信息，去掉 LLM 话术）：
          - api 形态：request（值归一化 + 键排序）
          - func 形态：call_args / call_kwargs（纯函数无 request）
        不再考虑 description / expected_results / expected_status 之外的软描述字段，
        因为同一测试意图的措辞可能不同（如"id 为空" vs "id 为空值"），不应造成假差异。

        注意：expected_status 已体现在 request 之外；但测试功能主要由"请求内容"确定，
        故指纹 = entry + source_dir + request(归一化) [+ call_args]。
        """
        def _norm(v):
            return json.dumps(TestCaseStore._normalize_value(v),
                              sort_keys=True, ensure_ascii=False) if v is not None else None
        # api 用例用 request；func 用例用 call_args/call_kwargs
        if c.get("request") is not None:
            req = c["request"]
            if isinstance(req, dict):
                # 剔除"空值键"后再算指纹：LLM 有时输出 "query": {}，有时干脆不输出该键，
                # 两者语义相同（实测重复漏网的边角）。body 除外——"请求体为空对象"本身
                # 是"缺必填字段"用例的测试数据，必须保留差异
                req = {k: v for k, v in req.items()
                       if k not in ("headers", "query", "files")
                       or (isinstance(v, dict) and v) or not isinstance(v, dict)}
            payload = {"req": _norm(req)}
        else:
            payload = {"call_args": _norm(c.get("call_args")),
                       "call_kwargs": _norm(c.get("call_kwargs"))}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _case_signatures_by_entry(self, entry: str, source_dir: str = "") -> set[str]:
        """读取某 entry + source_dir 下已存在的用例签名集合（用于写入去重）。"""
        existing = self.get_by_entry_source(entry, source_dir)
        return {self._case_signature(c.to_dict()) for c in existing}

    def save_test_cases(self, entry: str, module: str, cases: list[dict[str, Any]],
                        source_dir: str = "", fingerprint: str = "") -> list[int]:
        """批量保存一组用例，返回插入的 id 列表。

        写入去重（两层，均为确定性规则，不做语义判断）：
          1. 签名去重：同 entry + source_dir 下 request/call_args（归一化）相同的用例跳过
          2. 描述去重：同 entry + source_dir 下 description 完全相同的用例跳过——
             实测回灌/重新生成会产出"描述一字不差、仅 body 值微差"的用例（语义重复，
             如"未认证访问"两条仅 title 字面不同），描述字符串完全一致视为同一条
        """
        ids = []
        seen = self._case_signatures_by_entry(entry, source_dir)
        seen_desc = {c.description.strip() for c in self.get_by_entry_source(entry, source_dir)
                     if c.description}
        for c in cases:
            sig = self._case_signature(c)
            desc = str(c.get("description", "")).strip()
            if sig in seen:
                continue            # 已存在完全相同 request，跳过
            if desc and desc in seen_desc:
                continue            # 已存在完全相同 description，跳过（语义重复）
            seen.add(sig)
            if desc:
                seen_desc.add(desc)
            cur = self._conn.execute(
                """INSERT INTO test_cases
                   (entry, module, description, request, expected_status,
                    expected_results, test_steps, created_at, source_dir, fingerprint)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (entry, module, c.get("description", ""),
                 json.dumps(c.get("request"), ensure_ascii=False) if c.get("request") else None,
                 c.get("expected_status"),
                 json.dumps(c.get("expected_results", []), ensure_ascii=False),
                 json.dumps(c.get("test_steps", []), ensure_ascii=False),
                 _now(), source_dir, fingerprint))
            ids.append(cur.lastrowid)
        self._conn.commit()
        return ids

    # ---------------- bug 发现注册表（potential_bug 持久化档案） ----------------

    def upsert_bug_finding(self, entry: str, source_dir: str, description: str,
                           request: Any, expected_status: Any, evidence: Any) -> int:
        """登记/更新一条 bug 发现（同 entry+source_dir+description 幂等）。"""
        row = self._conn.execute(
            "SELECT id FROM bug_findings WHERE entry=? AND IFNULL(source_dir,'')=? AND description=?",
            (entry, source_dir or "", description)).fetchone()
        if row:
            return row["id"]
        cur = self._conn.execute(
            """INSERT INTO bug_findings
               (entry, source_dir, description, request, expected_status, evidence, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (entry, source_dir or "", description,
             json.dumps(request, ensure_ascii=False) if request is not None else None,
             expected_status,
             json.dumps(evidence, ensure_ascii=False) if evidence is not None else None,
             _now()))
        self._conn.commit()
        return cur.lastrowid

    def list_bug_findings(self, entry: str | None = None,
                          source_dir: str = "") -> list[dict[str, Any]]:
        """查询 bug 发现注册表（可按 entry / source_dir 过滤）。"""
        q = "SELECT * FROM bug_findings"
        cond, args = [], []
        if entry:
            cond.append("entry=?")
            args.append(entry)
        if source_dir:
            cond.append("IFNULL(source_dir,'')=?")
            args.append(source_dir or "")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id"
        out: list[dict[str, Any]] = []
        for r in self._conn.execute(q, args).fetchall():
            out.append({
                "id": r["id"], "entry": r["entry"], "source_dir": r["source_dir"] or "",
                "description": r["description"],
                "request": json.loads(r["request"]) if r["request"] else None,
                "expected_status": r["expected_status"],
                "evidence": json.loads(r["evidence"]) if r["evidence"] else None,
                "created_at": r["created_at"],
            })
        return out

    def get_bug_finding(self, finding_id: int) -> dict[str, Any] | None:
        """按 id 查询单条 bug 发现。"""
        row = self._conn.execute("SELECT * FROM bug_findings WHERE id=?", (finding_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"], "entry": row["entry"], "source_dir": row["source_dir"] or "",
            "description": row["description"],
            "request": json.loads(row["request"]) if row["request"] else None,
            "expected_status": row["expected_status"],
            "evidence": json.loads(row["evidence"]) if row["evidence"] else None,
            "created_at": row["created_at"],
        }

    def delete_bug_finding(self, finding_id: int) -> bool:
        """删除一条 bug 发现（误报/服务端已修复时手动清除）。"""
        cur = self._conn.execute("DELETE FROM bug_findings WHERE id=?", (finding_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def delete_by_entry_source(self, entry: str, source_dir: str = "") -> int:
        """删除某 entry + source_dir 下的全部用例（重新生成/回灌覆盖旧版本用）。"""
        if source_dir:
            cur = self._conn.execute(
                "DELETE FROM test_cases WHERE entry=? AND source_dir=?", (entry, source_dir))
        else:
            cur = self._conn.execute("DELETE FROM test_cases WHERE entry=?", (entry,))
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _is_related_root(dir_a: str, dir_b: str) -> bool:
        """判断两个源码目录是否为同一项目的不同 root（祖先/后代目录关系）。"""
        if not dir_a or not dir_b:
            return False
        a, b = str(dir_a).rstrip("/"), str(dir_b).rstrip("/")
        return a == b or a.startswith(b + "/") or b.startswith(a + "/")

    @staticmethod
    def _entry_suffix(a: str, b: str) -> bool:
        """a 是否与 b 相同或 a 以 b 的完整点分段为后缀（段级比较，避免子串误判）。"""
        sa, sb = a.split("."), b.split(".")
        return len(sa) >= len(sb) and sa[-len(sb):] == sb

    def deduplicate(self) -> int:
        """清理库中的重复/过期记录，返回删除条数。四条确定性规则：

        1. 批次覆盖：同 entry + source_dir 只保留 created_at 最新的批次——
           重新生成/回灌语义上是"替换当前版本"（前端也这么展示），但历史实现
           落库时不删旧批次，导致库中累积多版本全部显示（用户看到的重复主因）
        2. 签名去重：同 entry + source_dir 内 request/call_args（归一化）相同，保留最新
        3. 描述去重：同 entry + source_dir 内 description 完全相同，保留最新
        4. root 变化归并：换 root 重新分析后同一功能点的 entry 多/少项目名前缀、
           source_dir 变化（如 backend 子目录 vs 项目上级），entry+source_dir 判重失效；
           entry 互为段级后缀 + source_dir 互为祖先/后代 → 视为同一功能点，只留最新 root
        """
        removed = 0
        rows = self._conn.execute(
            "SELECT * FROM test_cases ORDER BY id").fetchall()
        # 规则 1：批次覆盖
        latest: dict[tuple[str, str], str] = {}
        for r in rows:
            key = (r["entry"], r["source_dir"] or "")
            ca = r["created_at"] or ""
            if key not in latest or ca > latest[key]:
                latest[key] = ca
        for r in rows:
            key = (r["entry"], r["source_dir"] or "")
            if (r["created_at"] or "") < latest[key]:
                self.delete(r["id"])
                removed += 1
        # 规则 4：root 变化归并（在批内去重前做——否则同功能点的两批各自成立）
        rows = self._conn.execute(
            "SELECT * FROM test_cases ORDER BY id").fetchall()
        recs = [{"id": r["id"], "entry": r["entry"],
                 "sd": r["source_dir"] or "", "ca": r["created_at"] or ""} for r in rows]
        groups: list[list[dict]] = []
        for rec in recs:
            for g in groups:
                rep = g[0]
                if (self._entry_suffix(rec["entry"], rep["entry"])
                        or self._entry_suffix(rep["entry"], rec["entry"])) \
                        and self._is_related_root(rec["sd"], rep["sd"]):
                    g.append(rec)
                    break
            else:
                groups.append([rec])
        for g in groups:
            if len({x["sd"] for x in g}) <= 1:
                continue    # 同 root（规则 1/2/3 已管），组内多个 entry 视为不同功能点
            # 用 id（自增，单调）判定"最新"——created_at 秒级精度在快速连续操作时
            # 无法区分先后，导致归并结果不确定（实测同秒创建时挂/过随机）
            latest_sd = max({x["sd"] for x in g},
                            key=lambda d: max(x["id"] for x in g if x["sd"] == d))
            keep_id = max(x["id"] for x in g if x["sd"] == latest_sd)
            for x in g:
                if x["sd"] != latest_sd or x["id"] < keep_id:
                    self.delete(x["id"])
                    removed += 1
        # 规则 2/3：批内签名/描述去重（保留最新 id）
        rows = self._conn.execute(
            "SELECT * FROM test_cases ORDER BY id").fetchall()
        seen_sig: set = set()
        seen_desc: set = set()
        for r in rows:
            case = self._row_to_case(r)
            sig = (r["entry"], r["source_dir"] or "", self._case_signature(case.to_dict()))
            desc = (r["entry"], r["source_dir"] or "", (case.description or "").strip())
            if sig in seen_sig or (desc[2] and desc in seen_desc):
                self.delete(r["id"])
                removed += 1
            else:
                seen_sig.add(sig)
                seen_desc.add(desc)
        return removed

    def attach_review(self, entry: str, review: dict[str, Any] | None,
                      module: str | None = None):
        """给某入口的所有用例挂上评审（评分 + 详情）。"""
        if not review:
            return
        score = review.get("score")
        review_data = json.dumps(review, ensure_ascii=False)
        if module:
            self._conn.execute(
                "UPDATE test_cases SET review_score=?, review_data=? WHERE entry=? AND module=?",
                (score, review_data, entry, module))
        else:
            self._conn.execute(
                "UPDATE test_cases SET review_score=?, review_data=? WHERE entry=?",
                (score, review_data, entry))
        self._conn.commit()

    def list_by_entry(self, entry: str) -> list[StoredCase]:
        rows = self._conn.execute(
            "SELECT * FROM test_cases WHERE entry=? ORDER BY id", (entry,)).fetchall()
        return [self._row_to_case(r) for r in rows]

    def get_by_entry_source(self, entry: str, source_dir: str = "") -> list[StoredCase]:
        """按 entry + source_dir 查询已落库的用例（缓存读取）。

        返回该功能点在该目录下的全部历史用例，按 id 升序。
        用于"重复生成同一功能点"时命中缓存，避免重复调 LLM。
        """
        if source_dir:
            rows = self._conn.execute(
                "SELECT * FROM test_cases WHERE entry=? AND source_dir=? ORDER BY id",
                (entry, source_dir)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM test_cases WHERE entry=? ORDER BY id", (entry,)).fetchall()
        return [self._row_to_case(r) for r in rows]

    def cache_hit(self, entry: str, source_dir: str = "") -> bool:
        """判断该功能点是否已有落库用例（缓存命中）。"""
        return bool(self.get_by_entry_source(entry, source_dir))

    def list_by_module(self, module: str) -> list[StoredCase]:
        rows = self._conn.execute(
            "SELECT * FROM test_cases WHERE module=? ORDER BY id", (module,)).fetchall()
        return [self._row_to_case(r) for r in rows]

    def all(self) -> list[StoredCase]:
        rows = self._conn.execute("SELECT * FROM test_cases ORDER BY id").fetchall()
        return [self._row_to_case(r) for r in rows]

    def get(self, case_id: int) -> StoredCase | None:
        row = self._conn.execute("SELECT * FROM test_cases WHERE id=?", (case_id,)).fetchone()
        return self._row_to_case(row) if row else None

    def delete(self, case_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM test_cases WHERE id=?", (case_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def entries(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT entry FROM test_cases ORDER BY entry").fetchall()
        return [r["entry"] for r in rows]

    @staticmethod
    def _row_to_case(row) -> StoredCase:
        def _load(s):
            try:
                return json.loads(s) if s else None
            except (json.JSONDecodeError, TypeError):
                return None
        return StoredCase(
            id=row["id"],
            entry=row["entry"],
            module=row["module"],
            description=row["description"],
            request=_load(row["request"]),
            expected_status=row["expected_status"],
            expected_results=_load(row["expected_results"]) or [],
            test_steps=_load(row["test_steps"]) or [],
            review_score=row["review_score"],
            review_data=_load(row["review_data"]),
            created_at=row["created_at"],
            source_dir=row["source_dir"],
            fingerprint=row["fingerprint"] if "fingerprint" in row.keys() else "",
        )

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print(__doc__)
