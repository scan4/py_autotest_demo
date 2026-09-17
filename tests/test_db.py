"""db.py 测试：签名归一化、去重四规则、bug 注册表 CRUD（对应 7.7.22/7.7.42 真实 bug）。"""
import pytest

from pyst.storage.db import TestCaseStore


@pytest.fixture()
def store(tmp_path):
    s = TestCaseStore(tmp_path / "t.db")
    yield s
    s.close()


E, M, SD = "backend.app.api.routes.items.create_item", "m.py", "/proj"


def _case(desc, url="/api/v1/items/", method="POST", body=None, extra=None):
    req = {"method": method, "url": url}
    if body is not None:
        req["body"] = body
    if extra:
        req.update(extra)
    return {"description": desc, "request": req, "expected_status": 200,
            "test_steps": ["1"], "expected_results": ["1"]}


def test_signature_empty_keys_equivalent(store):
    store.save_test_cases(E, M, [_case("a", extra={"query": {}, "files": {}})], source_dir=SD)
    assert store.save_test_cases(E, M, [_case("a")], source_dir=SD) == []


def test_signature_body_value_difference_kept(store):
    store.save_test_cases(E, M, [_case("a", body={"title": "A"})], source_dir=SD)
    assert len(store.save_test_cases(E, M, [_case("b", body={"title": "B"})], source_dir=SD)) == 1


def test_signature_body_empty_object_is_meaningful(store):
    store.save_test_cases(E, M, [_case("a", url="/x", body={})], source_dir=SD)
    store.save_test_cases(E, M, [_case("b", url="/x")], source_dir=SD)
    descs = {c.description: c.request.get("body") for c in store.get_by_entry_source(E, SD)}
    assert descs["a"] == {} and descs["b"] is None


def test_save_dedup_same_signature(store):
    c = _case("x", body={"title": "A"})
    store.save_test_cases(E, M, [c], source_dir=SD)
    assert store.save_test_cases(E, M, [dict(c)], source_dir=SD) == []


def test_save_dedup_same_description_different_body(store):
    """7.7.42：描述一字不差、仅 body 值微差 → 视为重复（LLM 复制变体实测）"""
    store.save_test_cases(E, M, [_case("未认证访问", body={"title": "未认证条目"})], source_dir=SD)
    assert store.save_test_cases(E, M, [_case("未认证访问", body={"title": "测试条目"})], source_dir=SD) == []


def test_dedup_prunes_old_batches(store):
    store.save_test_cases(E, M, [_case("old1", body={"t": "1"}), _case("old2", body={"t": "2"})], source_dir=SD)
    store._conn.execute("UPDATE test_cases SET created_at='2026-01-01 00:00:00'")
    store._conn.commit()
    store.save_test_cases(E, M, [_case("new")], source_dir=SD)
    assert store.deduplicate() >= 2
    assert [c.description for c in store.get_by_entry_source(E, SD)] == ["new"]


def test_dedup_merges_root_renamed_entry(store):
    store.save_test_cases(E, M, [_case("a")], source_dir="/proj/backend")
    store.save_test_cases("backend." + E, M, [_case("b")], source_dir="/proj")
    store.deduplicate()
    descs = [c.description for c in store.all()]
    assert descs.count("a") == 0 and descs.count("b") == 1


def test_bug_finding_upsert_idempotent(store):
    f = {"method": "POST", "url": "/x"}
    i1 = store.upsert_bug_finding(E, SD, "空白 title 可入库", f, 400, {"actual_status": 200})
    i2 = store.upsert_bug_finding(E, SD, "空白 title 可入库", f, 400, {"actual_status": 200})
    assert i1 == i2 and len(store.list_bug_findings(source_dir=SD)) == 1


def test_bug_finding_get_delete_filter(store):
    fid = store.upsert_bug_finding(E, SD, "空白 title", {"method": "POST"}, 400,
                                   {"actual_status": 200})
    assert store.get_bug_finding(fid)["description"] == "空白 title"
    assert store.delete_bug_finding(fid)
    assert store.get_bug_finding(fid) is None
    assert not store.delete_bug_finding(fid)
