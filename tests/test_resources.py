"""资源占位符协议测试（7.7.45 扩展）：random uuid 静态替换、资源工厂、缺失资源提示。"""
import json

from pyst.eval.executor import (_build_resource_pool, _sub_uuid,
                                execute_suite_with_resources)


def test_sub_random_uuid_static():
    req = {"method": "DELETE", "url": "/api/v1/items/<random_uuid>"}
    out = _sub_uuid(req, lambda m: "11111111-1111-1111-1111-111111111111",
                    [__import__("re").compile(r"<(random|随机)[_-]?uuid>", 2)])
    assert out["url"].startswith("/api/v1/items/11111111-")
    assert "<random_uuid>" not in json.dumps(out)


def test_resource_pool_extracts_uuid_from_created_cases():
    cases = [{"description": "正常创建条目：提供合法的请求体，验证成功返回",
              "request": {"method": "POST", "url": "/api/v1/items/",
                          "body": {"title": "t"}}}]
    created = {"verdict": "PASS", "status": 200,
               "response_body": '{"title":"t","id":"aaaaaaa1-1111-1111-1111-111111111111"}'}
    import pyst.eval.executor as ex
    orig = ex.execute_http_case
    ex.execute_http_case = lambda case, base, token="": created
    pool = _build_resource_pool(cases, "http://b", "", limit=3)
    ex.execute_http_case = orig
    assert pool == ["aaaaaaa1-1111-1111-1111-111111111111"]


def test_resource_pool_ignores_non_uuid_id():
    cases = [{"description": "正常创建条目", "request": {"method": "POST", "url": "/x"}}]
    import pyst.eval.executor as ex
    orig = ex.execute_http_case
    ex.execute_http_case = lambda case, base, token="": {
        "verdict": "PASS", "status": 200, "response_body": '{"id": "not-a-uuid"}'}
    pool = _build_resource_pool(cases, "http://b", "")
    ex.execute_http_case = orig
    assert pool == []


def test_suite_with_resources_injects_ids(monkeypatch):
    """update 用例的 <item_id> 注入工厂预创建的真实 UUID"""
    import pyst.eval.executor as ex
    seq = []

    def fake_exec(case, base, token=""):
        seq.append(case)
        url = case["request"]["url"]
        if case["request"]["method"] == "POST" and "创建" in case.get("description", ""):
            return {"verdict": "PASS", "status": 200, "method": "POST",
                    "response_body": '{"id":"bbbbbbb2-2222-2222-2222-222222222222"}'}
        return {"verdict": "PASS" if "bbbbbbb2" in url else "FAIL",
                "status": 200 if "bbbbbbb2" in url else 404, "method": "PUT"}
    monkeypatch.setattr(ex, "execute_http_case", fake_exec)
    cases = [
        {"description": "正常创建条目：成功创建并返回", "request": {"method": "POST", "url": "/items",
         "body": {"title": "t"}}, "expected_status": 200},
        {"description": "更新已存在物品", "request": {"method": "PUT", "url": "/items/<item_id>",
         "body": {"title": "u"}}, "expected_status": 200},
    ]
    results, stats = execute_suite_with_resources(cases, "http://b", token="")
    # 工厂预执行 1 次 + 最终全量执行（含工厂用例本身）→ PUT 是最后一条调用
    upd = [c["request"]["url"] for c in seq if c["request"]["method"] == "PUT"][0]
    assert "bbbbbbb2-2222" in upd


def test_suite_missing_resource_hint(monkeypatch):
    """无可用资源（创建用例全失败）→ 结果 hint 指向资源准备问题"""
    import pyst.eval.executor as ex
    monkeypatch.setattr(ex, "execute_http_case",
                        lambda case, base, token="": {"verdict": "FAIL", "status": 404,
                                                      "method": case.get("request", {}).get("method", ""),
                                                      "response_body": "nf"})
    cases = [
        {"description": "正常创建条目：成功创建并返回", "request": {"method": "POST", "url": "/items",
         "body": {"title": "t"}}, "expected_status": 200},
        {"description": "更新已存在物品", "request": {"method": "PUT", "url": "/items/<item_id>",
         "body": {"title": "u"}}, "expected_status": 200},
    ]
    logs = []
    results, _ = execute_suite_with_resources(cases, "http://b", token="", log=logs.append)
    upd = [r for r in results if "PUT" == r["method"]][0]
    assert "资源占位符" in upd["reason"]
