"""assertions.py 测试：意图提取词表、可接受状态集、verdict 判定、响应体结构检查。"""
from pyst.eval.assertions import (assert_case, intent_of_case,
                                  response_body_checks)


def _intent(desc, results=None):
    return intent_of_case({"description": desc, "expected_results": results or []})


def test_intent_extraction_table():
    assert _intent("未认证用户访问：不携带认证信息") == "auth"
    assert _intent("无效 JWT 创建条目：验证鉴权失败") == "auth"
    assert _intent("title 为超长字符串：验证长度上限校验") == "validation"
    assert _intent("缺少必填字段 title：验证字段校验失败") == "validation"
    assert _intent("请求体为非法 JSON：验证解析错误处理") == "validation"
    assert _intent("正常创建条目：提供合法的请求体") == "success"
    assert _intent("可选字段缺失：仅提供 title，可省略") == "success"
    assert _intent("创建后验证数据持久化：确认已写入数据库") == "success"
    assert _intent("删除不存在的条目：验证资源不存在") == "not_found"
    assert _intent("多余字段：验证是否被忽略或拒绝") == "probe"


def test_intent_priority_auth_over_others():
    assert _intent("未认证访问：验证校验失败返回") == "auth"


def test_verdict_success_intent():
    r = assert_case({"description": "正常创建：验证成功返回", "expected_status": 200}, 200, "{}")
    assert r["verdict"] == "PASS" and r["mode"] == "exact"


def test_verdict_validation_accepts_400_and_422():
    case = {"description": "缺少必填字段：验证校验失败", "expected_status": 422, "expected_results": []}
    assert assert_case(case, 422, '{"detail":[]}')["verdict"] == "PASS"
    assert assert_case(case, 400, '{"detail":"x"}')["verdict"] == "PASS"


def test_verdict_rejects_wrong_semantics():
    case = {"description": "缺少必填字段：验证校验失败", "expected_status": 422, "expected_results": []}
    r = assert_case(case, 500, "err")
    assert r["verdict"] == "FAIL" and "意图不符" in r["reason"]


def test_body_checks_4xx_detail_and_2xx_json():
    assert any("detail" in c for c in response_body_checks("validation", 422, '{"detail":[]}'))
    assert any("JSON 可解析" in c for c in response_body_checks("success", 200, '{"id":1}'))
    assert any("不是合法 JSON" in c for c in response_body_checks("success", 200, "<html>"))
