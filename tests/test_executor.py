"""executor.py 测试：body 格式、token 策略、占位符展开、路径预检、意图断言（7.7.18~7.7.45）。"""
from types import SimpleNamespace

from pyst.eval.executor import (_apply_token, _expand_long_value, _assert,
                                _is_token_placeholder, execute_http_case)


class FakeResp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


# ---------------- body 发送格式（7.7.18 json_invalid bug） ----------------

def test_body_dict_goes_json(monkeypatch):
    captured = {}
    def fake_request(method, url, **kw):
        captured.update(kw)
        return FakeResp(200, "{}")
    monkeypatch.setattr("pyst.eval.executor.requests.request", fake_request)
    r = execute_http_case({"description": "x", "request": {"method": "POST", "url": "/x",
                           "headers": {"Content-Type": "application/json"}, "body": {"a": 1}}})
    assert captured.get("json") == {"a": 1} and captured.get("data") is None


def test_body_form_goes_data(monkeypatch):
    captured = {}
    def fake_request(method, url, **kw):
        captured.update(kw)
        return FakeResp(200, "{}")
    monkeypatch.setattr("pyst.eval.executor.requests.request", fake_request)
    execute_http_case({"description": "x", "request": {"method": "POST", "url": "/token",
                      "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                      "body": {"username": "u", "password": "p"}}})
    assert captured.get("data") == {"username": "u", "password": "p"} and captured.get("json") is None


# ---------------- 头部非 ASCII 预检（7.7.19/7.7.26） ----------------

def test_non_ascii_header_rejected():
    r = execute_http_case({"description": "x", "request": {"method": "GET", "url": "http://localhost:1/x",
                           "headers": {"Authorization": "Bearer 超级用户token"}}})
    assert r["verdict"] == "ERROR" and "非 ASCII" in r["reason"]


def test_is_token_placeholder_variants():
    assert _is_token_placeholder("Bearer <valid_jwt>")
    assert _is_token_placeholder("Bearer <有效token>")
    assert _is_token_placeholder("Bearer 超级用户token")
    assert not _is_token_placeholder("Bearer eyJhbGciOi.J9.abc")
    assert not _is_token_placeholder("Bearer realtoken")


# ---------------- token 三级策略（7.7.23） ----------------

def test_apply_token_inject_when_missing():
    h = _apply_token({"description": "正常创建", "expected_status": 200},
                     {"Content-Type": "application/json"}, token="TOK")
    assert h["Authorization"] == "Bearer TOK"


def test_apply_token_skip_negative_case():
    h = _apply_token({"description": "未认证访问：不携带认证信息", "expected_status": 401},
                     {"Content-Type": "application/json"}, token="TOK")
    assert "Authorization" not in h


def test_apply_token_replaces_placeholder_any_variant():
    h = _apply_token({"description": "x", "expected_status": 200},
                     {"Authorization": "Bearer <有效token>"}, token="TOK")
    assert h["Authorization"] == "Bearer TOK"


# ---------------- 占位符展开（7.7.25） ----------------

def test_expand_long_over():
    v = _expand_long_value("<超长字符串：重复A至超过 title 字段长度上限（如 256 字符）>")
    assert len(v) == 512


def test_expand_long_exact():
    v = _expand_long_value("<超长字符串：恰好达到 title 长度上限（如 255 字符）>")
    assert len(v) == 255


def test_expand_long_default():
    assert len(_expand_long_value("<超长字符串：超过上限>")) == 5000


# ---------------- 意图断言（7.7.45） ----------------

def _assert_of(desc, exp, actual, body=""):
    resp = FakeResp(actual, body)
    return _assert({"description": desc, "expected_status": exp, "expected_results": []},
                   resp, {"entry": "e"})


def test_assert_exact_match():
    r = _assert_of("正常创建条目：验证成功返回", 200, 200)
    assert r["verdict"] == "PASS" and r["assertion"]["mode"] == "exact"


def test_assert_intent_validation_400_vs_422():
    """预期 400 实际 422：校验拒绝语义一致 → PASS（预期码写错的根治）"""
    r = _assert_of("缺少必填字段 title：验证字段校验失败", 400, 422, '{"detail":[{"loc":["body"]}]}')
    assert r["verdict"] == "PASS" and r["assertion"]["mode"] == "intent"


def test_assert_intent_auth_401_vs_403():
    r = _assert_of("无效 JWT 创建条目：验证鉴权失败", 401, 403)
    assert r["verdict"] == "PASS" and r["assertion"]["intent"] == "auth"


def test_assert_intent_mismatch_fail():
    r = _assert_of("正常创建：验证成功返回", 200, 500)
    assert r["verdict"] == "FAIL" and r["assertion"]["intent"] == "success"


def test_assert_probe_falls_back_exact():
    r = _assert_of("未知字段：验证是否被忽略或拒绝", 200, 404)
    assert r["verdict"] == "FAIL" and r["assertion"]["mode"] == "probe"
