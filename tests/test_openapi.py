

# ---------------- 7.7.56 原生 FastAPI 默认 operationId 的确定性匹配 ----------------

def _native_app():
    """无自定义 generate_unique_id 的原生 FastAPI 应用（默认 operationId 格式）。"""
    from fastapi import FastAPI
    app = FastAPI()

    @app.post('/api/v1/users/')
    def create_user():
        return {}

    @app.post('/api/v1/private/users/')
    def create_user_private():
        return {}
    return app


def test_match_native_operation_id_disambiguates_same_name():
    """原生默认格式（op_id=函数名+路径转写+method）：同名函数各自精确命中。"""
    from pyst.core.openapi import parse_endpoints, match_endpoints
    eps = parse_endpoints(_native_app().openapi())
    r1 = match_endpoints(eps, entry='app.api.routes.users.create_user',
                         module='app.api.routes.users')
    assert [e['url'] for e in r1['endpoints']] == ['/api/v1/users/']
    assert r1['match'] == 'operation_id' and r1['ambiguous'] is False
    r2 = match_endpoints(eps, entry='app.api.routes.private.create_user_private',
                         module='app.api.routes.private')
    assert [e['url'] for e in r2['endpoints']] == ['/api/v1/private/users/']
    assert r2['ambiguous'] is False
