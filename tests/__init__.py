"""pyst 自动化测试套件。

纪律（7.7.44）：
- 数据库测试用 tmp_path 临时库，永不触碰 pyst/test_cases.db
- LLM 调用一律 mock，测试永不打外部 API
- 每个测试对应一个真实踩过的 bug（见开发计划 7.7.44 清单）
"""
