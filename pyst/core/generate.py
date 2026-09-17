#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4：基于功能点档案生成测试用例（开发计划 M4）
====================================================
把 Phase 3 产出的 FeaturePoint 档案发给大模型，生成测试用例。

【本阶段回答】"给定一个功能点档案，能生成多少条高质量测试用例？"

【输入】FeaturePoint.to_dict()（JSON 档案，含签名/调用子图/控制流/外部依赖）
【输出】list[dict] 测试用例，每条含 description / test_steps / expected_results
        （与 TestBrain 的 test_case_generator 输出格式对齐，方便后续落库复用）

【设计参考】pythonTest 是独立项目，不依赖 TestBrain。但 TestBrain 的架构是设计参考：
    - 确定性分析交给 AST 工具，设计交给 LLM，只喂结构化事实（对齐 java_code_analyzer）
    - prompt 风格参考 apps/ai_agents/test_case_generator/configs/prompt_config.yaml
    - 与路径 B（纯需求文本）的区别：输入是"代码事实档案"，不是裸需求 ——
      这正是"喂源码 + AST 摘要"的中间态

【铁律（对齐 java_code_analyzer 的 prompt 约束）】
    只能依据档案中的事实设计用例，不得编造档案之外的类/方法/接口/字段。

【可测试性设计】
    build_prompt 是纯函数，不依赖任何外部服务，可独立单测。
    generate_test_cases 接受可注入的 llm_invoke 回调；默认实现走独立 llm_client
    （OpenAI 兼容协议），不依赖 TestBrain 的 Django 环境。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .features import FeaturePoint


@dataclass
class TestCase:
    description: str
    test_steps: list[str]
    expected_results: list[str]
    request: dict[str, Any] | None = None   # 可直接执行的 HTTP 请求参数（Phase 5）
    expected_status: int | None = None      # 预期 HTTP 状态码（断言依据）
    call_args: list[Any] | None = None      # 纯函数调用位置参数（test_level="func"）
    call_kwargs: dict[str, Any] | None = None  # 纯函数调用关键字参数（test_level="func"）

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "test_steps": self.test_steps,
            "expected_results": self.expected_results,
            "request": self.request,
            "expected_status": self.expected_status,
            "call_args": self.call_args,
            "call_kwargs": self.call_kwargs,
        }


def _render_signature(feature: dict[str, Any]) -> str:
    """渲染入口签名（缺省时回退到 entry 限定名）"""
    sig = feature.get("signature")
    if sig:
        return sig
    return feature.get("entry", "unknown")


def _render_call_edges(feature: dict[str, Any]) -> str:
    """渲染本地调用子图：缩进 + 行号 + callee + 每个下游节点的控制逻辑。

    改动2：每个节点带下游函数自身的控制流骨架（分支/循环/异常），
    让 LLM 生成用例时能覆盖下游的边界和异常路径。
    用深度缩进 + 层级标注，避免 LLM 混淆节点归属。
    """
    edges = feature.get("call_edges") or []
    if not edges:
        return "  （无本地调用）"
    lines = []
    for e in edges:
        depth = e.get("depth", 1)
        pad = "  " * depth
        callee = e.get("callee", "?")
        line = e.get("line", "?")
        lines.append(f"{pad}L{line}: {callee}  [{e.get('status', '')}]")
        # 下游节点的控制逻辑（改动2 核心）
        controls = e.get("controls") or []
        if controls:
            lines.append(f"{pad}  └─ 该函数控制逻辑:")
            for c in controls:
                cond = f" {c.get('cond', '')}" if c.get("cond") else ""
                branch = f" [{c.get('branch')}]" if c.get("branch") else ""
                lines.append(f"{pad}     L{c.get('line', '?')}: {c.get('kind', '?')}{cond}{branch}")
        else:
            lines.append(f"{pad}  └─ （该函数无本地控制流或为外部依赖）")
    return "\n".join(lines)


def _render_external_calls(feature: dict[str, Any]) -> str:
    """渲染外部依赖：第三方/对象属性链"""
    exts = feature.get("external_calls") or []
    if not exts:
        return "  （无）"
    return "  " + ", ".join(str(x) for x in exts)


def _render_control_sites(feature: dict[str, Any]) -> str:
    """渲染控制流骨架：每条 if/for/while/try 一行"""
    sites = feature.get("control_sites") or []
    if not sites:
        return "  （无分支/循环/异常处理）"
    lines = []
    for c in sites:
        cond = f" {c.get('cond')}" if c.get("cond") else ""
        branch = f" [{c.get('branch')}]" if c.get("branch") else ""
        lines.append(f"  L{c.get('line', '?')}: {c.get('kind', '?')}{cond}{branch}")
    return "\n".join(lines)


def build_refine_prompt(feature: dict[str, Any], cases: list[dict[str, Any]],
                        review: dict[str, Any], case_count: int = 10) -> str:
    """构建"根据评审意见重新生成用例"的 prompt。

    输入三份信息，要求 LLM 按评审意见修正现有用例并补充缺失场景：
      - 功能点档案（接口/控制流/签名）
      - 原测试用例（待修正）
      - 评审意见（weaknesses/missing_scenarios/suggestions）

    记忆管理暂不考虑：每次把档案/原用例/评审打包发给 LLM，不依赖上下文记忆。
    """
    # 功能点基础信息
    entry = feature.get("entry", "?")
    signature = feature.get("signature") or entry
    docstring = feature.get("docstring") or "（无）"
    control_sites = feature.get("control_sites") or []
    control_render = "\n".join(
        f"  L{c.get('line', '?')}: {c.get('kind')} {c.get('cond', '') or ''}"
        for c in control_sites
    ) or "  （无）"
    iface = feature.get("interface")
    iface_render = f"  URL: {iface.get('url')} | Method: {', '.join(iface.get('methods', []))}" if iface else "  （无接口信息）"

    # 原用例
    cases_render = []
    for i, c in enumerate(cases, 1):
        cases_render.append(
            f"[{i}] {c.get('description', '')}\n"
            f"    request: {c.get('request')}\n"
            f"    预期状态码: {c.get('expected_status', '无')}\n"
            f"    预期结果: {c.get('expected_results', [])}\n"
            f"    步骤: {c.get('test_steps', [])}"
        )
    cases_text = "\n".join(cases_render) or "（无现有用例）"

    # 评审意见
    review_render = (
        f"评分: {review.get('score')}/10  建议: {review.get('recommendation')}\n"
        f"缺点: {review.get('weaknesses')}\n"
        f"缺失场景: {review.get('missing_scenarios')}\n"
        f"改进建议: {review.get('suggestions')}"
    )
    orig_n = len(cases)
    missing_n = len(review.get("missing_scenarios") or [])

    return f"""你是一位专业的软件测试专家。之前生成了一批测试用例并做了 AI 评审，现在请你**根据评审意见修正并完善这些测试用例**，重新输出改进后的完整用例集。

【铁律】只能依据下面"功能点档案"中列出的事实设计用例，不得编造档案之外的类、方法、接口、字段、URL 或行为。

【功能点档案】
入口符号: {entry}
签名: {signature}
业务说明: {docstring}
控制流骨架（分支/循环/异常处理）:
{control_render}
接口信息:
{iface_render}

【原测试用例】（请在其基础上修正，不是全盘重写，共 {orig_n} 条）
{cases_text}

【评审意见】（必须逐条吸收，修正对应的问题；共指出 {missing_n} 个缺失场景）
{review_render}

【任务】
请根据评审意见：
1. 修正原用例中评审指出的缺点（weaknesses）
2. 补充评审指出的缺失场景（missing_scenarios）
3. 采纳合理的改进建议（suggestions）
4. 保留原用例中正确的部分，不要退化
5. 最终输出 {case_count} 条改进后的完整用例（目标构成：原 {orig_n} 条修正保留 + 补充 {missing_n} 个缺失场景；
   缺失场景多于该数量时可适当突破，按实际需要输出，但禁止为凑数生成重复/低价值用例）

【输出格式要求】（与之前一致，必须严格 JSON 数组，含 []，前后无废话）
每条用例包含:
- "description": 用例描述
- "request": 可执行 HTTP 请求（method/url/headers/query/body/files）
- "expected_status": 预期 HTTP 状态码
- "test_steps": 步骤列表
- "expected_results": 预期结果列表

【严禁】字符串值内部出现未转义的英文引号 "（会破坏 JSON 导致解析失败）。
如需在描述/预期中表达引用，请使用中文引号 “ ” 或不加引号。
所有字符串值必须用英文双引号包裹，值内部的引号必须转义为 \\"。
【边界值表示】超长字符串/大文本边界禁止真的重复几百字（会打爆输出 token 导致截断）；用简短占位描述即可。

[
  {{
    "description": "改进后的用例描述",
    "request": {{"method": "GET", "url": "/path", "headers": {{}}, "query": {{}}, "body": {{}}, "files": {{}}}},
    "expected_status": 200,
    "test_steps": ["1. 步骤1"],
    "expected_results": ["1. 结果1"]
  }}
]
"""


def _render_interface(feature: dict[str, Any]) -> str:
    """渲染 HTTP 接口信息（Phase 5；支持多接口——OpenAPI 探测可能 1 个功能点对应多个路由）"""
    ifaces = feature.get("interfaces") or []
    if not ifaces and feature.get("interface"):
        ifaces = [feature["interface"]]
    if not ifaces:
        return '  （未提取到接口定义，若为 HTTP 入口请标注"事实数据不足"）'

    def _render_one(iface: dict[str, Any], idx: int | None = None) -> list[str]:
        lines = []
        head = f"  [{idx}] " if idx is not None else "  "
        lines.append(f"{head}URL: {iface.get('url', '?')}")
        lines.append(f"{head}Method: {', '.join(iface.get('methods', []))}")
        if iface.get("operation_id"):
            lines.append(f"{head}operationId: {iface['operation_id']}")
        params = iface.get("params") or []
        if params:
            lines.append(f"{head}请求参数:")
            for p in params:
                lines.append(f"{head}  - {p.get('name')}  [位置: {p.get('location')}]  [类型: {p.get('type_hint') or '未知'}]")
        if iface.get("file_field"):
            lines.append(f"{head}文件上传字段: {iface['file_field']}")
        return lines

    lines: list[str] = []
    if len(ifaces) == 1:
        lines.extend(_render_one(ifaces[0]))
    else:
        lines.append(f"  （该入口对应 {len(ifaces)} 个接口，请按测试场景选用正确的 method+URL）")
        for i, iface in enumerate(ifaces, 1):
            lines.extend(_render_one(iface, i))
    return "\n".join(lines)


def build_prompt(feature: dict[str, Any], case_count: int = 10,
                 case_design_methods: str = "", case_categories: str = "",
                 test_level: str = "api") -> str:
    """把 FeaturePoint 档案渲染成喂给 LLM 的 prompt 文本。

    结构（对齐 java_code_analyzer 的"事实数据"风格）：
     1. 系统角色 + 铁律（只依据档案事实，不编造）
     2. 功能点入口签名 + docstring（业务语义）
     3. 本地调用子图（它调用了哪些本地符号）
     4. 控制流骨架（if/for/while/try → 场景线索）
     5. 外部依赖（第三方调用标注）
     6. 输出约束（JSON 格式）

    Args:
        feature: FeaturePoint.to_dict() 产物
        case_count: 期望生成用例条数
        case_design_methods: 用例设计方法（等价类/边界值/场景法等，逗号分隔）
        case_categories: 用例类型（功能/性能/兼容/安全等，逗号分隔）
        test_level: "api"=输出可执行 HTTP request（默认）；"func"=输出纯函数调用参数 call_args/call_kwargs
    """
    entry = feature.get("entry", "?")
    signature = _render_signature(feature)
    docstring = feature.get("docstring") or "（无）"
    module = feature.get("module", "?")
    line_range = feature.get("line_range") or [0, 0]
    decorators = feature.get("decorators") or []
    call_edges = _render_call_edges(feature)
    external_calls = _render_external_calls(feature)
    control_sites = _render_control_sites(feature)
    interface = _render_interface(feature)

    methods = case_design_methods or "所有适用的测试用例设计方法"
    categories = case_categories or "所有适用的测试类型"

    # 按 test_level 决定输出格式（func 模式渲染纯函数调用约束，api 模式渲染 HTTP request 约束）
    if test_level == "func":
        output_section = _build_func_output_section(interface)
    else:
        output_section = _build_api_output_section(interface)

    return f"""你是一位专业的软件测试专家，擅长根据代码静态分析结果设计全面的、可直接执行的测试用例。

【铁律】只能依据下面"功能点档案"中列出的事实设计用例，不得编造档案之外的类、方法、接口、字段、URL 或行为。档案里没有的信息，明确标注"事实数据不足"。

【功能点档案】
入口符号: {entry}
所在模块: {module}
代码范围: 行 {line_range[0]}-{line_range[1]}
签名: {signature}
业务说明: {docstring}
装饰器: {', '.join(decorators) if decorators else '无'}

{output_section}

本地调用（该入口直接调用的本地符号）:
{call_edges}

控制流骨架（分支/循环/异常处理，测试场景的关键线索）:
{control_sites}

外部依赖（第三方/对象属性链，标注参考）:
{external_calls}

【任务】
请根据以上功能点档案，使用{methods}，为该功能点生成{case_count}条{categories}的测试用例。

【输出格式要求——必须严格遵守，否则解析失败】
- 必须严格返回 JSON 数组，包含在 [] 中
- 不要在 JSON 之前或之后添加任何解释文本
- 所有字符串的值必须用英文双引号包裹
- 字符串内部【严禁使用任何引号】：
  - 禁止使用英文引号 "（否则会破坏 JSON 结构导致解析失败）
  - 如需表达引用，请使用中文引号 “ ”（全角）或不加引号
  - 例如：正确写法是 "expected_results": ["1. 返回未提供用户ID错误"]，绝不能写成 "未提供用户ID" 这种带英文引号的形式
- 字符串内若确实需要英文引号，必须转义为 \\"
- 数组元素之间用逗号分隔
- 字段名必须是双引号包裹的字符串

输出格式:
{_FUNC_OUTPUT_EXAMPLE if test_level == "func" else _API_OUTPUT_EXAMPLE}
"""


_API_OUTPUT_EXAMPLE = """[
  {{
    "description": "测试用例描述",
    "request": {{
      "method": "GET",
      "url": "/path/to/api",
      "headers": {{}},
      "query": {{"param1": "value1"}},
      "body": {{}},
      "files": {{}}
    }},
    "expected_status": 200,
    "test_steps": ["1. 步骤1", "2. 步骤2"],
    "expected_results": ["1. 结果1", "2. 结果2"]
  }}
]"""


_FUNC_OUTPUT_EXAMPLE = """[
  {{
    "description": "测试用例描述",
    "call_args": ["实际值1", 30],
    "call_kwargs": {{"可选关键字参数": "值"}},
    "test_steps": ["1. 步骤1", "2. 步骤2"],
    "expected_results": ["1. 结果1", "2. 结果2"]
  }}
]"""


def _build_api_output_section(interface: dict | None) -> str:
    """api 形态：HTTP 接口信息 + request 字段约束。"""
    iface_render = f"""【HTTP 接口信息】（这是可直接执行的关键，务必依据它构造 request）
{interface if interface else "  （未提取到接口定义）"}"""
    return iface_render + """

【任务补充】
每条用例必须能直接通过 HTTP 请求执行，因此必须包含可执行的 request 参数。

生成的每条测试用例必须包含以下字段:
1. "description": 测试用例描述，简明扼要说明测试目的和内容
2. "request": 可直接执行的 HTTP 请求参数（必须依据上面的【HTTP 接口信息】构造）:
   - "method": HTTP 方法（如 "GET"/"POST"）
   - "url": 接口 URL（用上面的 URL，可拼接 query，如 "/api/x?a=1"）
   - "headers": 可选，请求头对象（如 {"Content-Type": "application/json"}）
   - "query": 可选，query 参数对象（GET 请求的参数放这里）
   - "body": 可选，请求体对象（POST 的 form/json 参数放这里）
   - "files": 可选，文件上传对象（{"字段名": "文件名或文件路径"}，仅当接口有文件字段时）
3. "test_steps": 测试步骤列表，从 1 到 n 编号
4. "expected_results": 每个步骤对应的预期结果列表，从 1 到 n 编号
5. "expected_status": 预期 HTTP 状态码（整数，如 200/400/404/500）。这是执行断言的依据：
   - 正常/成功路径：写 200（或真实的成功状态码）
   - 参数错误/校验失败：写 400
   - 资源不存在：写 404
   - 无法确定时，根据接口信息推断，不确定就省略该字段
   - 必须与 expected_results 描述一致（如描述"返回错误提示"，则状态码应为 4xx）
6. 【鉴权占位约定】需要有效凭证的用例，Authorization 头必须且只能写
   "Authorization": "Bearer <valid_jwt>"（执行时平台自动替换为真实 token）；
   严禁使用其他占位写法（如 <有效token>、<your_token>）或中文占位——HTTP 头仅允许
   ASCII 字符，非法写法会导致请求根本发不出去
   【负向凭证占位】测试"无效/过期凭证被拒绝"的用例，用 "Bearer <invalid_jwt>" /
   "Bearer <expired_jwt>" 占位（平台原样发送，服务端会返回 401/403——正是预期）；
   严禁给这类用例写 <valid_jwt>（会被替换为有效 token，用例意图被破坏）

【约束】
- request 里的 method/url/参数名必须来自【HTTP 接口信息】，不得编造接口信息中不存在的 URL 或参数名
- 接口信息里的参数，用例要合理赋值（正常值、边界值、空值、非法值等，取决于测试场景）
- 【边界值表示】构造超长字符串/大文本边界时，禁止在 JSON 里真的重复几百字（会打爆输出 token 导致截断）；用占位描述并以 < > 包裹，执行时平台会自动展开为真实超长字符串，如 title 值写 "<超长字符串：超过 title 长度上限（如 256 字符）>"
- 【可自动化约束】严禁生成依赖故障注入/数据库 mock/内部状态篡改的用例（如"模拟数据库插入失败验证 500"）——测试从外部发 HTTP 请求，无法控制服务内部，这类用例不可执行；错误处理场景只能通过外部手段构造（非法参数、越界值、不存在的资源 ID）
- 如果【HTTP 接口信息】显示"未提取到接口定义"，则 request 只能给出推测值并标注"推测"二字"""


def _build_func_output_section(interface: dict | None) -> str:
    """func 形态：被测函数是纯函数（非 HTTP 入口），输出 call_args/call_kwargs。"""
    return """【被测对象说明】
该功能点是一个【纯函数】（非 HTTP 接口），直接在代码中调用执行。没有 URL / HTTP 请求。
函数签名中的每个参数都要根据其语义设计测试值：正常值、边界值、空值（None/空串）、非法类型/非法值等。

【任务补充】
每条用例必须包含被测函数的调用参数，用于直接调用该函数执行：
1. "description": 测试用例描述，简明扼要说明测试目的和内容
2. "call_args": 调用该函数时的【位置参数列表】，按签名顺序填入实际值
   - 如函数是 def validate_user(name, age)，则 call_args: ["张三", 30]
   - 参数值要与测试场景匹配（正常/空/边界/非法）
3. "call_kwargs": 可选，调用时的关键字参数对象（如 {"name": "张三", "age": 30}）；用位置参数填充即可留空 {}
4. "test_steps": 测试步骤列表，从 1 到 n 编号
5. "expected_results": 每个步骤对应的预期结果列表，从 1 到 n 编号

【约束】
- call_args 的元素个数与顺序，必须与被测函数签名一致
- call_args 里不得包含"被测函数签名之外"的参数名或参数
- 【边界值表示】构造超长字符串/大文本边界时，禁止真的重复几百字（会打爆输出 token 导致截断）；用简短占位描述，如 "超长字符串：重复A至超过长度上限"
- 字符串参数值用英文双引号包裹；数字/布尔/None 直接用字面量"""



# ---------- JSON 解析与校验（参考 test_case_generator） ----------

def _extract_json_array(text: str) -> str:
    """从 LLM 响应中提取 JSON 数组文本（容忍前后废话/截断）。

    策略（按顺序）：
      1. 括号配平：从第一个 '[' 开始做括号配平，找到真正的外层 ']'
         （跳过字符串内的 '[' / ']'，避免被 test_steps 里的 [] 干扰）
      2. 截断补全：配平不到结尾 ']' 时，从最后一个完整 '}' 处截断并补 ']'
    """
    if not text:
        return ""
    start = text.find("[")
    if start == -1:
        return ""
    # 1) 括号配平扫描，找出最外层数组的结束位置
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]   # 找到完整数组
        i += 1
    # 2) 无结尾 ']'：模型输出被截断（常见于输出 token 打爆，如用例里带了超长重复字符串）
    body = text[start:]
    # 从尾部逐个候选切割点（} 或 "）向前试解析，取第一个能配平的——
    # 单一 rfind("},") 不够：截断点可能落在最后一个用例内部（如字符串中间），
    # 此时最近的 }, 在残缺用例内部，切出来必然非法。json.loads 验证保证切出来的必合法。
    import json as _json
    candidates = []
    for i in range(len(body) - 1, start, -1):
        if body[i] in "}\"":
            candidates.append(i)
            if len(candidates) >= 60:          # 足够覆盖最后几个用例，控制开销
                break
    for cut in candidates:
        cand = body[: cut + 1].rstrip().rstrip(",")   # 顺带清掉可能的尾逗号
        try:
            _json.loads(cand + "]")
            return cand + "]"
        except Exception:
            continue
    return ""


def _parse_llm_json(text: str) -> Any:
    """容错解析 LLM 返回的 JSON。

    先尝试标准 json.loads；失败时做启发式修复：
    - LLM 常见问题：字符串值内误用未转义的英文引号（如 "未提供用户ID"）
      把"值字符串内部"的多余引号替换为中文引号（“ ”），不破坏 JSON 结构。
    仅做这一种安全修复；还失败则抛出原异常。
    """
    import json as _json

    first_err: Exception | None = None
    try:
        return _json.loads(text)
    except Exception as e:
        # 记录第一次的异常，若修复后仍失败则抛回原异常
        first_err = e

    repaired = _repair_quotes_in_values(text)
    try:
        return _json.loads(repaired)
    except Exception:
        if first_err:
            raise first_err
        raise


def _repair_quotes_in_values(text: str) -> str:
    """把 JSON 字符串值内部的未转义英文引号替换为中文引号。

    原理：扫描文本。每当遇到一个英文引号 " 时，尝试向后找到它的配对引号
    （最近的下一个 "）。这对引号包着的就是"字符串值"。
    但如果 LLM 在值内部又写了未转义的 "，扫描会提前闭合。

    本实现采用"保守修复"：逐字符扫描，维护 in_str 状态。进入字符串后，
    遇到一个未转义的 " 时，检查它是否是真正的字符串收尾——判断标准是：
    它之后的非空白字符必须是 , ] } 或文本结束（说明这是合法的值结尾）。
    如果不是，说明这是值内部的多余引号 → 替换为中文引号 “”。
    """
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if ch == "\\" and in_str and i + 1 < n:
            out.append(ch)
            out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
                i += 1
                continue
            # in_str 中遇到引号：判断是否字符串收尾
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ",]}:":
                # 合法的字符串收尾（后面是结构分隔符）
                in_str = False
                out.append(ch)
            else:
                # 值内部的多余引号 → 中文引号
                out.append("“")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# 合法 HTTP 方法（api 模式校验用）：防 LLM 伪造伪协议请求
_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


def _validate_test_cases(raw: Any, case_count: int, test_level: str = "api") -> list[TestCase]:
    """逐条校验并裁剪测试用例，非法条目丢弃。

    Args:
        raw: LLM 返回的原始 JSON 数组
        case_count: 期望条数上限
        test_level: "func" 时校验 call_args（纯函数调用参数）；"api" 时校验 request
    """
    valid: list[TestCase] = []
    if not isinstance(raw, list):
        return valid
    for item in raw:
        try:
            if not isinstance(item, dict):
                continue
            desc = item.get("description")
            steps = item.get("test_steps")
            expected = item.get("expected_results")
            if not isinstance(desc, str) or not desc.strip():
                continue
            if not isinstance(steps, list) or not steps:
                continue
            if not isinstance(expected, list) or not expected:
                continue
            req = item.get("request")
            exp_status = item.get("expected_status")
            call_args = item.get("call_args")
            call_kwargs = item.get("call_kwargs")
            if test_level == "func":
                # 纯函数形态：必须带 call_args（可执行调用参数），否则丢弃
                if not isinstance(call_args, list):
                    continue
                valid.append(TestCase(desc.strip(), steps, expected,
                                      request=None,
                                      expected_status=None,
                                      call_args=call_args,
                                      call_kwargs=call_kwargs if isinstance(call_kwargs, dict) else None))
            else:
                # api 模式校验：request 必须是**合法 HTTP 请求**。
                # 实测坑：非 HTTP 入口（纯函数）在 api 模式下，LLM 会把"函数调用"伪装成
                # request（method=CALL / url=callable://模块.函数）——必须拒绝，否则执行器
                # 会把它当 HTTP 发出去，得到毫无意义的 400。
                if not isinstance(req, dict):
                    continue
                method = str(req.get("method") or "").upper()
                url = req.get("url")
                if method not in _HTTP_METHODS:
                    continue
                if not isinstance(url, str) or not (
                        url.startswith("/") or url.startswith("http://") or url.startswith("https://")):
                    continue
                # expected_status 必须是合理的 HTTP 状态码（0/负数/超范围 → 视为无断言，走关键词兜底）
                if isinstance(exp_status, int) and not (100 <= exp_status <= 599):
                    exp_status = None
                valid.append(TestCase(desc.strip(), steps, expected,
                                      request=req,
                                      expected_status=exp_status
                                      if isinstance(exp_status, int) else None))
        except Exception:
            continue
    return valid[:case_count]


# ---------- LLM 调用 ----------

def _default_llm_invoke(prompt: str, provider: str, **kwargs) -> str:
    """默认 LLM 调用：走独立 LLM 客户端（OpenAI 兼容协议，不依赖 TestBrain）。

    kwargs 可透传：api_key、base_url、model、temperature 等。
    详见 llm_client.py —— pythonTest 自包含，仅把 TestBrain 作为设计参考。
    """
    from ..llm.client import chat_completion
    return chat_completion(prompt, provider=provider, **kwargs)


def generate_test_cases(feature: "FeaturePoint", provider: str = "deepseek",
                        case_count: int = 10,
                        case_design_methods: str = "",
                        case_categories: str = "",
                        test_level: str = "api",
                        llm_invoke: Callable[[str, str], str] | None = None,
                        **llm_kwargs: Any) -> list[TestCase]:
    """调用 LLM 生成测试用例。

    Args:
        feature: Phase 3 产出的 FeaturePoint
        provider: LLM 提供商（deepseek/qwen）
        case_count: 期望用例条数
        case_design_methods: 用例设计方法（逗号分隔）
        case_categories: 用例类型（逗号分隔）
        test_level: "api"=HTTP request 用例（默认）；"func"=纯函数 call_args 用例
        llm_invoke: 可注入的 LLM 调用回调 (prompt, provider) -> str，默认走独立 llm_client
        **llm_kwargs: 透传给 LLM 的额外参数

    Returns:
        校验通过的测试用例列表（≤ case_count 条）
    """
    # FeaturePoint 统一转成 dict
    feature_dict = feature.to_dict()
    prompt = build_prompt(
        feature_dict,
        case_count=case_count,
        case_design_methods=case_design_methods,
        case_categories=case_categories,
        test_level=test_level,
    )
    invoke = llm_invoke or (lambda p, prov: _default_llm_invoke(p, prov, **llm_kwargs))
    response = invoke(prompt, provider)

    json_text = _extract_json_array(response)
    if not json_text:
        raise ValueError("无法从 LLM 响应中提取 JSON 数组")
    try:
        raw = _parse_llm_json(json_text)
    except Exception as e:
        raise ValueError(f"解析 LLM JSON 失败: {e}\n原始响应: {response}")

    cases = _validate_test_cases(raw, case_count, test_level=test_level)
    if not cases:
        raise ValueError("LLM 未返回任何合法测试用例")
    return cases


def refine_test_cases(feature: dict[str, Any], cases: list[dict[str, Any]],
                      review: dict[str, Any], provider: str = "deepseek",
                      case_count: int = 10,
                      llm_invoke: Callable[[str, str], str] | None = None,
                      **llm_kwargs: Any) -> list[TestCase]:
    """根据评审意见重新生成（修正 + 补充）测试用例。

    Args:
        feature: 功能点档案 dict
        cases: 原测试用例 dict 列表（TestCase.to_dict()）
        review: 评审结果 dict（weaknesses/missing_scenarios/suggestions 等）
        provider: LLM 提供商
        case_count: 目标用例条数
        llm_invoke: 可注入的 LLM 回调（默认走独立 llm_client）
        **llm_kwargs: 透传给 LLM 的额外参数

    Returns:
        改进后的测试用例列表（TestCase），已做校验和裁剪
    """
    prompt = build_refine_prompt(feature, cases, review, case_count=case_count)
    invoke = llm_invoke or (lambda p, prov: _default_llm_invoke(p, prov, **llm_kwargs))
    response = invoke(prompt, provider)

    json_text = _extract_json_array(response)
    if not json_text:
        raise ValueError("无法从 LLM 响应中提取 JSON 数组")
    try:
        raw = _parse_llm_json(json_text)
    except Exception as e:
        raise ValueError(f"解析 LLM JSON 失败: {e}\n原始响应: {response}")

    refined = _validate_test_cases(raw, case_count)
    if not refined:
        raise ValueError("LLM 未返回任何合法测试用例")
    return refined


if __name__ == "__main__":
    print(__doc__)
