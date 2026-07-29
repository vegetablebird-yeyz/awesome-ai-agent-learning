# 第9章：LangChain 快速上手

> 作者：青松与桑叶  
> 难度：进阶入门  
> 本章目标：看懂 LangChain 的核心抽象，并把第4—7章手写过的组件用框架重新组装起来

---

## 9.1 为什么现在才学 LangChain

你已经手写过消息列表、工具调用和 ReAct 循环，所以现在看 LangChain，不会觉得它在变魔术。它做的主要工作，是把提示词、模型、解析器、检索器和工具包装成统一接口，让组件更容易替换、组合和观测。

LangChain 适合这些场景：

- 希望用统一写法切换模型或检索器
- 需要把多个处理步骤串成管道
- 希望接入回调、追踪、流式输出等工程能力

如果程序只有一次模型调用，直接使用模型 SDK 往往更清楚。框架不是必选项，能减少重复代码时再用。

## 9.2 三个核心概念

### Prompt

`ChatPromptTemplate` 负责把变量填入消息模板。模板让提示词和业务代码分开，也能在调用前检查变量是否齐全。

### Model

`ChatOpenAI` 名字里虽然有 OpenAI，但也能连接许多兼容 OpenAI 协议的服务。模型对象接收消息，返回 `AIMessage`。

### Output Parser

`StrOutputParser` 把 `AIMessage` 转成普通字符串。需要结构化结果时，可以改用 JSON 或 Pydantic 解析器，并在业务层再次校验。

三者通过 **LCEL（LangChain Expression Language）** 连接：

```python
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"question": "什么是 Agent？"})
```

竖线不是把文本简单拼起来，而是把前一个组件的输出交给后一个组件。这个结构和命令行管道很像。

## 9.3 环境准备

建议使用 Python 3.11，并在仓库根目录创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
API_KEY=your-api-key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
```

不要提交 `.env`。示例会检查必要变量，缺少时直接报错，不会偷偷使用某个昂贵模型。

## 9.4 可运行代码

完整源码在 [`examples/09_langchain_quickstart.py`](../../examples/09_langchain_quickstart.py)。

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是耐心的 Python 助教。只回答学习问题，不编造资料。"),
        ("human", "请用不超过 120 字解释：{question}"),
    ]
)
chain = prompt | model | StrOutputParser()
print(chain.invoke({"question": "LCEL 中的竖线符号有什么作用？"}))
```

运行：

```bash
python examples/09_langchain_quickstart.py
```

输出由模型决定，大意应说明：LCEL 将多个可运行组件串联，前一步输出成为后一步输入。

## 9.5 从 Chain 走向工具

第6章里，工具由 JSON Schema 和 Python 函数组成。LangChain 的 `@tool` 装饰器会根据函数签名和文档字符串生成描述：

```python
from langchain_core.tools import tool

@tool
def lookup_course(chapter: int) -> str:
    """查询 1 到 14 章的课程标题。"""
    titles = {9: "LangChain", 10: "LangGraph", 12: "RAG"}
    if chapter not in range(1, 15):
        return "章节必须在 1 到 14 之间"
    return titles.get(chapter, "该章标题请查阅根 README")
```

装饰器解决的是“如何描述工具”，不是“工具是否安全”。参数范围、权限、超时和副作用仍然由你的代码负责。

## 9.6 记忆该放在哪里

旧教程常见 `ConversationBufferMemory`，但新项目更适合显式管理消息历史，或在 LangGraph 中使用状态和 checkpointer。这样做有两个好处：

1. 消息什么时候保存、裁剪、删除都能看见
2. 多用户场景可以用 `thread_id` 隔离会话，避免串线

不要把所有历史永久塞进提示词。长对话应结合窗口裁剪、摘要和用户可控的长期记忆。

## 9.7 踩坑记录

### 导入路径变化

LangChain 已拆成多个包。模型集成通常来自 `langchain_openai`，基础接口来自 `langchain_core`。遇到 `ModuleNotFoundError` 时，先核对示例的锁定版本和导入路径，不要盲目安装名字相近的包。

### 兼容端点不完全兼容

有些服务只兼容聊天接口，不兼容工具调用或结构化输出。先用最小聊天链验证，再逐项开启工具、流式和 JSON 模式。

### 模板变量没有传全

`prompt` 中写了 `{question}`，调用时就必须传 `{"question": ...}`。变量名不一致会在请求模型之前报错。

### 重试导致重复副作用

读取类调用可以重试，发邮件、扣款、写数据库不能无条件重试。为有副作用的工具使用幂等键，并在执行前人工确认。

## 9.8 安全边界

- API Key 只放环境变量或密钥服务，不写进代码、日志和异常文本
- 给网络请求设置超时和有限重试，限制输入与输出长度
- 模型生成的工具参数一律视为不可信数据
- 删除、支付、发送、发布等操作必须显示参数并等待确认
- 不把网页或检索文档中的指令当成系统指令，防止提示注入

## 9.9 痛点：组件一多，手写胶水代码会失控

第4—7章的手写代码帮助我们看清了 Agent 的本质，但当项目继续增长，会遇到一组很现实的问题：

1. 提示词散落在字符串拼接里，改一个变量名就可能运行时报错
2. 模型返回的是消息对象，而业务只想得到字符串或结构化数据
3. 同一条链既要单次调用，又要批量、流式、重试和追踪
4. 工具的 Python 签名与发给模型的 JSON Schema 要维护两份
5. 更换模型供应商时，超时、重试和消息格式需要重复适配

可以把手写流程类比成自己组装一条生产线：

```text
手写方式：
  原料 → 自制传送带 → 自制夹具 → 自制检测仪 → 成品

LangChain 方式：
  原料 → 标准 Runnable → 标准 Runnable → 标准 Runnable → 成品
                       统一 invoke / batch / stream
```

LangChain 的价值不是让模型更聪明，而是给组件提供统一插头。它减少的是工程胶水，不是业务判断。

> **使用原则**：先判断有没有“组合与替换”的痛点，再决定是否引入框架。不要为了使用 LangChain 而使用 LangChain。

---

## 9.10 原理图：Chain 与工具 Agent 有什么不同

普通 LCEL Chain 是一条确定的流水线：

```text
┌────────────┐   ┌────────────┐   ┌────────────┐
│ Prompt     │ → │ Chat Model │ → │ Parser     │
│ 填充变量    │   │ 生成消息    │   │ 转成字符串  │
└────────────┘   └────────────┘   └────────────┘
      question        AIMessage          str
```

工具 Agent 多了一个由模型决定的循环：

```text
                         ┌─────────────────────┐
                         │                     │
                         ▼                     │
用户 → 消息列表 → 绑定工具的模型 → 有 tool_calls？ ── 是 ─→ 校验并执行工具
                                  │                    │
                                  否                   │ ToolMessage
                                  │                    │
                                  ▼                    └─────────────┘
                               最终回答
```

这里有四个关键事实：

- `bind_tools` 只是把工具描述交给模型，并不执行函数
- `AIMessage.tool_calls` 是模型提出的调用请求
- 本地代码根据白名单找到工具并执行
- 每个结果都必须用相同 `tool_call_id` 包装成 `ToolMessage`

这与第4章手写 Function Calling 的循环完全相同，LangChain 只是统一了消息、工具和调用接口。

---

## 9.11 分步代码一：先构建最小 LCEL 链

### 第一步：创建模型

```python
def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=require_env("API_KEY"),
        base_url=os.getenv("BASE_URL") or None,
        model=require_env("MODEL"),
        temperature=0,
        timeout=30,
        max_retries=2,
    )
```

逐段解释：

- `temperature=0` 让教学问答更稳定，但不保证逐字相同
- `timeout=30` 防止网络异常让进程无限等待
- `max_retries=2` 只适合模型读取类请求，不能照搬到扣款等副作用操作
- `base_url=None` 时使用 SDK 默认地址，配置兼容服务时再覆盖

### 第二步：定义提示模板

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是耐心的 Python 助教。只回答学习问题；"
            "不知道时明确说不知道，不编造资料。",
        ),
        ("human", "请用不超过 120 字解释：{question}"),
    ]
)
```

`{question}` 是模板变量。LangChain 会在调用模型前检查变量是否存在，因此比手写字符串拼接更早暴露错误。

### 第三步：用 LCEL 连接

```python
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"question": "LCEL 是什么？"})
```

数据在管道中的形态依次是：

```text
dict
  ↓ ChatPromptTemplate
PromptValue / messages
  ↓ ChatOpenAI
AIMessage
  ↓ StrOutputParser
str
```

如果删除 `StrOutputParser()`，最终得到的是 `AIMessage`，可以读取 `content`、`usage_metadata` 等信息；保留解析器则更方便直接交给普通业务函数。

---

## 9.12 分步代码二：把 Python 函数变成工具

### 课程查询工具

```python
@tool
def lookup_course(chapter: int) -> str:
    """查询第4到第14章的课程标题；chapter 必须是整数章号。"""
    if chapter < 4 or chapter > 14:
        return json.dumps(
            {"ok": False, "error": "章号必须在 4 到 14 之间"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"ok": True, "chapter": chapter, "title": COURSES[chapter]},
        ensure_ascii=False,
    )
```

`@tool` 会读取三类信息：

1. 函数名成为工具名
2. 文档字符串成为工具用途说明
3. 类型注解生成参数 Schema

但装饰器不会替你做范围校验。模型仍可能传入 `chapter=100`，因此函数体必须把输入当成不可信数据。

### 为什么返回 JSON 字符串

工具结果最终会进入模型上下文。统一返回下面的结构更容易处理：

```json
{
  "ok": false,
  "error": "章号必须在 4 到 14 之间"
}
```

相比直接抛出异常，结构化错误能让模型解释失败原因并请求用户修正。真正的系统异常仍应记录到脱敏日志。

### 安全计算工具

不要这样写：

```python
def calculate(expression: str):
    return eval(expression)  # 危险：可能执行任意 Python 代码
```

完整示例使用 `ast.parse(expression, mode="eval")` 解析表达式，并只允许：

- 数字常量
- 加、减、乘、除、整除、取模和有限指数
- 正负号
- 括号形成的表达式树

名称、函数调用、属性访问、列表推导等节点都会被拒绝。白名单比“过滤几个危险字符串”可靠，因为攻击者很容易绕过字符串黑名单。

---

## 9.13 分步代码三：显式实现工具循环

### 绑定工具

```python
TOOLS = [lookup_course, safe_calculate]
model_with_tools = model.bind_tools(TOOLS)
```

绑定后，模型可以返回普通回答，也可以返回一个或多个工具调用。是否调用仍由模型判断。

### 初始化消息

```python
messages = [
    SystemMessage(
        content=(
            "你是课程助教。查询章名或计算时必须使用工具；"
            "不得虚构工具结果；工具返回错误时向用户解释。"
        )
    ),
    HumanMessage(content=question),
]
```

系统消息描述决策规则，工具描述解释单个工具的能力。两者职责不同：

- 系统消息回答“整个 Agent 应怎样工作”
- 工具描述回答“这个工具何时用、参数是什么”

### 进入有限循环

```python
for round_number in range(1, max_rounds + 1):
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        return str(response.content)

    for tool_call in response.tool_calls:
        result = execute_tool_call(tool_call)
        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )
        )
```

逐段解释：

1. 模型看到完整消息并决定下一步
2. 先把 `AIMessage` 放回历史，保留工具调用请求
3. 没有 `tool_calls` 就把当前内容视为最终回答
4. 有调用就按白名单执行工具
5. 用对应 ID 追加 `ToolMessage`
6. 下一轮模型读取结果并决定继续调用还是回答

达到最大轮数时应抛出“未完成”，而不是伪装成功。生产系统还要记录请求 ID、耗时、轮数和 token 使用量。

---

## 9.14 完整案例：课程助教工具 Agent

完整代码位于：

```text
examples/09_langchain_quickstart.py
```

它包含两种运行模式：

```bash
# 模式1：最小 LCEL 问答链
python examples/09_langchain_quickstart.py chain

# 模式2：工具 Agent
python examples/09_langchain_quickstart.py agent "第10章讲什么？再算 18*24"
```

案例的职责划分如下：

| 模块 | 职责 | 安全措施 |
|------|------|----------|
| `build_model` | 创建统一模型配置 | 超时、有限重试、密钥来自环境 |
| `build_chain` | 组装 Prompt→Model→Parser | 模板变量检查 |
| `lookup_course` | 查询内置课程目录 | 章号范围校验 |
| `safe_calculate` | 计算四则表达式 | AST 白名单、长度和结果限制 |
| `execute_tool_call` | 分发工具调用 | 工具名白名单、结果截断 |
| `run_tool_agent` | 驱动工具循环 | 输入限制、最大轮数 |

### 预期运行轨迹

不同模型输出文字会不同，但轨迹应类似：

```text
[第 1 轮] 模型请求 2 个工具
  - lookup_course({'chapter': 10})
    -> {"ok": true, "chapter": 10, "title": "LangGraph 状态机"}
  - safe_calculate({'expression': '18*24'})
    -> {"ok": true, "expression": "18*24", "result": 432}

问题：第10章讲什么？再帮我计算 18*24。
最终回答：第10章是“LangGraph 状态机”，18×24=432。
```

模型可能并行提出两个调用，也可能分两轮调用。只要最终答案基于真实工具结果，两种路径都可以接受。

---

## 9.15 运行与验收

### 安装与配置

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
cp .env.example .env
```

配置 `.env` 后运行前面的两个命令。

### 验收清单

- [ ] `chain` 模式能返回关于 LCEL 的简短解释
- [ ] `agent` 模式确实打印 `lookup_course` 的调用轨迹
- [ ] `18*24` 的工具结果为 `432`
- [ ] 输入“第100章是什么”时工具返回范围错误，程序不崩溃
- [ ] 输入 `__import__('os').system('id')` 时计算工具拒绝执行
- [ ] 未配置 `API_KEY` 时给出明确错误，且不打印其他密钥
- [ ] Agent 超过最大轮数时报告未完成，不返回假成功

### 无 API 的静态自检

即使没有模型密钥，也可以先检查语法：

```bash
python -m py_compile examples/09_langchain_quickstart.py
```

还可以直接在 Python 中测试纯工具：

```python
from examples import 09_langchain_quickstart  # 文件名以数字开头，不能这样导入
```

上面的写法故意展示了一个常见错误：数字开头的模块不能用普通 `import` 语句导入。测试时可以运行脚本，或使用 `importlib.util.spec_from_file_location` 加载。

---

## 9.16 更多踩坑记录

### 坑1：工具函数有类型注解但没有文档字符串

**现象**：模型知道参数类型，却不知道什么时候使用工具。

**解决**：为工具写清楚用途、边界和参数含义。文档字符串是给模型看的接口说明，不只是给人看的注释。

### 坑2：漏加模型的 AIMessage

**现象**：已经得到工具调用，却只追加了 `ToolMessage`，服务端报消息顺序错误。

**解决**：先 `messages.append(response)`，再追加与每个调用对应的工具结果。

### 坑3：并行工具调用只处理了第一个

**现象**：模型一次返回两个 `tool_calls`，代码只执行索引 `0`，下一轮提示缺少工具结果。

**解决**：遍历所有调用，并为每个 ID 添加且只添加一个 `ToolMessage`。

### 坑4：把 ToolException 当作最终答案

**现象**：工具失败后循环直接中断，用户只看到堆栈。

**解决**：可预期的参数错误转成结构化工具结果；不可预期错误记录日志，并返回脱敏描述。

### 坑5：模型端点支持聊天但不支持工具

**现象**：普通 Chain 正常，Agent 从不调用工具或返回协议错误。

**解决**：确认所选模型和兼容端点支持 tool calling；先用一个最小工具做能力探测。

### 坑6：追踪日志泄露数据

**现象**：为了排错记录完整消息，用户隐私和工具结果进入第三方追踪平台。

**解决**：按环境开启追踪，发送前脱敏，限制保留时间，并允许用户删除。

---

## 9.17 动手练习

### 练习1：增加章节范围查询（简单）

实现 `list_courses(start: int, end: int)`，返回范围内所有章名。要求：

1. `start <= end`
2. 只能查询第4—14章
3. 最多一次返回6章
4. 错误使用统一 JSON 格式

### 练习2：给工具增加 Pydantic 参数模型（中等）

为计算工具定义显式输入模型，增加表达式最短和最长长度，并观察生成的 Schema。

### 练习3：增加批量 Chain 调用（中等）

使用 `chain.batch` 同时解释三个概念。设置合理并发数，比较串行与批量的耗时。

### 练习4：增加人工确认（困难）

添加一个模拟“写入学习计划”的工具。模型提出写入后，不要立即执行；先显示目标内容，只有用户输入确认才调用。

### 练习5：编写离线单元测试（困难）

为 `evaluate_expression` 测试：

- 正常四则运算
- 除零
- 超大指数
- 函数调用
- 属性访问
- 超大结果

测试不得访问真实模型 API。

## 9.18 思考题与答案

### 题1：LCEL 的 `|` 和普通函数嵌套有什么区别

**答案**：业务效果都可以用函数实现。LCEL 的价值是统一 `invoke`、`batch`、`stream` 等接口，并让回调和追踪更容易贯穿整条链。

### 题2：为什么简单模型调用不一定需要 LangChain

**答案**：框架会增加依赖、版本迁移和抽象成本。只有一个请求时，官方 SDK 更短、更透明，也更容易排错。

### 题3：`@tool` 是否能让函数自动变安全

**答案**：不能。它只帮助模型理解名称、描述和参数结构。权限校验、路径限制、参数白名单、超时和人工确认仍需开发者实现。

### 题4：为什么不建议把无限消息历史当作记忆

**答案**：它会增加费用和延迟，最终超出上下文窗口，还可能保留用户不希望长期保存的信息。记忆需要生命周期、隔离和删除机制。

### 题5：模型一次请求多个工具时，为什么不能只返回一个结果

**答案**：每个工具调用都有独立的 `tool_call_id`。缺少任意一个对应结果都会破坏消息协议，模型也无法获得完整观察。

### 题6：为什么工具错误更适合返回结构化结果

**答案**：可预期的业务错误不是程序崩溃。结构化结果让模型能区分成功与失败、解释原因并引导用户修正，同时便于测试和日志统计。

## 9.19 小结

LangChain 把提示词、模型和解析器变成可组合组件。先从 LCEL 小链开始，在确实需要工具、追踪或多模型切换时逐步增加抽象。下一章会用 LangGraph 把“循环和分支”画成一张可执行的状态图。

---

> 上一章：[第8章：Agent 行为工程](../08-agent-behavior-engineering/README.md)  
> 下一章：[第10章：LangGraph 状态机](../10-langgraph/README.md)
