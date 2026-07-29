# 第9章：LangChain 快速上手

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 9.1 为什么要学 LangChain？

走到这里，你已经手写过 ReAct Agent、记忆系统、工具系统、规划与反思机制。现在你再去看各种框架，就不会有"这都是什么黑魔法"的感觉了。

LangChain 的价值，不是让你"不用思考"，而是帮你把那些重复、标准化、容易写错的部分抽象出来。比如：

- Prompt 模板怎么拼接
- LLM 调用怎么串起来
- 工具怎么注册
- Agent 循环怎么复用
- 输出怎么结构化解析

换句话说，**前面几章你学的是原理，这一章开始你学的是工程效率。**

### 手写 Agent vs LangChain

| 对比维度 | 手写 Agent | LangChain |
|------|------|------|
| 学习价值 | 非常高，最能理解底层 | 高，能快速进入工程实践 |
| 开发速度 | 慢，很多样板代码 | 快，内置大量组件 |
| 可控性 | 最高 | 较高，但要遵循框架接口 |
| 调试难度 | 逻辑清晰，但代码多 | 代码少，但抽象层更多 |
| 适用阶段 | 学原理、做实验 | 做真实项目、快速迭代 |

你可以把 LangChain 理解为一盒乐高积木。前面你已经学会怎么手工造积木了，现在开始学怎么高效拼装。

---

## 9.2 LangChain 的核心心智模型

初学者最容易被 LangChain 各种名词绕晕。其实你只要记住下面这张表，就能快速抓住重点：

| 组件 | 作用 | 你前面章节里的对应概念 |
|------|------|------|
| `ChatModel` | 调用大模型 | LLM 大脑 |
| `PromptTemplate` | 管理提示词模板 | System Prompt / 用户 Prompt |
| `OutputParser` | 解析模型输出 | JSON 解析 / 结构化输出 |
| `Tool` | 定义工具 | Function Calling 工具 |
| `Agent` | 决定何时调用工具 | ReAct 循环中的决策者 |
| `AgentExecutor` | 驱动 Agent 执行 | 你手写的主循环 |
| `Runnable` | 用统一接口串联组件 | 你手写的流程编排 |

### LangChain 最重要的思想：万物皆可串联

在 LangChain 里，很多组件都可以像管道一样串起来：

```python
prompt | llm | output_parser
```

这行代码的意思非常直白：

1. 先把输入套进提示词模板
2. 再交给模型
3. 再把输出交给解析器

如果你还记得前面手写的代码，你会发现这其实就是把：

```python
formatted_prompt = build_prompt(user_input)
raw_output = call_llm(formatted_prompt)
result = parse_output(raw_output)
```

压缩成了一条统一的链路。

---

## 9.3 第一个 LangChain 程序

我们先从最简单的开始：用 LangChain 调一次模型，感受一下它的基本用法。

### 安装依赖

```bash
pip install langchain langchain-openai python-dotenv
```

### 最小可运行示例

```python
# hello_langchain.py
"""
第一个 LangChain 程序
作者：青松与桑叶
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    # 创建模型对象
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.3,
    )

    # 直接调用
    response = llm.invoke("请用一句通俗的话解释什么是 AI Agent")

    print("模型回复：")
    print(response.content)


if __name__ == "__main__":
    main()
```

### 运行

```bash
python hello_langchain.py
```

### 你会看到什么？

和你直接调用 OpenAI SDK 相比，这里最大的变化是：我们不再直接操作 HTTP 或底层 response，而是统一通过 `ChatOpenAI` 这个对象去调用模型。

这一步虽然简单，但非常重要。因为后面无论是 Prompt 模板、工具调用、Agent 执行器，最终都要围绕这个模型对象来展开。

---

## 9.4 Prompt Template：让提示词可复用

如果每次都手写大段提示词，很快你就会遇到两个问题：

1. 相同的提示词到处复制粘贴
2. 参数替换很容易出错

LangChain 的 `ChatPromptTemplate` 就是专门解决这个问题的。

### 示例：构建一个可复用的提示词模板

```python
# prompt_template_demo.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的 AI 老师，请用通俗、简洁的方式回答问题。"),
    ("human", "请解释这个概念：{concept}"),
])

chain = prompt | llm

result = chain.invoke({"concept": "RAG"})
print(result.content)
```

### 这个写法有什么好处？

- `system` 和 `human` 消息结构非常清晰
- `{concept}` 是显式参数，不容易拼错
- `prompt | llm` 这条链可以在别处复用

如果以后你想统一修改风格，比如把"通俗、简洁"改成"专业、详细"，只要改模板就可以了，不用满项目搜字符串。

---

## 9.5 Output Parser：让输出更稳定

大模型最让人头疼的一点，就是它明明答得很好，但格式经常不稳定。昨天返回 JSON，今天可能多说两句废话，程序一解析就报错。

LangChain 提供了解析器，帮助我们把模型输出变成更稳定的结构。

### 字符串解析器示例

```python
# output_parser_demo.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_template(
    "请用三条 bullet，总结 {topic} 的核心要点。"
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "LangChain"})
print(result)
```

### 为什么要显式加解析器？

因为 `llm.invoke()` 返回的是一个消息对象，而不是纯字符串。加上 `StrOutputParser()` 后，链的输出就变成了你最熟悉的字符串，更适合后续处理。

这看起来像一个小细节，但在真实项目里，**清晰的输入输出边界** 会大幅降低维护成本。

---

## 9.6 用 LangChain 创建一个带工具的 Agent

前面我们手写过多工具 Agent，现在用 LangChain 来做同样的事情，你会直观感受到：框架帮你省掉了多少样板代码。

### 第一步：定义工具

```python
# tools_demo.py
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    """
    mock_data = {
        "北京": "北京今天晴天，35度，湿度40%。",
        "上海": "上海今天多云，28度，湿度65%。",
        "广州": "广州今天阵雨，32度，湿度80%。",
    }
    return mock_data.get(city, f"暂时没有 {city} 的天气数据。")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式，只支持基本四则运算。
    """
    allowed = set("0123456789+-*/().% ")
    if not all(char in allowed for char in expression):
        return "表达式中包含非法字符。"

    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算失败：{e}"
```

这里的 `@tool` 装饰器很关键。它会把普通 Python 函数包装成 LangChain 可识别的工具对象。

### 第二步：创建 Agent

```python
# langchain_agent_demo.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools_demo import get_weather, calculate

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0,
)

tools = [get_weather, calculate]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个实用型智能助手。能查天气，也能做数学计算。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "北京今天多少度？顺便帮我算一下 (35 - 28) * 2"
})

print(result["output"])
```

### 这段代码背后发生了什么？

你可以把它理解成：

1. `create_tool_calling_agent()` 创建了一个具备工具调用能力的 Agent
2. `AgentExecutor` 帮你跑完整个循环
3. `verbose=True` 会打印中间过程，方便调试

也就是说，LangChain 帮你把前面手写的这些逻辑都封装好了：

- 判断要不要调用工具
- 生成工具参数
- 执行工具
- 把结果回填给模型
- 最终输出自然语言答案

---

## 9.7 把 Chain 和 Agent 分开理解

这是初学者特别容易混淆的一点。

### Chain 是什么？

`Chain` 更像是一条固定流水线：

```python
用户输入 -> Prompt 模板 -> 模型 -> 输出解析
```

它适合：

- 文案生成
- 摘要提取
- 分类打标
- 固定流程的数据清洗

### Agent 是什么？

`Agent` 更像一个会自己做决策的执行者：

```python
用户输入 -> 模型判断 -> 调哪个工具 -> 看结果 -> 再判断下一步
```

它适合：

- 信息查询
- 多工具组合任务
- 需要动态决策的复杂流程

### 一句话区分

- **Chain**：流程提前写死
- **Agent**：流程运行时决定

这个区分非常重要。因为很多本来用 Chain 就够的任务，大家硬要上 Agent，结果系统变慢、成本变高、调试变难。

---

## 9.8 一个很实用的建议：先会手写，再用 LangChain

到这里你应该已经发现了：如果你没有前面几章的基础，很多 LangChain 的接口会显得非常抽象。

比如：

- 为什么要有 `agent_scratchpad`？
- 为什么工具要注册成特定格式？
- 为什么 `AgentExecutor` 能自动帮你循环？

但你现在理解这些就很轻松，因为你已经亲手实现过一遍。

### 学 LangChain 的正确姿势

1. 先理解原理
2. 再用框架提效
3. 出问题时能回到底层排查

如果你一开始只会"抄框架代码"，那一旦 Agent 不调用工具、Prompt 不生效、解析器报错，你就很难知道问题到底出在哪。

---

## 9.9 动手练习

### 练习1：做一个"术语解释器"（难度：简单）

要求：

1. 使用 `ChatPromptTemplate`
2. 用户输入一个技术名词
3. 输出三段内容：
   - 一句话解释
   - 通俗类比
   - 常见误区

提示：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位擅长解释技术概念的老师。"),
    ("human", "请解释 {term}，输出包含：一句话解释、通俗类比、常见误区。"),
])
```

### 练习2：做一个"天气 + 计算" Agent（难度：中等）

要求：

1. 使用两个工具：`get_weather` 和 `calculate`
2. 支持用户提出复合问题
3. 打开 `verbose=True`，观察 Agent 的中间过程

示例输入：

```text
北京今天 35 度，如果我想在 28 度的上海和北京之间做温差对比，温差是多少？
```

### 练习3：给 Agent 增加一个文件保存工具（难度：中等）

让 Agent 支持把结果写入本地文件。

要求：

1. 自定义 `write_file(path, content)` 工具
2. 让 Agent 能完成"先查信息，再保存结果"的任务
3. 注意文件路径的安全性，避免任意写入敏感目录

---

## 9.10 小结

本章我们正式进入框架世界，并完成了 LangChain 的第一轮实战。

| 知识点 | 核心内容 |
|------|------|
| `ChatOpenAI` | 用统一接口调用兼容 OpenAI 格式的大模型 |
| `ChatPromptTemplate` | 让提示词可复用、可维护 |
| `OutputParser` | 让模型输出更稳定 |
| `@tool` | 把 Python 函数注册成工具 |
| `AgentExecutor` | 帮你跑完整个 Agent 执行循环 |

核心要点：

1. **LangChain 不是魔法，只是把常见模式封装起来了**
2. **Chain 适合固定流程，Agent 适合动态决策**
3. **理解底层原理后，再用框架会轻松很多**
4. **框架能提效，但不替你思考系统设计**

---

## 下一章预告

这一章你学会了用 LangChain 把常见组件拼起来，但 LangChain 更像"积木盒"，并不强制你怎么组织复杂流程。

如果你的系统开始出现下面这些需求：

- 一个任务要分多个阶段
- 每个阶段要维护状态
- 某些步骤失败后要回退
- 中间某一步需要人工确认

那就该上 **LangGraph** 了。

在 **第10章：LangGraph 状态机** 中，我们会学习：

- 什么是图结构工作流
- 如何管理 Agent 的状态
- 如何做条件分支与回路
- 如何接入人工审批节点

---

> 上一篇：[第8章：Agent 行为工程](../08-agent-behavior-engineering/README.md) | 下一篇：[第10章：LangGraph 状态机](../10-langgraph/README.md)
