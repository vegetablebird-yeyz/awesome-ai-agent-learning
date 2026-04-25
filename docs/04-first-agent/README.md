# 第4章：手写第一个 Agent

> **作者**：青松与桑叶
> **难度**：入门
> **前置知识**：Python 基础、了解什么是 LLM
> **本章目标**：不依赖任何框架，用纯 Python 从零构建一个能思考、调用工具的 Agent

---

## 4.1 本章目标

学完本章，你将：

1. **理解 Agent 的核心原理** —— Agent 不是魔法，就是一个"思考 → 行动 → 观察"的循环
2. **掌握 ReAct 模式** —— Reasoning + Acting，让 LLM 像人一样思考问题
3. **从零手写一个完整 Agent** —— 不用 LangChain、不用 CrewAI，纯 Python 实现
4. **学会 Function Calling** —— 让 LLM 调用外部工具

> **一句话总结**：Agent = LLM + 工具 + 循环。就这么简单。

---

## 4.2 环境准备

### 安装依赖

```bash
pip install openai python-dotenv
```

### 配置环境变量

创建 `.env` 文件：

```bash
# .env 文件内容
API_KEY=your-api-key-here
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
```

> **说明**：`BASE_URL` 和 `MODEL` 支持任何 OpenAI 兼容的 API，比如 DeepSeek、智谱、通义千问等，只需要把对应的地址和模型名填进去即可。

---

## 4.3 第一步：让 LLM 动起来

在构建 Agent 之前，我们先来回顾一下最基本的 LLM 调用。这是整个 Agent 的地基。

### 完整代码

```python
# step1_basic_llm.py
"""
第一步：最基础的 LLM 调用
这是构建 Agent 的地基，务必理解每一行
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 创建 OpenAI 客户端（兼容任何 OpenAI 格式的 API）
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# 调用 LLM
response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ],
)

# 打印回复
print(response.choices[0].message.content)
```

### 逐行解释

| 代码 | 说明 |
|------|------|
| `load_dotenv()` | 从 `.env` 文件读取环境变量，这样 API Key 不会硬编码在代码里 |
| `OpenAI(api_key=..., base_url=...)` | 创建客户端，`base_url` 让你可以切换不同的 API 提供商 |
| `client.chat.completions.create(...)` | 发送对话请求，`messages` 是对话历史 |
| `response.choices[0].message.content` | 取出 LLM 的回复文本 |

### 运行方法

```bash
python step1_basic_llm.py
```

### 预期输出

```
你好！我是一个 AI 助手，很高兴为你服务。
```

> **关键理解**：LLM 本质上就是一个函数 —— 输入是一段文本（messages），输出也是一段文本。Agent 就是在这个基础上加了"工具"和"循环"。

---

## 4.4 第二步：给 Agent 装上工具

LLM 只会"说话"，不会"做事"。我们要通过 **Function Calling** 让它能够调用外部工具。

### 什么是 Function Calling？

Function Calling 是 OpenAI 提供的一种机制，让 LLM 可以"告诉"你它想调用哪个函数，以及需要什么参数。注意：**LLM 本身不执行函数**，它只是告诉你应该调用哪个函数，然后由你的代码来实际执行。

流程如下：

```
用户提问 → LLM 思考 → LLM 返回"我想调用 xxx 函数，参数是 yyy"
→ 你的代码执行函数 → 把结果返回给 LLM → LLM 生成最终回答
```

### 完整代码

```python
# step2_function_calling.py
"""
第二步：给 LLM 装上工具
演示 Function Calling 的基本用法
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ============================================
# 1. 定义工具（用 JSON Schema 描述）
# ============================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：32 * 15",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# ============================================
# 2. 定义工具的实际执行逻辑
# ============================================
def get_weather(city: str) -> str:
    """模拟获取天气信息（实际项目中接入真实天气 API）"""
    # 这里用模拟数据，实际可以调用和风天气、OpenWeather 等 API
    weather_data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
    }
    data = weather_data.get(city, {"temperature": 25, "weather": "未知", "humidity": 50})
    return f"{city}的天气：{data['weather']}，温度：{data['temperature']}度，湿度：{data['humidity']}%"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全计算：只允许数字和基本运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# 工具名称到函数的映射
tool_map = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# ============================================
# 3. 发送请求，让 LLM 决定是否调用工具
# ============================================
user_message = "北京天气怎么样？"

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {"role": "system", "content": "你是一个有用的助手，可以查询天气和进行数学计算。"},
        {"role": "user", "content": user_message},
    ],
    tools=tools,  # 把工具定义传给 LLM
    tool_choice="auto",  # 让 LLM 自己决定是否调用工具
)

# ============================================
# 4. 处理 LLM 的响应
# ============================================
message = response.choices[0].message

# 检查 LLM 是否想要调用工具
if message.tool_calls:
    print(f"用户：{user_message}")
    print(f"LLM 想调用工具：{message.tool_calls[0].function.name}\n")

    for tool_call in message.tool_calls:
        # 获取函数名和参数
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"  函数名：{function_name}")
        print(f"  参数：{function_args}")

        # 执行函数
        result = tool_map[function_name](**function_args)
        print(f"  执行结果：{result}\n")

        # 把工具执行结果返回给 LLM，让它生成最终回答
        final_response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": user_message},
                message,  # LLM 之前要求调用工具的消息
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                },
            ],
        )
        print(f"最终回答：{final_response.choices[0].message.content}")
else:
    # LLM 没有调用工具，直接回答
    print(f"回答：{message.content}")
```

### 逐行解释

| 代码 | 说明 |
|------|------|
| `tools = [...]` | 用 JSON Schema 格式定义工具，告诉 LLM 有哪些工具可用、每个工具需要什么参数 |
| `"description"` | **非常关键！** 这是 LLM 理解工具用途的唯一途径，描述越清晰，LLM 越不容易选错工具 |
| `"required"` | 标记哪些参数是必须的 |
| `tool_map = {...}` | 把工具名映射到实际的 Python 函数，方便后续调用 |
| `message.tool_calls` | 如果 LLM 决定调用工具，这里会有内容；否则为 None |
| `json.loads(tool_call.function.arguments)` | LLM 返回的参数是 JSON 字符串，需要解析成字典 |
| `role: "tool"` | 把工具执行结果以 `tool` 角色发回给 LLM，`tool_call_id` 用于关联 |

### 运行方法

```bash
python step2_function_calling.py
```

### 预期输出

```
用户：北京天气怎么样？
LLM 想调用工具：get_weather

  函数名：get_weather
  参数：{'city': '北京'}
  执行结果：北京的天气：晴天，温度：35度，湿度：40%

最终回答：北京今天天气晴朗，温度35度，湿度40%。天气比较炎热，建议注意防晒和补水！
```

> **关键理解**：Function Calling 的本质是 LLM 输出结构化的"调用指令"，你的代码负责执行并把结果喂回去。LLM 永远不会自己执行代码。

---

## 4.5 第三步：实现 ReAct 循环（核心！）

这是本章最重要的部分。我们要把前面的知识组合起来，实现一个完整的 **ReAct（Reasoning + Acting）循环**。

### 什么是 ReAct？

ReAct 是一种让 Agent 像人一样解决问题的模式：

```
人类解决问题的过程：
  思考（Thought）："我需要知道北京天气" → 行动（Action）：查天气 → 观察（Observation）：35度晴天
  思考（Thought）："温度超过30度了，需要计算" → 行动（Action）：算32*15 → 观察（Observation）：480
  思考（Thought）："我已经有足够信息了" → 回答（Answer）：...

Agent 的 ReAct 循环：
  LLM 思考 → 决定调用工具 → 执行工具 → 把结果喂回 LLM → 继续思考 → ...
  直到 LLM 认为可以给出最终回答为止
```

**核心思想**：让 LLM 在一个循环中不断"思考 → 行动 → 观察"，直到任务完成。

### 完整代码

```python
# step3_react_loop.py
"""
第三步：实现 ReAct 循环
这是 Agent 的核心！理解这个，你就理解了所有 Agent 框架的本质。
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ============================================
# 1. 定义工具
# ============================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：32 * 15",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================
# 2. 工具执行函数
# ============================================
def get_weather(city: str) -> str:
    """获取天气信息（模拟数据）"""
    weather_data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
    }
    data = weather_data.get(city, {"temperature": 25, "weather": "未知", "humidity": 50})
    return json.dumps(data, ensure_ascii=False)


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "表达式包含不允许的字符"})
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


tool_map = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============================================
# 3. ReAct 循环（核心逻辑）
# ============================================
def run_agent(user_message: str, max_rounds: int = 5) -> str:
    """
    运行 Agent 的 ReAct 循环

    参数：
        user_message: 用户的问题
        max_rounds: 最大循环轮数（防止死循环）

    返回：
        Agent 的最终回答
    """
    # 维护对话历史
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，可以查询天气和进行数学计算。\n"
                "请根据用户的问题，选择合适的工具来获取信息。\n"
                "如果需要多步操作，请一步步来。\n"
                "获取到足够信息后，给出最终回答。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"{'='*60}")
    print(f"用户：{user_message}")
    print(f"{'='*60}\n")

    # ReAct 循环
    for round_num in range(1, max_rounds + 1):
        print(f"--- 第 {round_num} 轮 ---")

        # 调用 LLM
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # 情况1：LLM 没有调用工具 → 任务完成
        if not message.tool_calls:
            print(f"Agent 回答：{message.content}\n")
            return message.content

        # 情况2：LLM 想调用工具 → 执行工具并继续循环
        # 把 LLM 的消息加入历史
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  思考：需要调用 {function_name}，参数：{function_args}")

            # 执行工具
            result = tool_map[function_name](**function_args)
            print(f"  观察：{result}")

            # 把工具结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        print()  # 空行分隔

    return "达到最大循环轮数，Agent 未能完成任务。"


# ============================================
# 4. 运行测试
# ============================================
if __name__ == "__main__":
    # 测试1：简单问题（只需要一个工具）
    print("\n【测试1：简单问题】")
    run_agent("北京天气怎么样？")

    # 测试2：复杂问题（需要多个工具，多步推理）
    print("\n【测试2：复杂问题】")
    run_agent("北京天气怎么样？如果温度超过30度，帮我算一下32*15")

    # 测试3：不需要工具的问题
    print("\n【测试3：不需要工具】")
    run_agent("你好，你叫什么名字？")
```

### ReAct 循环详解

让我们用一个流程图来理解这个循环：

```
                    ┌─────────────────┐
                    │   用户提问       │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  调用 LLM       │◄──────────┐
                    └────────┬────────┘           │
                             ▼                    │
                    ┌─────────────────┐           │
                    │ LLM 是否要调用  │           │
                    │    工具？        │           │
                    └────────┬────────┘           │
                      ┌──────┴──────┐             │
                      │ 是          │ 否           │
                      ▼             ▼             │
               ┌──────────┐  ┌──────────┐        │
               │ 执行工具  │  │ 返回最终  │        │
               │ 获取结果  │  │ 回答      │        │
               └────┬─────┘  └──────────┘        │
                    │                            │
                    │ 把结果加入对话历史           │
                    └────────────────────────────┘
```

### 运行方法

```bash
python step3_react_loop.py
```

### 预期输出

```
【测试1：简单问题】
============================================================
用户：北京天气怎么样？
============================================================

--- 第 1 轮 ---
  思考：需要调用 get_weather，参数：{'city': '北京'}
  观察：{"temperature": 35, "weather": "晴天", "humidity": 40}

--- 第 2 轮 ---
Agent 回答：北京今天天气晴朗，温度35度，湿度40%。天气比较炎热，建议注意防晒和补水！

【测试2：复杂问题】
============================================================
用户：北京天气怎么样？如果温度超过30度，帮我算一下32*15
============================================================

--- 第 1 轮 ---
  思考：需要调用 get_weather，参数：{'city': '北京'}
  观察：{"temperature": 35, "weather": "晴天", "humidity": 40}

--- 第 2 轮 ---
  思考：需要调用 calculate，参数：{'expression': '32*15'}
  观察：{"expression": "32*15", "result": 480}

--- 第 3 轮 ---
Agent 回答：北京今天天气晴朗，温度35度，湿度40%。由于温度超过了30度，32*15的计算结果是480。

【测试3：不需要工具】
============================================================
用户：你好，你叫什么名字？
============================================================

--- 第 1 轮 ---
Agent 回答：你好！我是一个AI助手，你可以叫我小助手。有什么我可以帮你的吗？
```

> **关键理解**：ReAct 循环的本质就是一个 `while` 循环，每次循环让 LLM 决定下一步做什么。如果 LLM 不再调用工具，就说明它认为任务完成了。就这么简单！

---

## 4.6 完整代码

下面是整合了所有步骤的完整 Agent 代码，约 100 行，可以直接复制使用：

```python
# agent.py
"""
完整的 ReAct Agent —— 从零手写，无任何框架依赖
作者：青松与桑叶
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ==================== 工具定义 ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息（温度、天气状况、湿度）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
]

# ==================== 工具实现 ====================

def get_weather(city: str) -> str:
    """获取天气（模拟数据，实际可接入真实 API）"""
    data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
        "深圳": {"temperature": 30, "weather": "阴天", "humidity": 75},
    }
    result = data.get(city, {"temperature": 25, "weather": "未知", "humidity": 50})
    return json.dumps(result, ensure_ascii=False)


def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "不允许的字符"})
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_MAP = {"get_weather": get_weather, "calculate": calculate}

# ==================== Agent 核心 ====================

SYSTEM_PROMPT = (
    "你是一个智能助手，可以查询天气和进行数学计算。\n"
    "请根据用户的问题，合理选择工具，一步步解决问题。\n"
    "获取到足够信息后，用自然语言给出最终回答。"
)


def run_agent(user_message: str, max_rounds: int = 5) -> str:
    """运行 Agent"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for i in range(max_rounds):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [调用工具] {name}({args})")
            result = TOOL_MAP[name](**args)
            print(f"  [工具结果] {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Agent 达到最大循环次数，未能完成任务。"


# ==================== 交互入口 ====================

if __name__ == "__main__":
    print("Agent 已启动！输入 'quit' 退出\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        print()
        answer = run_agent(user_input)
        print(f"\nAgent：{answer}\n")
```

---

## 4.7 运行演示

### 启动 Agent

```bash
python agent.py
```

### 交互示例

```
Agent 已启动！输入 'quit' 退出

你：北京天气怎么样？如果温度超过30度，帮我算一下32*15

  [调用工具] get_weather({'city': '北京'})
  [工具结果] {"temperature": 35, "weather": "晴天", "humidity": 40}
  [调用工具] calculate({'expression': '32*15'})
  [工具结果] {"result": 480}

Agent：北京今天天气晴朗，温度35度，湿度40%。由于温度超过了30度，按照你的要求，32×15的计算结果是480。

你：上海呢？

  [调用工具] get_weather({'city': '上海'})
  [工具结果] {"temperature": 28, "weather": "多云", "humidity": 65}

Agent：上海今天多云，温度28度，湿度65%。温度没有超过30度，所以不需要进行额外计算。天气比较舒适！

你：quit
再见！
```

---

## 4.8 踩坑记录

### 坑1：API Key 配置错误

**错误信息**：`AuthenticationError: Incorrect API key provided`

**解决方法**：
1. 检查 `.env` 文件是否存在
2. 检查 `API_KEY` 是否正确
3. 检查 `.env` 文件路径是否正确（`load_dotenv()` 默认在当前目录找 `.env`）

### 坑2：BASE_URL 格式错误

**错误信息**：`ConnectionError` 或 `404 Not Found`

**解决方法**：
1. `BASE_URL` 不要以 `/` 结尾
2. 确认 URL 包含 `/v1` 路径（大多数兼容 API 需要）
3. 常见格式：
   - OpenAI：`https://api.openai.com/v1`
   - DeepSeek：`https://api.deepseek.com/v1`
   - 智谱：`https://open.bigmodel.cn/api/paas/v4`

### 坑3：模型不支持 Function Calling

**错误信息**：模型返回的 `tool_calls` 始终为 None

**解决方法**：
1. 确认模型支持 Function Calling（GPT-4、GPT-3.5-turbo、DeepSeek-V3 等都支持）
2. 某些小模型或旧版本不支持，尝试换一个模型
3. 检查 `tools` 参数的格式是否正确

### 坑4：Agent 死循环

**现象**：Agent 一直在调用工具，不停下来

**解决方法**：
1. 设置 `max_rounds` 限制最大循环次数（代码中已设置）
2. 优化系统提示词，明确告诉 LLM 何时应该给出最终回答
3. 检查工具返回的数据格式是否正确

### 坑5：工具参数解析失败

**错误信息**：`json.loads` 报错

**解决方法**：
1. LLM 返回的参数可能不是合法 JSON，加 `try-except` 处理
2. 在工具描述中给出参数示例，帮助 LLM 理解格式

---

## 4.9 动手练习

### 练习1：添加搜索工具（难度：简单）

给 Agent 添加一个 `search` 工具，让它能搜索信息：

```python
# 提示：定义一个新的工具
{
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索互联网上的信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
}

# 然后实现 search 函数，可以先用模拟数据
def search(query: str) -> str:
    # 模拟搜索结果
    return f"关于'{query}'的搜索结果：..."
```

### 练习2：优化系统提示词（难度：简单）

修改 `SYSTEM_PROMPT`，让 Agent：
1. 在回答时先总结用户的问题
2. 分步骤展示推理过程
3. 在回答末尾给出建议

### 练习3：添加日志记录（难度：中等）

给 Agent 添加日志功能，记录每一轮的：
1. LLM 的完整响应
2. 工具调用的参数和结果
3. 每轮的耗时

---

## 4.10 小结

恭喜你！你已经从零手写了一个完整的 Agent。让我们回顾一下核心知识：

| 概念 | 一句话解释 |
|------|-----------|
| **Agent** | LLM + 工具 + 循环 |
| **Function Calling** | LLM 输出结构化的工具调用指令，代码负责执行 |
| **ReAct 循环** | 思考 → 行动 → 观察 → 思考 → ... 直到任务完成 |
| **工具定义** | 用 JSON Schema 告诉 LLM 有什么工具、需要什么参数 |
| **max_rounds** | 防止 Agent 死循环的安全机制 |

### Agent 的本质

```
┌──────────────────────────────────────────┐
│                                          │
│   用户 ──→ LLM ──→ 调用工具 ──→ LLM ──→ 回答   │
│              ↑              │            │
│              └──────────────┘            │
│              （循环直到完成）              │
│                                          │
└──────────────────────────────────────────┘
```

所有花哨的 Agent 框架（LangChain、CrewAI、AutoGen...），底层都是这个逻辑。理解了这个，你就掌握了 Agent 的本质。

### 下一章预告

**第5章：让 Agent 拥有记忆**

目前的 Agent 有一个致命缺陷：它没有记忆。每次对话结束后，它就忘了之前聊了什么。下一章我们将学习如何给 Agent 加上短期记忆和长期记忆，让它成为一个真正"认识你"的助手。

---

> **作者**：青松与桑叶
> **下一章**：[第5章：让 Agent 拥有记忆](../05-agent-memory/README.md)
> **上一章**：[第3章：LLM 入门](../03-llm-basics/README.md)
