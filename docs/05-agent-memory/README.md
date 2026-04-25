# 第5章：让 Agent 拥有记忆

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 5.1 为什么需要记忆

在第4章中，我们手写了一个 ReAct Agent。它很酷，但有一个致命的缺陷 —— **它没有记忆**。

来看一个对话：

```
你：你好，我叫小明，我是一名 Python 开发者。

Agent：你好小明！很高兴认识你，Python 是一门很棒的语言。

你：我叫什么名字？

Agent：抱歉，我不知道你的名字。
```

发现问题了吗？Agent 刚刚才说过"你好小明"，下一秒就忘了。这是因为我们的 Agent 在每次处理用户消息时，只传入了当前这一条消息，没有把之前的对话历史一起传给 LLM。

再来看另一个场景：

```
# 第一次对话（周一）
你：帮我查一下北京天气。

Agent：北京今天晴，25度。

# 第二次对话（周三）
你：上次你帮我查的北京天气，现在变了吗？

Agent：抱歉，我没有之前的查询记录。
```

这就是没有记忆的 Agent —— 每次对话都像"初次见面"。

**一个有记忆的 Agent 应该是这样的：**

```
你：你好，我叫小明，我是一名 Python 开发者。

Agent：你好小明！很高兴认识你，Python 是一门很棒的语言。

你：我叫什么名字？

Agent：你叫小明，你是一名 Python 开发者。

你：根据我的职业，推荐一个适合我的技术栈。

Agent：既然你是 Python 开发者，我推荐你学习以下技术栈：...
```

本章我们就来解决这个问题，给 Agent 加上 **短期记忆** 和 **长期记忆**。

---

## 5.2 短期记忆

### 什么是短期记忆？

短期记忆就是**当前对话中的消息历史**。它的原理很简单：每次调用 LLM 时，把之前的对话消息也一起传进去。

```
没有短期记忆：
  LLM 只看到当前消息 → 不知道之前聊了什么

有短期记忆：
  LLM 看到完整对话历史 → 知道之前聊了什么
```

### 消息列表管理

实现短期记忆的核心就是维护一个 `messages` 列表：

```python
# 短期记忆 = 一个不断增长的消息列表
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "我叫小明"},
    {"role": "assistant", "content": "你好小明！"},
    {"role": "user", "content": "我叫什么？"},
    # LLM 能看到上面所有消息，所以能回答"你叫小明"
]
```

每次对话时：
1. 用户发来新消息 → 追加到 `messages`
2. 调用 LLM（传入完整的 `messages`）
3. LLM 回复 → 追加到 `messages`
4. 如果调用了工具 → 工具消息也追加到 `messages`

### 上下文窗口限制

LLM 能处理的文本长度是有限的，这个限制叫做 **上下文窗口（Context Window）**：

| 模型 | 上下文窗口 | 大约能放多少字 |
|------|-----------|--------------|
| GPT-4o-mini | 128K tokens | 约 10 万字 |
| GPT-4o | 128K tokens | 约 10 万字 |
| DeepSeek-V3 | 64K tokens | 约 5 万字 |
| Claude 3.5 | 200K tokens | 约 15 万字 |

看起来很大，但如果对话很长、或者工具返回的数据很多，很快就会用完。当消息超出上下文窗口时，LLM 会报错或者截断输入。

**解决方案**：当消息列表太长时，只保留最近的消息，丢弃最早的消息。这叫做 **滑动窗口（Sliding Window）**。

```python
def trim_messages(messages: list, max_messages: int = 20) -> list:
    """
    保持消息列表不超过指定长度
    始终保留 system 消息（第一条）
    """
    if len(messages) <= max_messages:
        return messages
    # 保留 system 消息 + 最近的 max_messages-1 条消息
    return [messages[0]] + messages[-(max_messages - 1):]
```

### 完整代码示例

下面我们在第4章 ReAct Agent 的基础上，加上短期记忆：

```python
# agent_with_short_memory.py
"""
带短期记忆的 ReAct Agent
在第4章基础上添加消息历史管理
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
            "description": "获取指定城市的天气信息",
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
            "description": "计算数学表达式",
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
    """获取天气（模拟数据）"""
    data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
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

# ==================== 记忆管理 ====================
class ConversationMemory:
    """
    对话记忆管理器
    负责维护消息列表，处理上下文窗口限制
    """

    def __init__(self, system_prompt: str, max_messages: int = 30):
        """
        初始化记忆

        参数：
            system_prompt: 系统提示词
            max_messages: 最大消息数量（防止超出上下文窗口）
        """
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_message(self, tool_call_id: str, content: str):
        """添加工具返回消息"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._trim()

    def add_assistant_tool_calls(self, message):
        """
        添加包含 tool_calls 的助手消息
        （需要把整个 message 对象存起来，因为里面有 tool_call_id）
        """
        self.messages.append(message)
        self._trim()

    def get_messages(self) -> list:
        """获取当前所有消息"""
        return self.messages

    def _trim(self):
        """裁剪消息列表，保留 system 消息 + 最近的消息"""
        if len(self.messages) > self.max_messages:
            self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]

    def clear(self):
        """清空对话历史（保留 system 消息）"""
        self.messages = [self.messages[0]]


# ==================== Agent 核心 ====================
SYSTEM_PROMPT = (
    "你是一个智能助手，可以查询天气和进行数学计算。\n"
    "请根据用户的问题，合理选择工具，一步步解决问题。\n"
    "获取到足够信息后，用自然语言给出最终回答。\n"
    "重要：请记住用户在对话中提到的个人信息。"
)


def run_agent(memory: ConversationMemory, max_rounds: int = 5) -> str:
    """
    运行 Agent（带记忆版本）

    参数：
        memory: 对话记忆管理器
        max_rounds: 最大循环轮数
    """
    for i in range(max_rounds):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=memory.get_messages(),
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # 情况1：LLM 没有调用工具 → 任务完成
        if not msg.tool_calls:
            memory.add_assistant_message(msg.content)
            return msg.content

        # 情况2：LLM 想调用工具
        memory.add_assistant_tool_calls(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [调用工具] {name}({args})")

            result = TOOL_MAP[name](**args)
            print(f"  [工具结果] {result}")

            memory.add_tool_message(tc.id, result)

    return "Agent 达到最大循环次数，未能完成任务。"


# ==================== 交互入口 ====================
if __name__ == "__main__":
    memory = ConversationMemory(SYSTEM_PROMPT)
    print("Agent 已启动（带短期记忆）！输入 'quit' 退出，'clear' 清空记忆\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            memory.clear()
            print("记忆已清空！\n")
            continue
        if not user_input:
            continue

        memory.add_user_message(user_input)
        print()
        answer = run_agent(memory)
        print(f"\nAgent：{answer}\n")
```

### 运行演示

```bash
python agent_with_short_memory.py
```

```
Agent 已启动（带短期记忆）！输入 'quit' 退出，'clear' 清空记忆

你：你好，我叫小明，我是一名 Python 开发者。

Agent：你好小明！很高兴认识你，作为一名 Python 开发者，你一定对编程充满热情。有什么我可以帮你的吗？

你：我叫什么名字？做什么工作的？

Agent：你叫小明，是一名 Python 开发者。

你：北京天气怎么样？

  [调用工具] get_weather({'city': '北京'})
  [工具结果] {"temperature": 35, "weather": "晴天", "humidity": 40}

Agent：北京今天天气晴朗，温度35度，湿度40%。天气比较炎热，小明要注意防晒和补水哦！

你：刚才查的是哪个城市的天气？

Agent：刚才查的是北京的天气。

你：quit
再见！
```

注意最后两个问题 —— Agent 都能正确回答，因为它有完整的对话记忆。

---

## 5.3 长期记忆

### 为什么需要长期记忆？

短期记忆只在当前对话中有效。一旦你关闭程序、重新启动，之前的对话就全丢了。

但很多时候，我们希望 Agent 能**记住跨会话的信息**：

- 用户的个人信息（名字、职业、偏好）
- 之前讨论过的话题
- 用户的历史请求

这就是长期记忆的作用。

### 简单文件存储

最简单的长期记忆实现方式就是用文件。每次对话结束时，把重要信息保存到文件；下次对话开始时，从文件中读取。

```python
# long_memory_file.py
"""
基于文件存储的长期记忆
把用户的个人信息保存到 JSON 文件中
"""
import os
import json

MEMORY_FILE = "user_memory.json"


class LongTermMemory:
    """
    基于文件的长期记忆
    用 JSON 文件存储用户的个人信息和偏好
    """

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        """从文件加载记忆"""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"user_info": {}, "preferences": {}, "history": []}

    def _save(self):
        """保存记忆到文件"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save_user_info(self, key: str, value: str):
        """保存用户信息"""
        self.data["user_info"][key] = value
        self._save()

    def get_user_info(self, key: str) -> str | None:
        """获取用户信息"""
        return self.data["user_info"].get(key)

    def get_all_user_info(self) -> str:
        """获取所有用户信息（用于拼接到 system prompt）"""
        if not self.data["user_info"]:
            return ""
        info_lines = [f"- {k}: {v}" for k, v in self.data["user_info"].items()]
        return "已知的用户信息：\n" + "\n".join(info_lines)

    def add_history(self, content: str):
        """添加历史记录"""
        self.data["history"].append(content)
        # 只保留最近 50 条
        self.data["history"] = self.data["history"][-50:]
        self._save()
```

### 向量数据库简介

当需要存储大量文本（比如几百篇文章、几千条对话记录）时，简单的文件存储就不够用了。这时候需要 **向量数据库（Vector Database）**。

**什么是向量数据库？**

```
传统数据库：精确匹配
  查询 "天气" → 只能找到包含"天气"这两个字的记录

向量数据库：语义匹配
  查询 "天气" → 能找到"今天阳光明媚"、"外面在下雨"、"气温35度"等语义相关的记录
```

原理很简单：
1. 把文本通过 Embedding 模型转换成一串数字（向量）
2. 把向量存入数据库
3. 查询时，把查询文本也转成向量，然后在数据库中找"距离最近"的向量

```
文本 → Embedding模型 → [0.12, -0.34, 0.56, ...]  ← 这就是向量
                              │
                              ▼
                        向量数据库
                        （存储和检索）
```

常见的向量数据库：
- **Chroma**：轻量级，适合学习和小项目
- **FAISS**：Facebook 开源，性能好
- **Pinecone**：云服务，无需自己部署
- **Milvus**：国产开源，功能强大

> **注意**：向量数据库是一个很大的话题，我们会在第11章（RAG 知识库 Agent）中深入学习。这里只需要知道它的概念即可。

### 完整代码示例

下面实现一个带长期记忆的 Agent，使用文件存储：

```python
# agent_with_long_memory.py
"""
带长期记忆的 ReAct Agent
使用文件存储跨会话的用户信息
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

# ==================== 长期记忆 ====================
MEMORY_FILE = "user_memory.json"


class LongTermMemory:
    """基于文件的长期记忆"""

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"user_info": {}, "history": []}

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save_info(self, key: str, value: str):
        self.data["user_info"][key] = value
        self._save()

    def get_context(self) -> str:
        """获取长期记忆上下文（拼接到 system prompt）"""
        if not self.data["user_info"]:
            return ""
        lines = [f"- {k}：{v}" for k, v in self.data["user_info"].items()]
        return "\n\n你已知关于用户的信息：\n" + "\n".join(lines)


# ==================== 工具定义 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
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
            "name": "save_user_info",
            "description": "保存用户的个人信息到长期记忆中",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "信息类别，如：姓名、职业、城市"},
                    "value": {"type": "string", "description": "信息内容"},
                },
                "required": ["key", "value"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
    }
    result = data.get(city, {"temperature": 25, "weather": "未知", "humidity": 50})
    return json.dumps(result, ensure_ascii=False)


def save_user_info(key: str, value: str) -> str:
    """保存用户信息到长期记忆"""
    long_memory.save_info(key, value)
    return f"已保存：{key} = {value}"


TOOL_MAP = {
    "get_weather": get_weather,
    "save_user_info": save_user_info,
}

# 全局长期记忆实例
long_memory = LongTermMemory()

# ==================== Agent 核心 ====================
def build_system_prompt() -> str:
    """构建系统提示词（包含长期记忆上下文）"""
    base = (
        "你是一个智能助手，可以查询天气。\n"
        "当用户告诉你关于他自己的信息时（如姓名、职业、喜好等），"
        "请主动使用 save_user_info 工具保存这些信息。\n"
        "在回答问题时，利用你已知的信息提供个性化的回复。"
    )
    memory_context = long_memory.get_context()
    return base + memory_context


def run_agent(messages: list, max_rounds: int = 5) -> str:
    """运行 Agent"""
    for i in range(max_rounds):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [调用工具] {name}({args})")

            result = TOOL_MAP[name](**args)
            print(f"  [工具结果] {result}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Agent 达到最大循环次数。"


# ==================== 交互入口 ====================
if __name__ == "__main__":
    print("Agent 已启动（带长期记忆）！输入 'quit' 退出\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        print()
        answer = run_agent(messages)
        print(f"\nAgent：{answer}\n")
```

---

## 5.4 完整代码：有记忆的 Agent

下面把短期记忆和长期记忆整合到一起，构建一个完整的"有记忆的 Agent"：

```python
# full_memory_agent.py
"""
完整的带记忆 Agent —— 整合短期记忆 + 长期记忆
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

# ==================== 长期记忆（文件存储） ====================
MEMORY_FILE = "user_memory.json"


class LongTermMemory:
    """基于文件的长期记忆管理"""

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"user_info": {}}

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def save_info(self, key: str, value: str):
        self.data["user_info"][key] = value
        self._save()

    def get_context(self) -> str:
        if not self.data["user_info"]:
            return ""
        lines = [f"- {k}：{v}" for k, v in self.data["user_info"].items()]
        return "\n\n你已知关于用户的信息（来自长期记忆）：\n" + "\n".join(lines)


# ==================== 短期记忆（消息列表） ====================
class ShortTermMemory:
    """对话短期记忆管理"""

    def __init__(self, system_prompt: str, max_messages: int = 30):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_messages = max_messages

    def add(self, message: dict):
        self.messages.append(message)
        self._trim()

    def get(self) -> list:
        return self.messages

    def clear(self):
        self.messages = [self.messages[0]]

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]


# ==================== 工具 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
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
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": "保存用户的个人信息到长期记忆",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "信息类别，如：姓名、职业、城市"},
                    "value": {"type": "string", "description": "信息内容"},
                },
                "required": ["key", "value"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    data = {
        "北京": {"temperature": 35, "weather": "晴天", "humidity": 40},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 65},
        "广州": {"temperature": 32, "weather": "阵雨", "humidity": 80},
    }
    result = data.get(city, {"temperature": 25, "weather": "未知", "humidity": 50})
    return json.dumps(result, ensure_ascii=False)


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "不允许的字符"})
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def save_user_info(key: str, value: str) -> str:
    long_memory.save_info(key, value)
    return f"已保存到长期记忆：{key} = {value}"


TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "save_user_info": save_user_info,
}

# ==================== 初始化 ====================
long_memory = LongTermMemory()

SYSTEM_PROMPT = (
    "你是一个智能助手，可以查询天气和进行数学计算。\n"
    "当用户告诉你关于他自己的信息时（姓名、职业、喜好等），"
    "请主动使用 save_user_info 工具保存。\n"
    "利用已知信息提供个性化回复。"
)


def run_agent(short_memory: ShortTermMemory, max_rounds: int = 5) -> str:
    """运行 Agent"""
    for i in range(max_rounds):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=short_memory.get(),
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            short_memory.add({"role": "assistant", "content": msg.content})
            return msg.content

        short_memory.add(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [调用工具] {name}({args})")
            result = TOOL_MAP[name](**args)
            print(f"  [工具结果] {result}")
            short_memory.add({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Agent 达到最大循环次数。"


# ==================== 交互入口 ====================
if __name__ == "__main__":
    # 构建包含长期记忆的 system prompt
    full_prompt = SYSTEM_PROMPT + long_memory.get_context()
    short_memory = ShortTermMemory(full_prompt)

    print("Agent 已启动（短期记忆 + 长期记忆）！")
    print("输入 'quit' 退出，'clear' 清空短期记忆\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            short_memory.clear()
            print("短期记忆已清空（长期记忆仍在）\n")
            continue
        if not user_input:
            continue

        short_memory.add({"role": "user", "content": user_input})
        print()
        answer = run_agent(short_memory)
        print(f"\nAgent：{answer}\n")
```

---

## 5.5 运行演示

### 第一次运行

```bash
python full_memory_agent.py
```

```
Agent 已启动（短期记忆 + 长期记忆）！
输入 'quit' 退出，'clear' 清空短期记忆

你：你好，我叫小明，我是一名 Python 开发者，住在上海。

  [调用工具] save_user_info({'key': '姓名', 'value': '小明'})
  [工具结果] 已保存到长期记忆：姓名 = 小明
  [调用工具] save_user_info({'key': '职业', 'value': 'Python 开发者'})
  [工具结果] 已保存到长期记忆：职业 = Python 开发者
  [调用工具] save_user_info({'key': '城市', 'value': '上海'})
  [工具结果] 已保存到长期记忆：城市 = 上海

Agent：你好小明！很高兴认识你。作为一名上海的 Python 开发者，你一定对技术充满热情。有什么我可以帮你的吗？

你：quit
再见！
```

### 第二次运行（重新启动程序后）

```bash
python full_memory_agent.py
```

```
Agent 已启动（短期记忆 + 长期记忆）！
输入 'quit' 退出，'clear' 清空短期记忆

你：你还记得我是谁吗？

Agent：当然记得！你是小明，一名 Python 开发者，住在上海。有什么需要帮忙的吗？

你：帮我查一下我所在城市的天气。

  [调用工具] get_weather({'city': '上海'})
  [工具结果] {"temperature": 28, "weather": "多云", "humidity": 65}

Agent：小明，上海今天多云，温度28度，湿度65%。天气比较舒适，适合外出活动！
```

看到了吗？即使重新启动了程序，Agent 依然记得你是谁、住在哪里。这就是长期记忆的力量。

---

## 5.6 踩坑记录

### 坑1：消息列表无限增长导致报错

**现象**：对话进行到一定程度后，突然报错 `This model's maximum context length is...`

**原因**：消息列表超出了 LLM 的上下文窗口限制。

**解决方法**：使用滑动窗口机制，限制消息数量：

```python
# 始终保留 system 消息 + 最近的 N 条消息
if len(messages) > max_messages:
    messages = [messages[0]] + messages[-(max_messages - 1):]
```

### 坑2：工具消息丢失 tool_call_id

**现象**：LLM 报错 `tool_call_id not found`

**原因**：添加工具结果消息时，`tool_call_id` 和之前 LLM 返回的 `tool_call.id` 不匹配。

**解决方法**：确保使用同一个 `tool_call.id`：

```python
# 正确做法：从 LLM 的响应中获取 tool_call_id
for tc in msg.tool_calls:
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,  # 必须和 LLM 返回的 id 一致
        "content": result,
    })
```

### 坑3：长期记忆文件编码问题

**现象**：保存中文时出现乱码。

**解决方法**：读写文件时指定 `encoding="utf-8"`：

```python
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### 坑4：Agent 不会主动保存用户信息

**现象**：用户告诉 Agent 自己的名字，但 Agent 没有调用 `save_user_info`。

**解决方法**：在 system prompt 中明确指示 Agent 主动保存用户信息：

```python
"当用户告诉你关于他自己的信息时（姓名、职业、喜好等），"
"请主动使用 save_user_info 工具保存这些信息。"
```

### 坑5：JSON 文件被意外覆盖

**现象**：程序异常退出后，`user_memory.json` 文件内容为空。

**解决方法**：先写入临时文件，再重命名：

```python
import tempfile

def _save(self):
    # 先写入临时文件
    tmp_path = self.filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(self.data, f, ensure_ascii=False, indent=2)
    # 再重命名（原子操作）
    os.replace(tmp_path, self.filepath)
```

---

## 5.7 动手练习

### 练习1：添加记忆查询工具（难度：简单）

给 Agent 添加一个 `get_user_info` 工具，让用户可以主动查询已保存的信息：

```python
# 提示：添加一个新的工具定义
{
    "type": "function",
    "function": {
        "name": "get_user_info",
        "description": "查询已保存的用户信息",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "要查询的信息类别"},
            },
            "required": ["key"],
        },
    },
}
```

### 练习2：实现对话摘要（难度：中等）

当消息列表快要超出限制时，不是简单地丢弃旧消息，而是让 LLM 把旧消息总结成一段"摘要"，然后保留摘要 + 最近的消息。

提示：
1. 检测消息数量是否接近限制
2. 如果接近，把旧消息发给 LLM，让它生成摘要
3. 用摘要替换旧消息

### 练习3：用 SQLite 替换文件存储（难度：中等）

把长期记忆从 JSON 文件改为 SQLite 数据库，支持按时间范围查询历史记录。

提示：
```python
import sqlite3
conn = sqlite3.connect("agent_memory.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS user_info (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
```

---

## 5.8 小结

本章我们学习了如何给 Agent 加上记忆系统：

| 记忆类型 | 实现方式 | 生命周期 | 适用场景 |
|---------|---------|---------|---------|
| **短期记忆** | 消息列表 | 当前对话 | 保持上下文连续 |
| **长期记忆** | 文件/数据库 | 跨会话 | 记住用户信息 |

核心要点：
1. **短期记忆** = 维护 `messages` 列表，每次调用 LLM 时传入完整历史
2. **上下文窗口** 是有限的，需要用滑动窗口机制裁剪消息
3. **长期记忆** 可以用简单的 JSON 文件实现，适合存储用户个人信息
4. **向量数据库** 适合存储大量文本，支持语义搜索（后续章节深入学习）
5. 通过 `save_user_info` 工具让 Agent 主动保存用户信息

---

## 下一章预告

记忆让 Agent "认识你"，但 Agent 还需要更强大的"双手"才能帮你做更多事。

在 **第6章：让 Agent 使用工具** 中，我们将：
- 深入理解 Function Calling 的原理
- 学习如何定义各种类型的工具
- 实现内置工具（代码执行器、文件操作）
- 构建自定义工具（搜索、数据库查询）
- 学习工具链 —— 多个工具组合完成复杂任务

---

> 上一篇：[第4章：手写第一个 Agent](../04-first-agent/README.md) | 下一篇：[第6章：让 Agent 使用工具](../06-agent-tools/README.md)
