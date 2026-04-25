# 第6章：让 Agent 使用工具

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 6.1 工具的本质：Function Calling 原理详解

在第4章中，我们已经用过 Function Calling，但你可能对它的底层原理还不太清楚。本章我们从原理出发，一步步构建一个功能丰富的"多功能 Agent"。

### Function Calling 到底是什么？

**Function Calling 不是让 LLM 执行代码**，而是一种让 LLM 输出"结构化指令"的机制。

```
普通 LLM 调用：
  输入："北京天气怎么样？"
  输出："北京今天晴，温度25度..."（纯文本）

Function Calling 调用：
  输入："北京天气怎么样？" + 工具定义
  输出：{
    "function": "get_weather",
    "arguments": {"city": "北京"}
  }（结构化的调用指令）
```

LLM 并没有真的去查天气，它只是"告诉"你的代码：**"你应该调用 get_weather 这个函数，参数是 city='北京'"**。

### 完整的交互流程

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  你的代码  │     │   LLM    │     │  你的代码  │     │   LLM    │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │ 1. 发送消息     │                │                │
     │ + 工具定义     │                │                │
     │ ──────────────→│                │                │
     │                │                │                │
     │ 2. 返回工具调用 │                │                │
     │   指令         │                │                │
     │←────────────── │                │                │
     │                │                │                │
     │ 3. 你的代码     │                │                │
     │   执行函数     │                │                │
     │   获取结果     │                │                │
     │                │                │                │
     │ 4. 发送工具结果 │                │                │
     │ ───────────────────────────────→│                │
     │                │                │                │
     │ 5. 返回最终回答 │                │                │
     │←─────────────────────────────── │                │
     │                │                │                │
```

关键点：
- **步骤2**：LLM 返回的不是普通文本，而是一个 `tool_calls` 对象
- **步骤3**：你的代码负责实际执行函数
- **步骤4**：把执行结果以 `role: "tool"` 的形式发回给 LLM
- **步骤5**：LLM 根据工具结果生成自然语言回答

---

## 6.2 定义工具：JSON Schema 格式详解

工具定义使用的是 **JSON Schema** 格式。别被这个名字吓到，其实很简单。

### 基本结构

```python
tool_definition = {
    "type": "function",                    # 固定为 "function"
    "function": {
        "name": "工具名称",                  # 函数名（英文，用下划线分隔）
        "description": "工具的详细描述",      # 非常重要！LLM 靠这个理解工具用途
        "parameters": {                     # 参数定义（JSON Schema 格式）
            "type": "object",               # 参数整体是一个对象
            "properties": {                 # 每个参数的定义
                "参数名": {
                    "type": "string",       # 参数类型：string/number/boolean/array
                    "description": "参数说明",
                    "enum": ["选项1", "选项2"],  # 可选：限定取值范围
                },
            },
            "required": ["必填参数1"],        # 必填参数列表
        },
    },
}
```

### 参数类型

JSON Schema 支持以下常用类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| `string` | 字符串 | `"北京"`、`"hello"` |
| `number` | 数字（整数或小数） | `25`、`3.14` |
| `integer` | 整数 | `1`、`100` |
| `boolean` | 布尔值 | `true`、`false` |
| `array` | 数组 | `["北京", "上海"]` |
| `object` | 对象（嵌套） | `{"city": "北京", "date": "2025-01-01"}` |

### 实际示例

```python
# 示例1：简单参数
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息，包括温度、天气状况和湿度",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、广州",
                },
            },
            "required": ["city"],
        },
    },
}

# 示例2：多个参数 + enum
{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "搜索航班信息",
        "parameters": {
            "type": "object",
            "properties": {
                "departure": {
                    "type": "string",
                    "description": "出发城市",
                },
                "destination": {
                    "type": "string",
                    "description": "目的城市",
                },
                "date": {
                    "type": "string",
                    "description": "出发日期，格式：YYYY-MM-DD",
                },
                "class_type": {
                    "type": "string",
                    "description": "舱位类型",
                    "enum": ["economy", "business", "first"],
                },
            },
            "required": ["departure", "destination", "date"],
        },
    },
}

# 示例3：数组参数
{
    "type": "function",
    "function": {
        "name": "calculate_statistics",
        "description": "计算一组数字的统计信息（平均值、最大值、最小值）",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "数字列表，例如：[1, 2, 3, 4, 5]",
                },
                "operation": {
                    "type": "string",
                    "description": "统计操作类型",
                    "enum": ["mean", "max", "min", "sum"],
                },
            },
            "required": ["numbers", "operation"],
        },
    },
}
```

---

## 6.3 内置工具：代码执行器与文件操作

### 代码执行器

让 Agent 能执行 Python 代码，这是非常强大的能力：

```python
# tool_code_executor.py
"""
代码执行器工具
让 Agent 能运行 Python 代码
"""
import subprocess
import json


def execute_python_code(code: str) -> str:
    """
    安全执行 Python 代码

    参数：
        code: Python 代码字符串

    返回：
        执行结果（stdout）或错误信息（stderr）
    """
    try:
        # 使用 subprocess 执行代码，设置超时时间
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,  # 最多执行30秒
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        if error:
            return f"执行错误：\n{error}"
        if output:
            return f"执行结果：\n{output}"
        return "代码执行成功，没有输出。"

    except subprocess.TimeoutExpired:
        return "执行超时（超过30秒），代码可能存在死循环。"
    except Exception as e:
        return f"执行异常：{e}"


# 对应的工具定义
code_executor_tool = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "执行 Python 代码并返回结果。"
            "适用于数学计算、数据处理、字符串操作等任务。"
            "代码应该使用 print() 输出结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                },
            },
            "required": ["code"],
        },
    },
}
```

### 文件操作工具

```python
# tool_file_operations.py
"""
文件操作工具
让 Agent 能读写文件
"""
import os


def read_file(filepath: str) -> str:
    """读取文件内容"""
    try:
        if not os.path.exists(filepath):
            return f"错误：文件 '{filepath}' 不存在"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # 限制返回内容长度，避免超出上下文窗口
        if len(content) > 5000:
            return content[:5000] + "\n\n...（内容过长，已截断）"
        return content
    except Exception as e:
        return f"读取文件错误：{e}"


def write_file(filepath: str, content: str) -> str:
    """写入文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件：{filepath}（{len(content)} 字符）"
    except Exception as e:
        return f"写入文件错误：{e}"


def list_files(directory: str = ".") -> str:
    """列出目录中的文件"""
    try:
        files = os.listdir(directory)
        if not files:
            return f"目录 '{directory}' 为空"
        result = []
        for f in sorted(files)[:50]:  # 最多显示50个
            full_path = os.path.join(directory, f)
            if os.path.isdir(full_path):
                result.append(f"[目录] {f}")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[文件] {f} ({size} 字节)")
        return "\n".join(result)
    except Exception as e:
        return f"列出文件错误：{e}"


# 对应的工具定义
file_tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件（会覆盖已有内容）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出指定目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径，默认为当前目录"},
                },
                "required": [],
            },
        },
    },
]
```

---

## 6.4 自定义工具：搜索与数据库查询

### 搜索工具

```python
# tool_search.py
"""
搜索工具
模拟网络搜索（实际项目中可接入真实搜索 API）
"""
import json
import random


def web_search(query: str) -> str:
    """
    模拟网络搜索

    在实际项目中，你可以接入：
    - DuckDuckGo API（免费）
    - SerpAPI（付费，质量高）
    - Bing Search API
    - Tavily（专为 AI Agent 设计的搜索 API）
    """
    # 模拟搜索结果（实际项目中替换为真实 API 调用）
    mock_results = {
        "Python": [
            {"title": "Python 官方文档", "url": "python.org", "snippet": "Python 是一种解释型、面向对象的高级编程语言。"},
            {"title": "Python 教程 - 菜鸟教程", "url": "runoob.com", "snippet": "Python3 教程，适合初学者入门。"},
        ],
        "AI Agent": [
            {"title": "什么是 AI Agent", "url": "example.com", "snippet": "AI Agent 是能自主感知环境、制定决策的 AI 系统。"},
            {"title": "Agent 开发入门", "url": "example.com", "snippet": "从零开始学习 AI Agent 开发。"},
        ],
    }

    # 查找匹配的结果
    results = []
    for keyword, items in mock_results.items():
        if keyword.lower() in query.lower():
            results.extend(items)

    if not results:
        results = [
            {"title": f"关于 '{query}' 的搜索结果", "url": "example.com",
             "snippet": f"这是关于 '{query}' 的模拟搜索结果。实际项目中请接入真实搜索 API。"},
        ]

    return json.dumps(results[:3], ensure_ascii=False, indent=2)


# 对应的工具定义
search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "搜索互联网上的信息。"
            "当需要查找实时信息、最新数据、或者你不确定的知识时使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
}
```

### 数据库查询工具

```python
# tool_database.py
"""
数据库查询工具
使用 SQLite 演示数据库操作
"""
import sqlite3
import json


def init_demo_db():
    """初始化演示数据库"""
    conn = sqlite3.connect(":memory:")  # 使用内存数据库
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)
    # 插入示例数据
    products = [
        ("Python 编程入门", "书籍", 59.9, 100),
        ("机械键盘", "电子产品", 299.0, 50),
        ("运动水杯", "生活用品", 39.9, 200),
        ("蓝牙耳机", "电子产品", 199.0, 80),
        ("AI Agent 开发实战", "书籍", 79.9, 30),
    ]
    conn.executemany(
        "INSERT INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
        products,
    )
    conn.commit()
    return conn


# 全局数据库连接
db_conn = init_demo_db()


def query_database(sql: str) -> str:
    """
    执行 SQL 查询（只允许 SELECT 语句，保证安全）

    参数：
        sql: SQL 查询语句

    返回：
        查询结果的 JSON 字符串
    """
    try:
        # 安全检查：只允许 SELECT 语句
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            return "错误：只允许执行 SELECT 查询语句"

        cursor = db_conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # 转换为字典列表
        results = [dict(zip(columns, row)) for row in rows]

        if not results:
            return "查询结果为空"

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"查询错误：{e}"


# 对应的工具定义
database_tool = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": (
            "查询产品数据库。"
            "数据库有一个 products 表，包含字段：id, name（名称）, category（类别）, price（价格）, stock（库存）。"
            "只支持 SELECT 查询。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT 查询语句"},
            },
            "required": ["sql"],
        },
    },
}
```

---

## 6.5 工具描述的艺术

工具描述（`description`）是 LLM 理解工具用途的**唯一途径**。描述写得好不好，直接决定了 LLM 能不能正确选择和使用工具。

### 好的描述 vs 差的描述

```python
# 差的描述 —— 太简短，LLM 不知道什么时候该用
{
    "name": "search",
    "description": "搜索信息",
}

# 好的描述 —— 详细说明了用途、适用场景、输入要求
{
    "name": "web_search",
    "description": (
        "搜索互联网上的信息。"
        "当需要查找实时信息、最新数据、新闻、或者你不确定的知识时使用。"
        "不要用此工具查找简单的常识问题。"
    ),
}

# 差的描述 —— 没有说明参数的含义和格式
{
    "name": "get_weather",
    "description": "获取天气",
    "parameters": {
        "properties": {
            "city": {"type": "string"},
            "date": {"type": "string"},
        },
    },
}

# 好的描述 —— 参数说明清晰，包含示例
{
    "name": "get_weather",
    "description": "获取指定城市和日期的天气信息",
    "parameters": {
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如：北京、上海、广州",
            },
            "date": {
                "type": "string",
                "description": "日期，格式为 YYYY-MM-DD，例如：2025-01-15。不指定则查询今天。",
            },
        },
    },
}
```

### 工具描述的写作技巧

1. **说清楚"做什么"**：工具的功能是什么
2. **说清楚"什么时候用"**：什么场景下应该用这个工具
3. **说清楚"什么时候不用"**：避免 LLM 滥用工具
4. **给参数加示例**：帮助 LLM 理解参数格式
5. **用英文命名，中文描述**：工具名用英文，描述用中文（或根据模型习惯调整）

---

## 6.6 工具链：多工具组合

有时候一个任务需要多个工具配合完成，这就是 **工具链（Tool Chain）**。

```
用户：帮我搜索 Python 入门教程，然后把搜索结果保存到文件里

工具链：
  步骤1：调用 web_search("Python 入门教程") → 获取搜索结果
  步骤2：调用 write_file("search_results.txt", 搜索结果) → 保存到文件
```

工具链不需要你手动编排 —— **ReAct 循环会自动处理**。LLM 会根据任务需要，自主决定调用哪些工具、以什么顺序调用。

你只需要：
1. 把所有可用的工具都定义好
2. 在 system prompt 中告诉 LLM 有哪些工具可用
3. 让 ReAct 循环自动运行

LLM 会自己判断：
- 先搜索还是先写文件？ → 先搜索
- 搜索结果要不要处理？ → 直接保存即可
- 还需要其他工具吗？ → 不需要了，任务完成

---

## 6.7 完整代码：多功能 Agent

下面把所有工具整合到一起，构建一个"多功能 Agent"：

```python
# multi_tool_agent.py
"""
多功能 Agent —— 整合多种工具
作者：青松与桑叶
"""
import os
import json
import sqlite3
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ==================== 工具实现 ====================

def get_weather(city: str) -> str:
    """获取天气（模拟数据）"""
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


def execute_python(code: str) -> str:
    """执行 Python 代码"""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        if result.stderr:
            return f"错误：\n{result.stderr}"
        return result.stdout if result.stdout else "执行成功，无输出。"
    except subprocess.TimeoutExpired:
        return "执行超时（30秒）"
    except Exception as e:
        return f"异常：{e}"


def web_search(query: str) -> str:
    """模拟网络搜索"""
    mock = {
        "Python": [
            {"title": "Python 官方文档", "snippet": "Python 是一种高级编程语言。"},
            {"title": "Python 入门教程", "snippet": "从零开始学习 Python 编程。"},
        ],
    }
    for keyword, items in mock.items():
        if keyword.lower() in query.lower():
            return json.dumps(items, ensure_ascii=False)
    return json.dumps([{
        "title": f"'{query}' 的搜索结果",
        "snippet": f"这是关于 '{query}' 的模拟结果。",
    }], ensure_ascii=False)


def read_file(filepath: str) -> str:
    """读取文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:5000] if len(content) > 5000 else content
    except FileNotFoundError:
        return f"文件 '{filepath}' 不存在"
    except Exception as e:
        return f"读取错误：{e}"


def write_file(filepath: str, content: str) -> str:
    """写入文件"""
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {filepath}（{len(content)} 字符）"
    except Exception as e:
        return f"写入错误：{e}"


def list_files(directory: str = ".") -> str:
    """列出目录文件"""
    try:
        files = os.listdir(directory)
        if not files:
            return f"'{directory}' 为空"
        return "\n".join(
            f"[{'目录' if os.path.isdir(os.path.join(directory, f)) else '文件'}] {f}"
            for f in sorted(files)[:30]
        )
    except Exception as e:
        return f"错误：{e}"


# 初始化演示数据库
db_conn = sqlite3.connect(":memory:")
db_conn.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER
    )
""")
db_conn.executemany(
    "INSERT INTO products VALUES (?,?,?,?,?)",
    [(1, "Python 入门", "书籍", 59.9, 100),
     (2, "机械键盘", "电子", 299.0, 50),
     (3, "蓝牙耳机", "电子", 199.0, 80),
     (4, "AI Agent 实战", "书籍", 79.9, 30)],
)
db_conn.commit()


def query_database(sql: str) -> str:
    """查询数据库（仅 SELECT）"""
    try:
        if not sql.strip().upper().startswith("SELECT"):
            return "错误：仅支持 SELECT 查询"
        cursor = db_conn.execute(sql)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
        return json.dumps(rows, ensure_ascii=False) if rows else "查询为空"
    except Exception as e:
        return f"查询错误：{e}"


# 工具映射
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "execute_python": execute_python,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "query_database": query_database,
}

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
                    "city": {"type": "string", "description": "城市名称，如：北京、上海"},
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
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "执行 Python 代码。适用于复杂数学计算、数据处理等场景。代码应用 print() 输出结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python 代码"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息。当需要查找实时数据、最新资讯或不确定的知识时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件（覆盖模式）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径，默认当前目录"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "查询产品数据库。products 表字段：id, name(名称), category(类别), price(价格), stock(库存)。"
                "仅支持 SELECT。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT 语句"},
                },
                "required": ["sql"],
            },
        },
    },
]

# ==================== Agent 核心 ====================
SYSTEM_PROMPT = (
    "你是一个多功能智能助手，拥有以下能力：\n"
    "- 查询天气\n"
    "- 数学计算\n"
    "- 执行 Python 代码\n"
    "- 搜索互联网\n"
    "- 读写文件\n"
    "- 查询数据库\n\n"
    "请根据用户的问题，选择合适的工具来完成任务。\n"
    "如果需要多步操作，请一步步来。\n"
    "获取到足够信息后，用自然语言给出最终回答。"
)


def run_agent(user_message: str, max_rounds: int = 8) -> str:
    """运行多功能 Agent"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    print(f"{'='*60}")
    print(f"用户：{user_message}")
    print(f"{'='*60}\n")

    for i in range(max_rounds):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"Agent：{msg.content}\n")
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [工具] {name}({json.dumps(args, ensure_ascii=False)})")

            result = TOOL_MAP[name](**args)
            # 截断过长的结果
            if len(result) > 2000:
                result = result[:2000] + "\n...（结果已截断）"
            print(f"  [结果] {result}\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "达到最大循环次数。"


# ==================== 交互入口 ====================
if __name__ == "__main__":
    print("多功能 Agent 已启动！输入 'quit' 退出\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue
        run_agent(user_input)
```

---

## 6.8 运行演示

```bash
python multi_tool_agent.py
```

### 测试1：单工具调用

```
你：北京天气怎么样？

============================================================
用户：北京天气怎么样？
============================================================

  [工具] get_weather({"city": "北京"})
  [结果] {"temperature": 35, "weather": "晴天", "humidity": 40}

Agent：北京今天天气晴朗，温度35度，湿度40%。天气比较炎热，注意防晒补水！
```

### 测试2：工具链（多工具组合）

```
你：帮我查一下数据库里有哪些书籍，然后把结果保存到 books.txt 文件里

============================================================
用户：帮我查一下数据库里有哪些书籍，然后把结果保存到 books.txt 文件里
============================================================

  [工具] query_database({"sql": "SELECT name, price, stock FROM products WHERE category = '书籍'"})
  [结果] [{"name": "Python 入门", "price": 59.9, "stock": 100}, {"name": "AI Agent 实战", "price": 79.9, "stock": 30}]

  [工具] write_file({"filepath": "books.txt", "content": "书籍列表：\n1. Python 入门 - 价格：59.9元，库存：100\n2. AI Agent 实战 - 价格：79.9元，库存：30\n"})

Agent：已查询到数据库中有以下书籍：
1. **Python 入门** - 价格：59.9元，库存：100本
2. **AI Agent 实战** - 价格：79.9元，库存：30本

结果已保存到 books.txt 文件中。
```

### 测试3：代码执行 + 计算

```
你：帮我用 Python 画一个九九乘法表

============================================================
用户：帮我用 Python 画一个九九乘法表
============================================================

  [工具] execute_python({"code": "for i in range(1, 10):\n    for j in range(1, i+1):\n        print(f'{j}x{i}={i*j}', end='\\t')\n    print()"})
  [结果] 1x1=1
1x2=2	2x2=4
1x3=3	2x3=6	3x3=9
...

Agent：这是九九乘法表：
1x1=1
1x2=2	2x2=4
1x3=3	2x3=6	3x3=9
...
```

---

## 6.9 动手练习

### 练习1：添加时间工具（难度：简单）

给 Agent 添加一个获取当前时间的工具：

```python
# 提示
from datetime import datetime

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### 练习2：添加翻译工具（难度：简单）

给 Agent 添加一个翻译工具，支持中英文互译：

```python
# 提示：可以用 LLM 本身来做翻译
def translate(text: str, target_lang: str) -> str:
    # 调用 LLM 进行翻译
    ...
```

### 练习3：添加安全机制（难度：中等）

给代码执行器添加安全沙箱：
1. 禁止执行网络请求（`import requests`、`urllib` 等）
2. 禁止文件系统操作（`open()`、`os` 模块等）
3. 限制内存使用

提示：可以使用 `subprocess` 的 `preexec_fn` 参数来限制子进程的权限。

---

## 6.10 小结

本章我们深入学习了 Agent 的工具系统：

| 知识点 | 核心内容 |
|--------|---------|
| **Function Calling 原理** | LLM 输出结构化指令，代码负责执行 |
| **JSON Schema** | 工具定义的标准格式 |
| **内置工具** | 代码执行器、文件操作 |
| **自定义工具** | 搜索、数据库查询 |
| **工具描述** | description 是 LLM 理解工具的关键 |
| **工具链** | ReAct 循环自动编排多工具组合 |

核心要点：
1. **工具描述写得好，LLM 用得对** —— 花时间打磨 description
2. **工具粒度要合适** —— 太粗不够灵活，太细增加选择难度
3. **工具链靠 ReAct 自动编排** —— 不需要手动设计流程
4. **安全第一** —— 代码执行、文件操作都要做好安全限制

---

## 下一章预告

现在 Agent 有了工具，能做很多事情了。但面对复杂任务，Agent 有时候会"想不清楚"——它不知道先做什么、后做什么，也不知道自己的回答对不对。

在 **第7章：让 Agent 学会规划** 中，我们将：
- 学习 Plan-and-Solve 模式 —— 先制定计划再执行
- 实现反思机制（Reflection）—— Agent 自我检查和纠错
- 让 Agent 面对复杂任务时更加从容

---

> 上一篇：[第5章：让 Agent 拥有记忆](../05-agent-memory/README.md) | 下一篇：[第7章：让 Agent 学会规划](../07-agent-planning/README.md)
