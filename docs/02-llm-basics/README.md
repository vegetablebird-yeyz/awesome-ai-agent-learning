# 第2章：LLM 基础入门

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 2.1 从一个比喻开始

在构建 Agent 之前，我们需要先理解 Agent 的"大脑"——**大语言模型（LLM，Large Language Model）**。

想象一下，你面前有一座巨大的图书馆，里面收藏了互联网上几乎所有的文字内容：书籍、论文、博客、论坛帖子、新闻、代码、维基百科……全部加起来可能有几十亿页。

现在，有一个人花了几年时间，把这些内容全部读完了。不只是"读过"，而是真正理解了其中的逻辑、语法、知识，甚至学会了推理和创作。

这个人就是 LLM。

当你问它问题时，它会根据自己"读过的所有内容"，给你一个最合理的回答。它不是在搜索答案，而是在"理解你的问题"之后，"生成"一个新的回答。

这就是 LLM 的本质：**一个读遍了海量文字、学会了理解和生成人类语言的超级 AI。**

---

## 2.2 什么是 LLM

**LLM（Large Language Model，大语言模型）** 是一种经过大规模文本数据训练的人工智能模型，能够理解和生成人类语言。

拆开来看：

- **Large（大）**：模型参数量很大，通常在数十亿到数千亿之间。参数越多，模型越"聪明"。
- **Language（语言）**：专门针对人类语言进行训练，支持中文、英文等多种语言。
- **Model（模型）**：一个通过数学方法学习数据规律的 AI 系统。

### 常见的 LLM

| 模型名称 | 开发者 | 特点 |
|---------|--------|------|
| GPT-4o / GPT-4o-mini | OpenAI | 综合能力强，生态完善 |
| Claude 3.5 | Anthropic | 长文本理解优秀，安全性好 |
| DeepSeek-V3 | DeepSeek | 国产开源，性价比极高 |
| 通义千问 | 阿里云 | 国产，中文能力强 |
| GLM-4 | 智谱 AI | 国产开源，多模态能力强 |
| Llama 3 | Meta | 开源标杆，社区活跃 |

> **好消息**：这些模型大多提供了兼容 OpenAI 格式的 API，学会调用一个，其他的触类旁通。

---

## 2.3 LLM 的核心能力

LLM 到底能做什么？主要有以下四大核心能力：

### 1. 文本生成

这是 LLM 最基本的能力。给它一个开头，它能续写一篇文章；给它一个主题，它能写出一篇完整的博客。

```
输入：请写一首关于春天的五言绝句
输出：春风拂柳绿，细雨润花红。
      燕子归来早，桃李满园中。
```

### 2. 理解指令

LLM 能准确理解你的意图，并按照要求完成任务。你让它"用小学生能懂的语言解释量子力学"，它就不会用专业术语。

```
输入：用一句话解释什么是黑洞
输出：黑洞就像宇宙中的一个超级吸尘器，连光都逃不出它的引力。
```

### 3. 逻辑推理

LLM 具备一定的推理能力，可以分析问题、推导结论。

```
输入：小明比小红大3岁，小红今年10岁，请问小明今年几岁？
输出：小红今年10岁，小明比小红大3岁，所以小明今年 10 + 3 = 13 岁。
```

### 4. 代码生成

LLM 能编写、解释、调试代码，支持 Python、JavaScript、Java 等主流编程语言。

```
输入：用 Python 写一个函数，判断一个数是否为质数
输出：
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

---

## 2.4 如何调用 LLM API

理论说了不少，现在让我们动手写代码！我们将调用 LLM API，让 AI 回答我们的问题。

### 2.4.1 环境准备

首先，安装 OpenAI 的 Python SDK：

```bash
pip install openai
```

> **说明**：虽然我们叫它"OpenAI SDK"，但它支持所有兼容 OpenAI API 格式的服务，包括 DeepSeek、通义千问等国产模型。

### 2.4.2 第一个 LLM 调用程序

创建一个文件 `chat.py`，写入以下代码：

```python
"""
第2章示例：调用 LLM API
支持 OpenAI / DeepSeek / 通义千问 等兼容接口
"""
import os
from openai import OpenAI

# ============================================================
# 第一步：配置 API 客户端
# ============================================================
# 从环境变量读取配置，如果没有设置则使用默认值
# 你可以通过以下方式设置环境变量：
#   方式1（临时）：export API_KEY="your-api-key"
#   方式2（代码中）：直接替换下面的默认值
client = OpenAI(
    api_key=os.getenv("API_KEY", "your-api-key"),       # 替换为你的 API Key
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),  # API 地址
)

# ============================================================
# 第二步：构建对话消息
# ============================================================
# messages 是一个列表，每个元素是一个字典，包含 role 和 content
# role 有三种：
#   - "system"：系统提示，设定 AI 的角色和行为规则
#   - "user"：用户的消息
#   - "assistant"：AI 的回复（用于多轮对话时提供上下文）
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是AI Agent？用一句话解释。"},
]

# ============================================================
# 第三步：调用 LLM
# ============================================================
response = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),  # 模型名称
    messages=messages,                          # 对话消息列表
    temperature=0.7,                            # 创造性程度（0-2，越高越有创意）
)

# ============================================================
# 第四步：获取并打印结果
# ============================================================
# response.choices 是一个列表，每个元素代表一个可能的回复
# 我们通常只取第一个（choices[0]）
answer = response.choices[0].message.content
print(answer)
```

### 2.4.3 运行方法

```bash
# 方式1：使用环境变量（推荐）
export API_KEY="sk-your-actual-api-key"
export BASE_URL="https://api.deepseek.com/v1"    # DeepSeek 用户
export MODEL="deepseek-chat"                      # DeepSeek 模型名
python chat.py

# 方式2：直接修改代码中的默认值
# 把 "your-api-key" 替换成你的真实 API Key
python chat.py
```

### 2.4.4 预期输出

```
AI Agent是一种能够自主感知环境、制定决策并使用工具来执行任务的智能系统。
```

> **注意**：由于 LLM 的输出具有随机性，你得到的具体文字可能不同，但意思应该相近。

### 2.4.5 不同平台的配置参考

| 平台 | BASE_URL | MODEL | 获取 API Key |
|------|----------|-------|-------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | [platform.deepseek.com](https://platform.deepseek.com) |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | [open.bigmodel.cn](https://open.bigmodel.cn) |

> **省钱小贴士**：如果你只是学习，推荐使用 DeepSeek 或通义千问，价格非常便宜，新用户通常还有免费额度。

---

## 2.5 Prompt Engineering 基础

**Prompt（提示词）** 就是你给 LLM 的"指令"。写好 Prompt 是让 LLM 发挥最大能力的关键技能。这叫做 **Prompt Engineering（提示工程）**。

下面介绍三个最实用的技巧：

### 技巧1：角色设定（System Prompt）

给 LLM 设定一个明确的角色，它的回答会更专业、更精准。

```python
"""
示例：通过 System Prompt 设定角色
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", "your-api-key"),
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
)

# 没有角色设定的回答
response_normal = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {"role": "user", "content": "Python 和 JavaScript 有什么区别？"},
    ],
)
print("【普通回答】")
print(response_normal.choices[0].message.content)
print()

# 有角色设定的回答 —— 设定为资深后端工程师
response_expert = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {
            "role": "system",
            "content": "你是一位有20年经验的资深软件架构师。"
                       "回答时请从技术架构的角度分析，"
                       "使用专业但易懂的语言，并给出实际建议。",
        },
        {"role": "user", "content": "Python 和 JavaScript 有什么区别？"},
    ],
)
print("【专家回答】")
print(response_expert.choices[0].message.content)
```

**效果对比**：没有角色设定时，LLM 会给出一个泛泛的对比；设定角色后，LLM 会从架构师的角度深入分析，给出更有深度的回答。

### 技巧2：少样本提示（Few-shot Prompting）

给 LLM 几个"示例"，让它照着你的格式来回答。这在需要特定输出格式时特别有用。

```python
"""
示例：少样本提示（Few-shot）
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", "your-api-key"),
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {
            "role": "system",
            "content": "你是一个情感分析助手。请判断用户评论的情感倾向，"
                       "只回复「正面」或「负面」。",
        },
        # 提供几个示例（Few-shot）
        {"role": "user", "content": "这个产品太好用了，强烈推荐！"},
        {"role": "assistant", "content": "正面"},
        {"role": "user", "content": "质量很差，退货了，再也不买。"},
        {"role": "assistant", "content": "负面"},
        {"role": "user", "content": "还行吧，中规中矩。"},
        {"role": "assistant", "content": "正面"},
        # 现在让 LLM 判断一条新的评论
        {"role": "user", "content": "包装破损，客服态度也不好，差评！"},
    ],
)

print(response.choices[0].message.content)
# 预期输出：负面
```

**关键点**：通过提供 2-3 个示例，LLM 就能准确理解你想要的输出格式和判断标准。

### 技巧3：思维链（Chain of Thought）

让 LLM "一步一步地想"，而不是直接给出答案。这能显著提升复杂推理任务的准确率。

```python
"""
示例：思维链（Chain of Thought）
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", "your-api-key"),
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
)

question = "一个商店打8折促销，某商品原价250元。小明有该商店的会员卡，"
           "可以额外享受折上9折。请问小明买这件商品需要付多少钱？"

# 不使用思维链 —— 直接回答
response_direct = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {"role": "user", "content": question + "\n请直接给出答案。"},
    ],
)
print("【直接回答】")
print(response_direct.choices[0].message.content)
print()

# 使用思维链 —— 一步一步推理
response_cot = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=[
        {
            "role": "user",
            "content": question + "\n请一步一步地思考，展示计算过程。",
        },
    ],
)
print("【思维链回答】")
print(response_cot.choices[0].message.content)
```

**预期输出**：

```
【思维链回答】
让我们一步一步来计算：

1. 首先，商品原价是 250 元
2. 商店打 8 折，所以折后价格 = 250 × 0.8 = 200 元
3. 小明有会员卡，可以额外享受折上 9 折
4. 最终价格 = 200 × 0.9 = 180 元

所以，小明买这件商品需要付 180 元。
```

> **为什么思维链有效？** 就像我们做数学题一样，写出计算过程比直接写答案更不容易出错。LLM 也是如此，"一步一步想"能让它更好地组织推理过程。

---

## 2.6 多轮对话

在实际应用中，我们通常需要多轮对话。LLM 本身是"无状态"的（它不记得之前的对话），所以我们需要把历史消息一并发送：

```python
"""
示例：多轮对话
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", "your-api-key"),
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
)

# 维护一个消息列表，保存完整的对话历史
messages = [
    {"role": "system", "content": "你是一个编程导师，擅长用简单的语言解释技术概念。"},
]

# 第一轮对话
messages.append({"role": "user", "content": "什么是函数？"})
response = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=messages,
)
assistant_reply = response.choices[0].message.content
print(f"AI: {assistant_reply}\n")

# 把 AI 的回复也加入消息列表（重要！）
messages.append({"role": "assistant", "content": assistant_reply})

# 第二轮对话 —— AI 能记住之前讨论的内容
messages.append({"role": "user", "content": "能给我举一个生活中的例子吗？"})
response = client.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    messages=messages,
)
print(f"AI: {response.choices[0].message.content}")
```

**关键点**：每次调用 API 时，都需要把完整的对话历史（包括 AI 之前的回复）一起发送。这就是为什么它叫"消息列表"而不是"单条消息"。

---

## 2.7 动手练习

现在轮到你了！请完成以下练习：

### 练习1：角色扮演

修改 2.4.2 节的代码，让 LLM 扮演以下角色回答"什么是 AI Agent"：

1. **小学老师**：用小学生能听懂的话解释
2. **科幻小说家**：用充满想象力的语言描述
3. **技术架构师**：从技术角度深入分析

提示：只需要修改 `messages` 中的 `system` 内容即可。

### 练习2：Few-shot 分类器

写一个程序，使用 Few-shot 技巧让 LLM 对新闻进行分类（科技/体育/娱乐/财经）。提供至少 3 个示例，然后让它分类一条新的新闻。

### 练习3：思维链计算器

用思维链技巧，让 LLM 解决一个复杂的数学应用题。对比"直接回答"和"思维链回答"的结果差异。

> **提示**：题目越复杂，思维链的优势越明显。试试这道题：
> "小华有 15 个苹果，他给了小明总数的 1/3，又给了小红剩下的一半，最后又买了 4 个。请问小华现在有几个苹果？"

---

## 2.8 踩坑记录

在实际使用中，你可能会遇到以下常见问题。这里帮你提前"排雷"：

### 坑1：API Key 错误

```
错误信息：AuthenticationError: Incorrect API key provided
```

**原因**：API Key 不正确或未设置。

**解决方法**：
1. 检查 API Key 是否复制完整（注意前后不要有多余的空格）
2. 确认环境变量是否正确设置：`echo $API_KEY`
3. 确认 API Key 是否已激活（有些平台需要充值后才能使用）

### 坑2：模型名称错误

```
错误信息：ModelNotFound: The model 'gpt-4' does not exist
```

**原因**：模型名称拼写错误或该模型不可用。

**解决方法**：
1. 去对应平台的文档查看可用的模型列表
2. 注意模型名称区分大小写
3. 确认你的账号是否有权限使用该模型

### 坑3：网络超时

```
错误信息：APITimeoutError: Request timed out
```

**原因**：网络连接不稳定或 API 服务响应慢。

**解决方法**：
1. 检查网络连接
2. 如果使用代理，确认代理配置正确
3. 在代码中设置更长的超时时间：
```python
client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    timeout=60.0,  # 超时时间设为60秒
)
```

### 坑4：输出为空

```
现象：response.choices[0].message.content 为 None
```

**原因**：可能是模型触发了内容过滤，或者请求参数有问题。

**解决方法**：
1. 检查输入内容是否包含敏感信息
2. 尝试使用不同的模型
3. 检查 `response.choices[0].finish_reason`，如果是 `content_filter` 则说明被过滤了

### 坑5：费用超出预期

**原因**：LLM API 是按"token"计费的，输入和输出都算 token。

**省钱技巧**：
1. 开发调试时使用便宜的小模型（如 `gpt-4o-mini`、`deepseek-chat`）
2. 控制 `max_tokens` 参数，避免生成过长的回复
3. 使用缓存机制，避免重复调用

---

## 2.9 小结

本章我们学习了：

- **LLM 是什么**：读遍海量文字、学会了理解和生成人类语言的超级 AI
- **LLM 的核心能力**：文本生成、理解指令、逻辑推理、代码生成
- **调用 LLM API**：使用 OpenAI 兼容 SDK，一行代码就能调用
- **Prompt Engineering 三大技巧**：
  - 角色设定（System Prompt）：让 AI 扮演专家
  - 少样本提示（Few-shot）：用示例教 AI 格式
  - 思维链（Chain of Thought）：让 AI 一步一步想
- **多轮对话**：通过维护消息列表实现上下文连续
- **常见踩坑**：API Key 错误、模型名称、网络超时等

---

## 下一章预告

现在你已经掌握了 LLM 的使用方法，但这只是 Agent 的"大脑"。一个完整的 Agent 还需要"双手"（工具）、"记忆"和"策略"。

在 **第3章：Agent 的核心组件** 中，我们将：
- 了解 Agent 的四大核心组件：大脑、工具、记忆、规划
- 学习 Function Calling 机制 —— 让 LLM 能调用外部工具
- 理解 ReAct 模式 —— Agent 如何"边想边做"
- 看到一个完整的 Agent 架构是什么样的

下一章见！

---

> 上一篇：[第1章：什么是 AI Agent](../01-what-is-agent/README.md) | 下一篇：[第3章：Agent 的核心组件](../03-agent-components/README.md)
