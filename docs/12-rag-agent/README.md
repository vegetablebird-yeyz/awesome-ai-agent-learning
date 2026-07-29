# 第12章：RAG 知识库 Agent

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 12.1 为什么大模型单靠自己不够？

很多初学者第一次做知识问答时，会下意识地觉得：

> 既然大模型已经很聪明了，直接把问题扔给它不就行了？

这在常识类问题上通常没问题，但一旦场景变成：

- 公司的内部文档
- 某个产品的专有说明书
- 最新的业务数据
- 你上传的 PDF / Word / Markdown

模型马上就会暴露限制。

### 三个根本问题

#### 1. 它不知道你的私有知识

大模型预训练时没有见过你公司的规章制度，也没读过你刚上传的会议纪要。

#### 2. 它的知识可能过时

你问一个最新版本 API 的字段变化，它很可能回答的是旧版本。

#### 3. 它会幻觉

当模型不知道答案时，它经常不是说"我不知道"，而是一本正经地编。

### RAG 的一句话定义

> **RAG（Retrieval-Augmented Generation）就是先检索，再生成。**

不是让模型"凭空想"，而是先给它找参考资料，再让它作答。

---

## 12.2 RAG 的工作流到底长什么样？

很多教程一句话就带过了："先召回文档，再生成答案。"  
这句话没错，但太粗了。

一个完整的 RAG 流程通常包含下面几步：

```text
文档准备
  ↓
文档切分
  ↓
向量化（Embedding）
  ↓
写入向量库
  ↓
用户提问
  ↓
问题向量化
  ↓
相似度检索
  ↓
把检索片段拼进 Prompt
  ↓
LLM 生成最终答案
```

### 用一句更通俗的话解释

RAG 就像考试时允许你翻开资料，但翻资料这件事必须做得足够快、足够准：

- 资料太多翻不到重点 → 检索差
- 摘出来的段落不相关 → 召回差
- 摘出来太长太乱 → Prompt 质量差
- 最后模型乱总结 → 生成阶段差

所以 RAG 从来不是"加个向量库"就结束了，它是一整条链路。

---

## 12.3 文档切分：最容易被低估的步骤

如果只让我选一个最容易被新手忽略、但又最影响效果的点，那一定是：**切分（Chunking）**。

### 为什么不能整篇塞进去？

因为：

1. 文档太长，超出上下文窗口
2. 即使塞得进去，检索粒度也太粗
3. 用户问的是一个局部问题，不需要整篇都进 Prompt

### 切太大有什么问题？

比如一个 chunk 有 3000 字：

- 信息太杂
- 召回虽然"大致相关"，但不够精准
- 真正重要的句子会被埋掉

### 切太小又有什么问题？

比如一个 chunk 只有一句话：

- 上下文不完整
- 模型看不懂前后逻辑
- 非常容易丢失语义

### 一个经验性建议

你可以从下面这个默认配置开始：

| 参数 | 建议值 |
|------|------|
| chunk size | 300 - 800 字或等价 token |
| overlap | 50 - 150 |
| 切分方式 | 优先按标题、段落、语义边界切 |

核心原则只有一句：

> **让每个 chunk 足够完整，能独立表达一个小主题。**

---

## 12.4 Embedding：把文本变成可检索的向量

RAG 能检索，不是因为数据库会"看懂文字"，而是因为我们先把文字变成了向量。

### 什么是 Embedding？

你可以把 Embedding 理解成：

> 把一句话映射成一个高维坐标点。

语义越接近的文本，在向量空间里距离就越近。

比如：

- "怎么重置密码"
- "忘记密码如何找回"

它们文字不完全一样，但语义很接近，所以向量距离也会比较近。

### 为什么不用关键词搜索？

关键词搜索只能看字面匹配。

例如用户问：

```text
如何找回账号？
```

而文档写的是：

```text
用户可通过手机号验证重置登录密码。
```

关键词可能完全对不上，但语义其实高度相关。  
这就是向量检索比传统搜索强的地方。

---

## 12.5 一个最小可运行的 RAG 示例

这一节我们先不接复杂框架，直接做一个最小闭环：

1. 读入本地文本
2. 切分
3. 向量化
4. 存入 FAISS
5. 检索并问答

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

### 准备一份文档

在当前目录创建一个 `knowledge.txt`：

```text
LangGraph 是一个用于构建有状态、多步骤 Agent 工作流的框架。
它特别适合需要分支、回路、人工审批和持久化状态的复杂任务。

LangChain 更像一个组件库，用于快速拼装 Prompt、模型、工具和链。
LangGraph 则更强调工作流控制和状态流转。
```

### 完整代码

```python
# simple_rag_agent.py
"""
最小可运行的 RAG 知识库 Agent
作者：青松与桑叶
"""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_documents() -> list[Document]:
    """
    从本地文本加载文档
    """
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": "knowledge.txt"})]


def build_vectorstore(documents: list[Document]):
    """
    文档切分 + 向量化 + 写入 FAISS
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model="text-embedding-3-small",  # 根据你的服务商调整
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ask_rag(question: str, vectorstore) -> str:
    """
    执行检索并生成回答
    """
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.2,
    )

    prompt = f"""
你是一个知识库问答助手。请严格根据提供的资料回答。
如果资料中没有明确答案，请直接说“资料中没有足够信息”，不要编造。

【资料】
{context}

【问题】
{question}
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    docs = load_documents()
    vectorstore = build_vectorstore(docs)

    question = "LangGraph 和 LangChain 有什么区别？"
    answer = ask_rag(question, vectorstore)

    print("问题：", question)
    print("回答：", answer)
```

### 运行

```bash
python simple_rag_agent.py
```

这个例子虽然简单，但已经包含了 RAG 的核心闭环。

---

## 12.6 为什么很多 RAG Demo 能跑，但不好用？

因为"能跑通"和"效果好"之间，差了很多工程细节。

### 问题1：召回不准

常见原因：

- chunk 切得不合理
- embedding 模型不适合中文或你的场景
- 文档噪音太多
- `k` 取值不合适

### 问题2：召回准了，但回答还是差

常见原因：

- Prompt 没约束，模型开始自由发挥
- 检索到的片段太长，重点被淹没
- 多个片段之间有冲突，模型没处理好

### 问题3：用户觉得它"像知道，其实没答到点上"

这是最常见的业务问题。

它不是完全胡说，但回答不够聚焦。  
这种问题往往和下面几件事有关：

- 查询改写不够好
- 检索只看相似度，不看任务意图
- 没有做 rerank

所以真正好用的 RAG，从来不是一个"上传文档即可问答"的按钮，而是一整套调优过程。

---

## 12.7 RAG Agent 和普通问答 Bot 的区别

这个区别你一定要搞清楚。

### 普通问答 Bot

特点：

- 直接用大模型回答
- 不额外接知识库
- 更依赖模型已有知识

适合：

- 常识问答
- 创意写作
- 日常聊天

### RAG Agent

特点：

- 回答前先检索资料
- 更强调"有依据"
- 更适合私有领域知识

适合：

- 文档问答
- 内部知识助手
- 客服知识库
- 产品手册问答

### 一个关键差别

普通问答 Bot 的核心问题是：

> 模型够不够聪明？

RAG Agent 的核心问题变成了：

> 我能不能把对的资料，在对的时机，拿给模型看？

这就是为什么做 RAG，重点往往不在模型，而在检索链路。

---

## 12.8 如何让 RAG 更可靠？

这里给你一套非常实用的改进方向。

### 1. 明确要求"有依据地回答"

Prompt 里不要只写"请回答问题"，而要明确写：

```text
请严格根据资料回答。
如果资料不足，请明确说明不知道。
不要编造。
```

### 2. 给出引用来源

如果能在输出里标注：

- 来源文件
- 来源段落
- 相关章节

用户信任感会高很多。

### 3. 控制召回条数

不是召回越多越好。

- 太少：容易漏掉关键信息
- 太多：噪音太大

通常从 `k=3` 或 `k=5` 开始试比较稳妥。

### 4. 做结果重排（Rerank）

先粗召回一批候选，再用 rerank 模型重排，效果往往会明显提升。

### 5. 保留结构化元信息

比如：

- 文档标题
- 章节名
- 页码
- 创建时间
- 业务类型

这些元信息在检索和展示时都非常有价值。

---

## 12.9 动手练习

### 练习1：把本章例子换成你自己的文档（难度：简单）

要求：

1. 找一份你熟悉的文档
2. 用同样方式建立最小 RAG
3. 测试至少 5 个问题

观察：

- 哪些问题答得准
- 哪些问题答得不准

### 练习2：调整 chunk 参数（难度：中等）

要求：

分别尝试：

- `chunk_size=200`
- `chunk_size=500`
- `chunk_size=800`

比较三种情况下的回答效果，看看切分粒度到底会怎么影响检索。

### 练习3：给回答加上来源引用（难度：中等）

要求：

1. 把检索出的文档 `metadata["source"]` 拼进最终回答
2. 输出格式类似：

```text
回答：......
参考来源：knowledge.txt
```

### 练习4：加入"不知道"机制（难度：中等）

要求：

如果检索结果相关性很差，或者没有足够片段支持回答，就让系统明确返回：

```text
资料中没有足够信息
```

这一步非常重要，因为它直接关系到系统的可信度。

---

## 12.10 小结

这一章我们把 RAG 从"概念词"拉回了真正的工程现实。

| 知识点 | 核心内容 |
|------|------|
| RAG 定义 | 先检索，再生成 |
| 文档切分 | 决定召回粒度和上下文质量 |
| Embedding | 把文本映射为可检索向量 |
| 向量库 | 存储 chunk 并做相似度搜索 |
| Prompt 约束 | 防止模型脱离资料乱答 |
| RAG Agent | 核心是"把对的资料给模型" |

核心要点：

1. **RAG 不是模型增强，而是知识增强**
2. **切分质量往往比你想象中更重要**
3. **好的 RAG 系统，重点不只是"能检索"，而是"检索得准"**
4. **用户最怕的不是系统说不知道，而是系统瞎编**

---

## 下一章预告

到这里，你已经能做出一个单 Agent 的知识库问答系统了。

但现实任务经常不是一个 Agent 就能优雅完成的。  
比如：

- 一个 Agent 负责检索资料
- 一个 Agent 负责写初稿
- 一个 Agent 负责审查和修正

这就是 **多 Agent 协作系统** 的世界。

在 **第13章：多 Agent 协作系统** 中，我们会继续往上走一层，学习：

- 多 Agent 为什么存在
- 常见协作模式有哪些
- 如何让不同 Agent 分工明确
- 为什么多 Agent 不是越多越好

---

> 上一篇：[第11章：Dify 可视化构建](../11-dify/README.md) | 下一篇：[第13章：多 Agent 协作系统](../13-multi-agent/README.md)
