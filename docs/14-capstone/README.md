# 毕业项目：智能研究助手

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 14.1 为什么要做毕业项目？

如果你一路跟着学到这里，已经掌握了很多单点能力：

- 会写基础 Agent
- 会加工具
- 会做记忆
- 会做规划与反思
- 会用 LangChain / LangGraph
- 会做 RAG
- 会设计多 Agent 协作

但真正决定你有没有"学会"的，不是你是否看懂每一章，而是：

> **你能不能把这些能力组合起来，做出一个完整的 Agent 应用。**

这就是毕业项目的意义。

### 为什么我选"智能研究助手"？

因为它几乎覆盖了 Agent 的全部核心能力，而且非常贴近真实使用场景。

一个像样的研究助手，通常需要：

- 理解用户任务
- 制定搜索计划
- 调用搜索工具
- 读取本地资料
- 必要时接入知识库
- 汇总信息并写成报告
- 自我审查，避免明显错误

这已经不是一个"能聊天的机器人"了，而是一个真正能工作的 Agent。

---

## 14.2 项目目标

我们要做的毕业项目叫：**智能研究助手（Research Assistant Agent）**

### 用户输入

例如：

```text
请帮我调研“LangChain 和 LangGraph 的区别”，输出一份适合技术分享的结构化总结。
```

### 系统输出

输出应该包含：

- 背景介绍
- 关键对比
- 典型适用场景
- 风险和局限
- 最终建议

### 这个项目要覆盖哪些能力？

| 能力 | 是否使用 |
|------|------|
| 基础 LLM 调用 | ✅ |
| 工具调用 | ✅ |
| 规划 | ✅ |
| 反思 | ✅ |
| RAG | ✅（可选扩展） |
| 多 Agent | ✅ |
| 工作流编排 | ✅ |

这就是为什么它适合作为毕业项目：  
**它不是某个单点功能，而是一整套 Agent 系统的缩影。**

---

## 14.3 先别急着写代码：先画架构

很多人一做项目就想直接开写，结果往往越写越乱。

在真正动手之前，我们先把架构想清楚。

### 项目整体结构

```text
用户提问
  ↓
主管 Agent：理解任务、制定研究计划
  ↓
检索 Agent：搜索外部资料 / 读取本地文档
  ↓
写作 Agent：根据资料写初稿
  ↓
审稿 Agent：检查完整性、准确性、结构性
  ├─ 通过 → 输出最终结果
  └─ 不通过 → 返回写作 Agent 修改
```

### 为什么这样拆？

因为研究任务天然包含三种不同工作：

1. **搜集信息**
2. **组织表达**
3. **质量审查**

如果让一个 Agent 全做，当然也能做，但职责会混在一起。  
拆开后，每个角色的目标更清楚，系统也更容易调试。

---

## 14.4 项目目录建议

建议你把项目组织成下面这样：

```text
research-assistant/
├── app.py                  # 主入口
├── agents/
│   ├── planner.py          # 主管/规划 Agent
│   ├── researcher.py       # 检索 Agent
│   ├── writer.py           # 写作 Agent
│   └── reviewer.py         # 审稿 Agent
├── tools/
│   ├── search.py           # 搜索工具
│   ├── file_loader.py      # 本地文件读取
│   └── rag.py              # RAG 检索（可选）
├── workflows/
│   └── graph.py            # LangGraph 工作流定义
├── data/
│   └── sample_docs/        # 你的本地文档
└── outputs/
    └── report.md           # 生成的报告
```

### 为什么要这样拆？

因为毕业项目已经不是单文件脚本了。

如果还把所有逻辑都塞进一个 `app.py`，你很快就会遇到：

- 不好维护
- 不好定位问题
- 不好扩展新功能

项目结构清晰，本身就是工程能力的一部分。

---

## 14.5 第一步：实现工具层

一个研究助手最少需要两类工具：

1. 搜索工具
2. 本地资料读取工具

### 搜索工具示例

```python
# tools/search.py
"""
搜索工具（演示版）
作者：青松与桑叶
"""
import json


def web_search(query: str) -> str:
    """
    模拟网络搜索
    实际项目中可以替换为 Tavily、SerpAPI、DuckDuckGo 等
    """
    mock_results = {
        "LangChain 和 LangGraph 的区别": [
            {
                "title": "LangChain 与 LangGraph 对比",
                "snippet": "LangChain 偏组件编排，LangGraph 偏状态化工作流控制。",
            },
            {
                "title": "什么时候用 LangGraph",
                "snippet": "当任务需要分支、回路、人工介入时，LangGraph 更合适。",
            },
        ],
    }

    results = mock_results.get(query, [
        {
            "title": f"关于 {query} 的检索结果",
            "snippet": "这是一个模拟搜索结果。真实项目请接入外部搜索 API。",
        }
    ])
    return json.dumps(results, ensure_ascii=False, indent=2)
```

### 本地文档读取工具示例

```python
# tools/file_loader.py
import os


def read_local_file(filepath: str) -> str:
    """
    读取本地文件内容
    """
    if not os.path.exists(filepath):
        return f"文件不存在：{filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) > 4000:
        return content[:4000] + "\n\n...（内容过长，已截断）"
    return content
```

### 一个工程上的提醒

毕业项目的工具不要贪多。  
**先保证 2-3 个核心工具真的好用，再考虑加更多能力。**

---

## 14.6 第二步：实现 4 个角色 Agent

我们按前面的架构，做 4 个角色。

### 1. Planner Agent：先想清楚要做什么

它负责：

- 读取用户任务
- 产出研究计划
- 明确需要检索哪些主题

```python
# agents/planner.py
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def planner_agent(task: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名研究任务规划助手。"
                    "请把用户任务拆解为 3-5 个研究步骤，输出结构化计划。"
                ),
            },
            {"role": "user", "content": task},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
```

### 2. Researcher Agent：先找资料

它负责：

- 根据计划调用搜索工具
- 必要时读取本地文档
- 汇总成研究笔记

### 3. Writer Agent：把研究笔记写成可读内容

它负责：

- 把检索结果整理成初稿
- 不直接臆测没有证据的内容

### 4. Reviewer Agent：从读者视角找问题

它负责：

- 看是否偏题
- 看是否有遗漏
- 看结构是否清晰

这里最重要的不是代码，而是你对**角色职责**的定义是否明确。

---

## 14.7 第三步：用 LangGraph 把它们串起来

到这里，你已经有：

- 工具
- 多个 Agent 角色

下一步就是把流程显式编排出来。

### 工作流设计

```text
START
  ↓
planner
  ↓
researcher
  ↓
writer
  ↓
reviewer
  ├─ 通过 → END
  └─ 不通过 → writer
```

### 代码骨架

```python
# workflows/graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.writer import writer_agent
from agents.reviewer import reviewer_agent


class ResearchState(TypedDict):
    task: str
    plan: str
    notes: str
    draft: str
    review: str
    passed: bool


def planner_node(state: ResearchState) -> dict:
    return {"plan": planner_agent(state["task"])}


def researcher_node(state: ResearchState) -> dict:
    return {"notes": researcher_agent(state["task"], state["plan"])}


def writer_node(state: ResearchState) -> dict:
    return {"draft": writer_agent(state["task"], state["notes"])}


def reviewer_node(state: ResearchState) -> dict:
    review, passed = reviewer_agent(state["task"], state["draft"])
    return {"review": review, "passed": passed}


def route_after_review(state: ResearchState) -> str:
    if state["passed"]:
        return END
    return "writer"


builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges("reviewer", route_after_review)

graph = builder.compile()
```

### 这一步最关键的思维转变

你会发现：

- 多 Agent 负责"分工"
- LangGraph 负责"流程控制"

这两者结合起来，才更像一个完整的生产级 Agent 系统。

---

## 14.8 第四步：让项目真的可用，而不是只会跑

这一步很多教程都会跳过，但我认为最重要。

一个毕业项目，不应该只是：

```bash
python demo.py
```

然后输出几行看起来不错的文本。

它至少还应该考虑：

### 1. 输出保存

把最终结果写到 `outputs/report.md`

### 2. 错误处理

比如：

- 搜索失败怎么办
- 某个 Agent 超时怎么办
- 审稿连续两轮都不过怎么办

### 3. 运行日志

打印清楚：

- 当前执行到哪个节点
- 每个 Agent 的输入输出摘要

### 4. 可复现

同一个输入，系统大致应该能稳定输出同类质量的结果。

这就是从"会写 Demo"到"有工程意识"的区别。

---

## 14.9 一个建议：先做 MVP，再慢慢扩展

毕业项目最容易犯的错误，就是一上来想做一个无敌强大的系统。

比如：

- 同时支持网页搜索、PDF、Excel、数据库
- 同时支持多语言输出
- 同时支持 6 个 Agent 协作
- 同时支持 GUI 页面和 API 服务

这样十有八九会把自己做崩。

### 正确的路线

先做最小版本：

#### MVP v1

- 1 个用户输入
- 4 个角色 Agent
- 2 个工具
- 1 条 LangGraph 流程
- 输出一份 Markdown 报告

#### MVP v2

- 加本地知识库
- 加引用来源
- 加审稿回路

#### MVP v3

- 加 Web UI
- 加历史记录
- 加人工确认

### 为什么这么做？

因为毕业项目最重要的，不是功能堆得多，而是：

> **你能清楚地讲出它为什么这么设计、每一层解决了什么问题。**

---

## 14.10 项目验收标准

如果你做完后，想判断自己这个毕业项目算不算合格，可以对照下面这张表。

| 项目 | 合格标准 |
|------|------|
| 能跑通 | 输入问题后，系统能完整走完流程并输出结果 |
| 有分工 | 至少有 3 个职责不同的 Agent |
| 有工具 | 至少接入 2 个真实可用的工具 |
| 有流程 | 使用 LangGraph 或等价流程控制方案 |
| 有回路 | 至少有一个"不过则返回重做"的机制 |
| 有产物 | 最终输出能保存为报告文件 |
| 有解释 | 你能清楚解释每个 Agent 和节点的设计理由 |

### 进阶标准

如果你还想做得更像一个像样作品，可以继续加：

- 来源引用
- 日志与可观测性
- Web 页面
- 多轮对话支持
- 人工确认节点

---

## 14.11 你做完这个项目，真正收获了什么？

表面上，你只是做了一个研究助手。

但本质上，你完成的是一整套 AI Agent 工程训练：

### 你已经掌握了

1. 如何拆任务
2. 如何设计角色
3. 如何组织工具
4. 如何编排流程
5. 如何做反思与回退
6. 如何把系统从 Demo 推向"可展示作品"

### 更重要的是，你会开始拥有判断力

比如你再看一个新的 Agent 项目时，你会下意识地想：

- 它的状态放在哪？
- 这个流程为什么这么设计？
- 这个地方为什么要单 Agent，不是多 Agent？
- 这里适合用 LangGraph 还是普通链？
- 这个 RAG 的检索是不是有问题？

这种判断力，比会背框架 API 更值钱。

---

## 14.12 动手练习

严格来说，这一整章本身就是最大的练习。  
但我还是给你一份更细的落地清单。

### 练习1：做出 MVP 版本

要求：

1. 具备 4 个角色 Agent
2. 至少 2 个工具
3. 用 LangGraph 编排
4. 能输出 Markdown 报告

### 练习2：给报告加"来源引用"

要求：

1. 检索结果保留来源字段
2. 写作 Agent 在文末附上参考来源

### 练习3：做一个失败回退机制

要求：

1. Reviewer Agent 输出是否通过
2. 如果不通过，返回 Writer Agent 重写
3. 最多重试 2 轮，防止无限循环

### 练习4：给你的项目写一页说明文档

要求写清楚：

- 系统架构
- 角色分工
- 工具清单
- 工作流设计
- 已知问题

这个练习很重要，因为它会逼你真正把项目想清楚。

---

## 14.13 结语

如果你一路读到这里并且真的动手做了，恭喜你，你已经不是"看过一些 Agent 教程的人"了。

你已经完成了从：

- 知道什么是 Agent
- 到会手写 Agent
- 再到会用框架、会做 RAG、会做多 Agent、会做工作流

这是一条非常完整、非常扎实的学习路径。

当然，这不是终点。

真正的 Agent 工程还有很多值得继续探索的方向，比如：

- 评测与 Benchmark
- 观测与调试
- 成本控制
- 安全与权限
- 长时任务与持久化
- 生产级部署

但至少现在，你已经拥有了一个非常坚实的起点。

### 最后送你一句我很认同的话

> **学 Agent，最忌讳只会调用框架；最宝贵的是理解原理之后，能自己搭系统。**

如果这个系列真的帮你把这件事走通了，那它就完成了自己的使命。

---

## 本系列总结

| 部分 | 你学到了什么 |
|------|------|
| 认知篇 | Agent 是什么，LLM 是怎么工作的 |
| 构建篇 | 如何手写一个完整 Agent |
| 进阶篇 | 行为工程、框架、RAG、多 Agent |
| 实战篇 | 如何把这些能力组合成完整项目 |

---

> 上一篇：[第13章：多 Agent 协作系统](../13-multi-agent/README.md)
