# 第10章：LangGraph 状态机

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 10.1 为什么 LangChain 还不够？

上一章我们用 LangChain 很快搭好了一个 Agent。对于简单任务，这已经很好用了。

但一旦任务变复杂，你很快就会遇到这些问题：

- 一个任务要拆成多个阶段，执行顺序要严格控制
- 某一步失败后，不能简单报错退出，而是要回退重试
- 某些步骤要根据运行结果走不同分支
- 中间还可能插入人工确认

这时候你会发现，单纯靠 `AgentExecutor` 已经不够优雅了。你需要一个更明确的"流程骨架"。

### LangGraph 的定位

LangGraph 可以理解为：**专门为 Agent 工作流设计的状态机框架。**

如果说：

- LangChain 像一盒乐高积木
- 那 LangGraph 就像一张"施工蓝图"

它不只是告诉你"有哪些积木可以用"，而是让你明确：

- 当前处于哪个阶段
- 下一步该去哪里
- 什么条件下回退
- 哪些状态要被保留

---

## 10.2 什么是状态机？

别被"状态机"这个词吓到。它其实是一种非常朴素的思想：

> **系统在任意时刻都处于某个状态，并且会根据条件从一个状态转移到另一个状态。**

### 先看一个简单例子

假设我们要做一个"研究助手"：

```text
用户提问
  ↓
制定搜索计划
  ↓
执行搜索
  ↓
整理答案
  ↓
输出结果
```

如果搜索结果不够怎么办？

```text
执行搜索
  ↓
检查结果是否足够
  ├─ 足够 → 整理答案
  └─ 不足 → 回到制定搜索计划
```

这就是最典型的状态机。

### 为什么 Agent 特别适合状态机？

因为 Agent 天生就是"分阶段推进"的系统：

| Agent 行为 | 状态机视角 |
|------|------|
| 理解任务 | 初始状态 |
| 制定计划 | 规划状态 |
| 调用工具 | 执行状态 |
| 检查结果 | 校验状态 |
| 重试或结束 | 分支状态 |

前面你手写 ReAct 循环的时候，其实已经在手写一个"隐式状态机"了。LangGraph 只是把这件事显式化、工程化。

---

## 10.3 LangGraph 的核心概念

LangGraph 有 3 个最核心的概念：

### 1. State：状态

状态就是整个工作流共享的数据。

比如：

```python
{
    "question": "帮我总结 LangGraph 的作用",
    "plan": ["先解释概念", "再给代码示例"],
    "search_results": "...",
    "final_answer": "..."
}
```

### 2. Node：节点

节点就是一个步骤。它通常是一个函数，接收状态，返回对状态的更新。

例如：

- `make_plan`
- `search_docs`
- `write_answer`

### 3. Edge：边

边表示节点之间怎么连接。

分两种：

- **普通边**：执行完 A，一定去 B
- **条件边**：执行完 A，根据结果决定去 B 还是 C

你可以把 LangGraph 想成：

```text
状态 = 公共数据
节点 = 处理步骤
边 = 步骤之间的路由规则
```

---

## 10.4 第一个 LangGraph 工作流

我们先写一个最简单的例子：用户提问后，先生成计划，再输出答案。

### 安装依赖

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

### 最小示例

```python
# hello_langgraph.py
"""
第一个 LangGraph 工作流
作者：青松与桑叶
"""
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.3,
)


class AgentState(TypedDict):
    question: str
    plan: str
    answer: str


def make_plan(state: AgentState) -> dict:
    """
    第一步：为用户问题生成一个简短计划
    """
    question = state["question"]
    response = llm.invoke(
        f"请为这个问题生成一个两步的回答计划：{question}"
    )
    return {"plan": response.content}


def write_answer(state: AgentState) -> dict:
    """
    第二步：根据计划生成最终答案
    """
    question = state["question"]
    plan = state["plan"]
    response = llm.invoke(
        f"用户问题：{question}\n\n回答计划：{plan}\n\n请根据计划给出最终回答。"
    )
    return {"answer": response.content}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("make_plan", make_plan)
graph_builder.add_node("write_answer", write_answer)

graph_builder.add_edge(START, "make_plan")
graph_builder.add_edge("make_plan", "write_answer")
graph_builder.add_edge("write_answer", END)

graph = graph_builder.compile()

result = graph.invoke({
    "question": "请通俗解释什么是 LangGraph",
    "plan": "",
    "answer": "",
})

print("计划：")
print(result["plan"])
print("\n最终答案：")
print(result["answer"])
```

### 这个例子最重要的地方是什么？

不是代码本身有多复杂，而是你要开始建立一种新的思维：

- LangChain 更像"函数组合"
- LangGraph 更像"工作流编排"

当你的 Agent 需要明确的阶段控制时，LangGraph 就会非常顺手。

---

## 10.5 给工作流加上条件分支

真正体现 LangGraph 价值的，是条件分支。

我们来做一个简单场景：

1. 先搜索资料
2. 再检查资料够不够
3. 如果不够，就继续补搜
4. 如果够了，就写答案

### 代码示例

```python
# branching_langgraph.py
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.2,
)


class ResearchState(TypedDict):
    question: str
    search_results: str
    is_enough: bool
    answer: str


def search_once(state: ResearchState) -> dict:
    question = state["question"]
    response = llm.invoke(
        f"请模拟搜索并返回关于这个问题的简短资料：{question}"
    )
    return {"search_results": response.content}


def judge_results(state: ResearchState) -> dict:
    search_results = state["search_results"]
    response = llm.invoke(
        "请判断下面这份资料是否足够回答问题。"
        "如果足够，只返回 yes；如果不足，只返回 no。\n\n"
        f"{search_results}"
    )
    decision = response.content.strip().lower()
    return {"is_enough": decision == "yes"}


def write_answer(state: ResearchState) -> dict:
    question = state["question"]
    search_results = state["search_results"]
    response = llm.invoke(
        f"问题：{question}\n\n资料：{search_results}\n\n请生成最终回答。"
    )
    return {"answer": response.content}


def route_after_judge(state: ResearchState) -> str:
    if state["is_enough"]:
        return "write_answer"
    return "search_once"


builder = StateGraph(ResearchState)
builder.add_node("search_once", search_once)
builder.add_node("judge_results", judge_results)
builder.add_node("write_answer", write_answer)

builder.add_edge(START, "search_once")
builder.add_edge("search_once", "judge_results")
builder.add_conditional_edges("judge_results", route_after_judge)
builder.add_edge("write_answer", END)

graph = builder.compile()

result = graph.invoke({
    "question": "LangGraph 和 LangChain 有什么区别？",
    "search_results": "",
    "is_enough": False,
    "answer": "",
})

print(result["answer"])
```

### 你要注意什么？

`route_after_judge()` 这个函数非常关键。它决定了执行完 `judge_results` 之后，下一步去哪里。

这就是 LangGraph 和普通链式调用最大的区别：  
**它允许你的 Agent 有"回路"和"分支"。**

---

## 10.6 LangGraph 最适合什么场景？

不是所有项目都需要 LangGraph。它最适合下面这些场景：

### 场景1：多阶段工作流

例如：

- 先分析需求
- 再生成计划
- 再调用工具
- 再审查结果
- 最后输出答案

### 场景2：需要失败重试

例如：

- 调用搜索 API 失败就重试
- 检查结果不够就补搜
- 生成答案后再做反思修正

### 场景3：需要人工介入

例如：

- 生成方案后需要用户点击确认
- 审核不过要退回修改

### 场景4：需要持久化状态

例如：

- 多轮任务不能丢上下文
- 长时间运行任务需要断点恢复

### 不适合的场景

如果只是：

- 简单问答
- 单次摘要
- 固定模板生成
- 一步到位的小工具调用

那用 LangGraph 反而有点重了。  
这时候普通 LangChain Chain 或手写逻辑就够了。

---

## 10.7 Human-in-the-Loop：把人接进流程

这是 LangGraph 很实用的一个方向。

有些任务，Agent 不能自己拍板，比如：

- 删除数据库记录
- 执行线上发布
- 给客户发正式邮件
- 修改财务数据

这时候最稳妥的做法，不是让 Agent 自主完成，而是：

```text
Agent 先生成方案
    ↓
等待人工确认
    ↓
确认通过后继续执行
```

### 一个简化的思路

你可以把状态里加一个字段：

```python
class ApprovalState(TypedDict):
    task: str
    draft: str
    approved: bool
```

然后在路由里判断：

- `approved == True` → 继续执行
- `approved == False` → 先停下来

在真实项目里，这种人工介入节点非常常见。  
很多"生产级 Agent"之所以可靠，不是因为它完全自动化，而是因为它知道**什么时候必须让人来拍板**。

---

## 10.8 LangGraph 和我们前面学的规划/反思有什么关系？

其实关系非常紧密。

你在第7章学的：

- `Plan-and-Solve`
- `Reflection`

放到 LangGraph 里，几乎就是天然的节点设计：

```text
用户问题
  ↓
规划节点
  ↓
执行节点
  ↓
反思节点
  ├─ 通过 → 输出结果
  └─ 不通过 → 回到执行节点
```

这也是为什么我建议你先学手写，再学 LangGraph。

如果你没有前面那套认知，看到 LangGraph 只会觉得它是"一个 API 很多的框架"；  
但你现在会意识到：**它其实是在把你已经理解的 Agent 模式工程化。**

---

## 10.9 动手练习

### 练习1：做一个"规划 → 执行 → 总结"图（难度：简单）

要求：

1. 定义一个 `State`
2. 至少包含 3 个节点
3. 从 `START` 开始，到 `END` 结束

建议节点：

- `make_plan`
- `execute_task`
- `summarize`

### 练习2：加一个反思回路（难度：中等）

要求：

1. 新增 `reflect` 节点
2. 如果反思认为答案不好，就返回 `execute_task`
3. 如果反思认为答案合格，就进入 `END`

提示：

```python
def route_after_reflect(state):
    if state["is_good"]:
        return END
    return "execute_task"
```

### 练习3：加一个人工确认节点（难度：中等）

场景：

Agent 先写一份邮件草稿，再等待用户确认，确认后才正式发送。

要求：

1. 状态中加入 `approved`
2. 路由函数根据 `approved` 决定是否继续
3. 先用手动修改状态的方式模拟人工确认

---

## 10.10 小结

这一章你学到的，不只是一个新框架，而是一种更清晰的 Agent 设计方式。

| 知识点 | 核心内容 |
|------|------|
| `State` | 工作流共享数据 |
| `Node` | 每个处理步骤 |
| `Edge` | 节点之间的流转关系 |
| 条件边 | 根据状态决定下一步去哪 |
| 回路 | 支持重试、反思、补搜等模式 |
| Human-in-the-Loop | 关键节点接入人工确认 |

核心要点：

1. **LangGraph 本质上是 Agent 工作流状态机**
2. **它特别适合多阶段、可回退、可分支的复杂任务**
3. **前面学的规划和反思，都能自然映射为图节点**
4. **复杂系统不是靠"一个超级 Prompt"解决的，而是靠清晰的流程结构**

---

## 下一章预告

前两章你已经体验了两条主线：

- 用 LangChain 快速拼装组件
- 用 LangGraph 构建复杂工作流

接下来，我们换一个视角：  
如果你不是想手写代码，而是想更快把一个 Agent 产品搭起来，有没有低代码方案？

答案就是 **Dify**。

在 **第11章：Dify 可视化构建** 中，我们会学习：

- Dify 的核心概念
- 如何可视化搭建 Agent 和 Workflow
- 它适合什么场景，不适合什么场景
- 代码框架和低代码平台该怎么选

---

> 上一篇：[第9章：LangChain 快速上手](../09-langchain/README.md) | 下一篇：[第11章：Dify 可视化构建](../11-dify/README.md)
