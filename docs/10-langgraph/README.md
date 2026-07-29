# 第10章：LangGraph 状态机

> 作者：青松与桑叶  
> 难度：进阶  
> 本章目标：用“状态、节点、边”表达可循环、可暂停、可恢复的 Agent 工作流

---

## 10.1 为什么需要一张图

第4章的 ReAct 循环和第7章的规划流程都能用 `for` 与 `if` 编写。流程变长后，状态散落在局部变量中，重试会从头开始，人类也很难在关键步骤插手。

LangGraph 把程序拆成三部分：

- **State**：所有节点共同读写的数据
- **Node**：完成一个小步骤的函数
- **Edge**：决定下一步去哪里的连线

它特别适合循环审查、人机确认、失败恢复和长时间任务。只有固定的 A→B→C 时，普通函数或 LCEL 已经够用。

## 10.2 状态不是聊天记录的别名

状态可以包含消息，也可以包含计数器、审批结果、检索证据和错误信息：

```python
class ReviewState(TypedDict):
    topic: str
    draft: str
    feedback: str
    revision_count: int
    approved: bool
```

节点不要悄悄修改全局变量，而应返回本次更新：

```python
def revise_draft(state: ReviewState) -> dict:
    return {
        "draft": state["draft"] + "例如：先检索，再附来源回答。",
        "revision_count": state["revision_count"] + 1,
    }
```

这种写法让状态变化可以记录、回放和测试。

## 10.3 环境与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
python examples/10_langgraph_review.py
```

示例完全离线，不需要 API Key。完整源码在 [`examples/10_langgraph_review.py`](../../examples/10_langgraph_review.py)。

预期输出：

```text
如何构建可靠的 RAG：先明确输入，再处理数据，最后验证输出。例如：输入问题，检索资料，再附来源回答。
审查：包含例子，可以发布。 修订次数：1
```

## 10.4 搭建第一张图

```python
graph = StateGraph(ReviewState)
graph.add_node("write", write_draft)
graph.add_node("review", review_draft)
graph.add_node("revise", revise_draft)

graph.add_edge(START, "write")
graph.add_edge("write", "review")
graph.add_conditional_edges(
    "review",
    route_after_review,
    {"revise": "revise", "finish": END},
)
graph.add_edge("revise", "review")
app = graph.compile()
```

`START` 和 `END` 是特殊节点。条件边返回的是路由标签，不应直接相信模型生成的任意节点名；把可选路径写成固定映射更安全。

## 10.5 循环必须有出口

审查节点可能永远说“不通过”。示例同时使用 `approved` 和 `revision_count`：

```python
def route_after_review(state: ReviewState) -> str:
    if state["approved"] or state["revision_count"] >= 2:
        return "finish"
    return "revise"
```

生产系统还应设置图级递归上限、每个节点的超时和总费用预算。达到上限不代表任务成功，应返回“需要人工处理”的明确状态。

## 10.6 持久化与会话隔离

checkpointer 能保存每一步状态，让程序中断后继续。使用时要为每个会话提供稳定且不可猜测的 `thread_id`：

```python
config = {"configurable": {"thread_id": "user-42-session-a8f1"}}
result = app.invoke(initial_state, config=config)
```

不要让不同用户复用同一个 ID。持久化状态可能包含隐私或密钥，应加密、设置过期时间，并提供删除能力。

## 10.7 人机协作

适合暂停并等待人的节点包括：

- 发布文章、发消息或提交工单之前
- 删除文件、修改生产数据之前
- 检索证据不足，但任务要求高准确性时

人机协作不是让用户回答含糊的“是否继续”。界面应展示即将执行的动作、目标、关键参数和可预见影响，用户才能作出有效确认。

## 10.8 踩坑记录

### 节点返回完整旧状态

节点通常只需返回变化字段。反复返回整个状态会让合并规则难以理解，消息列表还可能重复。

### 条件边标签不匹配

路由函数返回 `"revise"`，映射里却写 `"revision"`，编译或运行就会失败。路由标签最好使用 `Literal` 或集中常量。

### 把失败当结束

达到最大修订次数只是停止条件，不等于审查通过。状态中应区分 `approved`、`failed` 和 `needs_human`。

### 持久化了不可序列化对象

打开的文件、数据库连接和客户端对象不应放入状态。状态保存普通字符串、数字、列表和字典，资源在节点内部创建或注入。

## 10.9 安全边界

- 图中每个循环都有次数、时间和费用上限
- 条件路由只允许进入预先注册的节点
- 多用户状态用 `thread_id` 隔离，并设置保留期限
- 外部副作用放在人类确认节点之后
- 日志记录状态摘要，不记录 API Key、完整隐私文本或认证头
- 恢复执行时重新检查权限，不能沿用过期授权

## 10.10 痛点：长流程为什么不能只靠 if/for

先看一个“能跑但难维护”的伪代码：

```python
draft = write(topic)
for _ in range(3):
    approved, feedback = review(draft)
    if approved:
        break
    draft = revise(draft, feedback)
publish(draft)
```

代码很短，却隐藏了很多问题：

- 程序崩溃后，应该从写作、审查还是修订恢复
- 循环结束是“已通过”还是“次数用完”
- 两个用户同时运行时，草稿是否会串线
- 发布前需要人类确认时，状态保存在哪里
- 想回放某次错误时，如何知道每一步看到了什么
- 后续增加“证据不足”和“合规拒绝”分支时，嵌套会越来越深

可以把普通流程和 LangGraph 类比为两种导航方式：

```text
普通脚本像口头导航：
  “先直走，看到路口左转，再走一会……”
  中途挂断后，很难知道自己走到了哪里。

LangGraph 像地铁线路图：
  当前站 = State
  车站处理 = Node
  线路 = Edge
  换乘判断 = Conditional Edge
  刷卡记录 = Checkpoint
```

LangGraph 不会自动提高内容质量。它解决的是流程可见性、状态管理、分支循环和恢复执行。

---

## 10.11 原理总图：状态如何在节点之间流动

本章完整案例是一条“写作—审查—修订”工作流：

```text
                         ┌───────────────────────────┐
                         │                           │
                         ▼                           │
START → write → review → route ── revise ───────────┘
                   │       │
                   │       ├── human → END
                   │       │
                   │       └── finish → END
                   │
                   └─ 每个节点读取 State，返回局部更新
```

一次运行的状态变化可能是：

```text
new
 ↓ write
drafted
 ↓ review（缺少例子）
review_failed
 ↓ revise
revised，revision_count=1
 ↓ review（缺少安全边界）
review_failed
 ↓ revise
revised，revision_count=2
 ↓ review
approved
 ↓ END
```

如果 `max_revisions=0`，第一次审查失败后不会修订，而是进入 `human`：

```text
review_failed → human → needs_human → END
```

注意，`END` 只是图停止，不等于业务成功。成功条件应看 `approved`，人工接管应看 `needs_human`。

---

## 10.12 分步代码一：设计可持久化 State

```python
class ReviewState(TypedDict):
    topic: str
    audience: str
    draft: str
    feedback: str
    revision_count: int
    max_revisions: int
    approved: bool
    needs_human: bool
    status: str
```

### 为什么显式列出这些字段

| 字段 | 谁写入 | 谁读取 | 作用 |
|------|--------|--------|------|
| `topic` | 调用方 | write | 原始任务 |
| `audience` | 调用方 | write | 控制内容受众 |
| `draft` | write/revise | review | 当前产物 |
| `feedback` | review/human | revise/界面 | 失败原因 |
| `revision_count` | revise | route | 循环计数 |
| `max_revisions` | 调用方 | route | 安全上限 |
| `approved` | review/human | route/业务 | 成功标志 |
| `needs_human` | human | 业务 | 人工接管标志 |
| `status` | 每个节点 | 日志/界面 | 可观测阶段 |

状态应保存“业务事实”，而不是运行资源。不要放入：

- 模型客户端
- 数据库连接
- 打开的文件对象
- 锁、线程、协程
- 包含密钥的请求头

这些对象要么不可序列化，要么会把凭据带进检查点。

### 初始状态工厂

```python
def initial_state(topic: str, max_revisions: int = 3) -> ReviewState:
    if not topic.strip():
        raise ValueError("topic 不能为空")
    if max_revisions < 0 or max_revisions > 10:
        raise ValueError("max_revisions 必须在 0 到 10 之间")
    return {
        "topic": topic,
        "audience": "Python 初学者",
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "approved": False,
        "needs_human": False,
        "status": "new",
    }
```

集中构造状态有两个好处：调用方不会漏字段，输入边界也只需维护一处。

---

## 10.13 分步代码二：节点只返回局部更新

写作节点：

```python
def write_draft(state: ReviewState) -> dict:
    draft = (
        f"{state['topic']}：面向{state['audience']}，"
        "先明确输入，再处理数据，最后验证输出。"
    )
    return {
        "draft": draft,
        "status": "drafted",
        "feedback": "",
        "approved": False,
        "needs_human": False,
    }
```

节点接收完整状态，但只返回变化字段。LangGraph 会把更新合并回状态。

审查节点：

```python
def review_draft(state: ReviewState) -> dict:
    missing = []
    if "例如" not in state["draft"]:
        missing.append("具体例子")
    if "安全" not in state["draft"]:
        missing.append("安全边界")

    if not missing:
        return {
            "approved": True,
            "feedback": "包含例子和安全边界，可以发布。",
            "status": "approved",
        }
    return {
        "approved": False,
        "feedback": "请补充：" + "、".join(missing),
        "status": "review_failed",
    }
```

示例故意使用确定规则而不是 LLM，这样不配置 API Key 也能观察图结构。真实项目可以把节点内部替换为模型调用，但节点的输入输出契约不变。

修订节点：

```python
def revise_draft(state: ReviewState) -> dict:
    revised = state["draft"]
    if "具体例子" in state["feedback"] and "例如" not in revised:
        revised += "例如：输入问题，检索资料，再附来源回答。"
    elif "安全边界" in state["feedback"] and "安全" not in revised:
        revised += "安全要求：证据不足时拒答，并对日志脱敏。"
    return {
        "draft": revised,
        "revision_count": state["revision_count"] + 1,
        "status": "revised",
    }
```

计数器必须在真正发生修订时增加。如果在 `review` 增加，会把“审查次数”和“修订次数”混为一谈。

---

## 10.14 分步代码三：条件分支与循环出口

```python
def route_after_review(
    state: ReviewState,
) -> Literal["revise", "human", "finish"]:
    if state["approved"]:
        return "finish"
    if state["revision_count"] >= state["max_revisions"]:
        return "human"
    return "revise"
```

判断顺序很重要：

1. 已通过时立即结束
2. 未通过且达到上限时转人工
3. 未通过且还有预算时继续修订

注册固定映射：

```python
graph.add_conditional_edges(
    "review",
    route_after_review,
    {
        "revise": "revise",
        "human": "human",
        "finish": END,
    },
)
graph.add_edge("revise", "review")
```

路由函数返回标签，映射决定真实节点。不要把模型输出直接当节点名，否则提示注入或幻觉可能把流程导向未授权路径。

### 三层停止机制

可靠工作流至少有三层限制：

```text
业务层：max_revisions
图运行层：recursion_limit
节点资源层：timeout / token budget / cost budget
```

业务层给用户可理解的结果；图运行层防止意外环路；资源层避免单节点耗尽系统。

---

## 10.15 分步代码四：检查点与 thread_id

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = build_graph(checkpointer=memory)

config = {
    "configurable": {"thread_id": "tutorial-a8f1"},
    "recursion_limit": 20,
}
result = app.invoke(initial_state("如何构建可靠的 RAG"), config=config)
```

### Checkpointer 保存什么

检查点通常记录每个超级步骤之后的状态、下一节点和相关元数据。可以读取当前快照：

```python
snapshot = app.get_state(config)
print(snapshot.values["status"])
print(snapshot.next)
```

也可以查看历史：

```python
history = list(app.get_state_history(config))
for item in history:
    print(item.values.get("status"), item.next)
```

`MemorySaver` 只适合教程和单进程测试，进程退出后数据消失。生产环境要使用持久化后端，并设计：

- 数据库连接池
- 加密与访问控制
- 保留期限
- 用户删除接口
- 备份与恢复
- 旧状态 Schema 迁移

### thread_id 为什么不能写死

如果所有请求都使用 `"default"`，不同用户会共享同一条状态历史。推荐由后端生成不可猜测 ID，并在访问检查点前验证当前用户是否拥有该会话。

```text
错误：thread_id = user_input
问题：可猜测，可能读取别人的会话

正确：thread_id = 服务端生成的随机 ID
同时保存：owner_user_id → thread_id 的授权关系
```

---

## 10.16 完整案例逐段解释

完整源码位于：

```text
examples/10_langgraph_review.py
```

它分成六层：

1. `ReviewState`：定义节点共享数据
2. `write/review/revise/human`：四个职责单一的节点
3. `route_after_review`：显式条件分支
4. `build_graph`：只负责拓扑
5. `initial_state`：输入验证与默认值
6. `run_checkpoint_demo`：会话隔离和历史读取

### 为什么专门有 human 节点

达到上限后直接 `END` 会丢失“为什么结束”。`mark_for_human` 写入：

```python
{
    "needs_human": True,
    "approved": False,
    "status": "needs_human",
    "feedback": "已达到修订上限，请人工审查",
}
```

下游系统可以据此创建待办，而不会把未通过草稿自动发布。

### 两个会话如何证明隔离

示例为 A、B 分别生成 `thread_id`：

```text
会话 A：max_revisions=3 → 自动修订后通过
会话 B：max_revisions=0 → 直接转人工
```

最后比较两个结果的 `topic`。这只是最小演示，正式测试还应并发运行多个线程并验证各自历史。

---

## 10.17 运行与验收

### 运行

```bash
python -m pip install -r examples/requirements.txt
python examples/10_langgraph_review.py
```

不需要 `.env` 或模型 Key。

### 预期关键输出

```text
=== 会话 A ===
状态：approved
修订次数：2
通过：True
需要人工：False
检查点状态：approved

=== 会话 B（修订上限为 0）===
状态：needs_human
修订次数：0
通过：False
需要人工：True
```

### 验收清单

- [ ] 会话 A 至少经历一次 `revise → review` 循环
- [ ] 会话 A 最终 `approved=True`
- [ ] 会话 B 不修订，直接 `needs_human=True`
- [ ] 两个会话使用不同 `thread_id`
- [ ] `get_state` 能读取最终状态
- [ ] `get_state_history` 返回多个检查点
- [ ] 将 `recursion_limit` 调小后能阻止无限图运行
- [ ] 状态中没有客户端、连接、密钥或文件句柄

---

## 10.18 更多踩坑记录

### 坑1：把状态当可变全局字典

节点直接修改 `state["draft"]` 可能在简单测试中工作，但会让更新边界和并发行为难以推理。返回局部更新更清楚。

### 坑2：每次请求重新创建 MemorySaver

如果 checkpointer 随请求销毁，后续自然找不到历史。应用生命周期和检查点后端生命周期要匹配。

### 坑3：恢复时重复执行副作用

检查点可能停在“请求已发出但结果未写回”的边界。发送邮件、创建订单等节点需要幂等键，并在恢复前查询操作状态。

### 坑4：修改 State Schema 后旧检查点无法读取

新增必填字段会让旧状态缺值。为状态设计版本号、默认值和迁移函数，发布前用旧快照做兼容测试。

### 坑5：只看最终状态，不看路径

偶尔成功不代表图正确。测试应断言关键节点顺序、分支选择和循环次数，而不只是最终文本。

### 坑6：把 recursion_limit 当业务成功标准

触发递归限制是保护机制生效，表示流程异常或预算不足。应捕获并转成失败/人工处理，不是发布结果。

---

## 10.19 动手练习

### 练习1：增加“拒绝发布”分支（简单）

当主题包含敏感占位词时，路由到 `rejected` 节点，并在状态里保存拒绝原因。

### 练习2：记录节点轨迹（中等）

给状态添加 `trace` 列表。使用合适的 reducer 追加节点名，避免每个节点覆盖旧轨迹。

### 练习3：模拟节点失败与重试（中等）

让写作节点第一次抛出异常，验证检查点历史，并设计有限重试。注意不要对副作用节点盲目重试。

### 练习4：增加人工修改后恢复（困难）

在人工节点暂停，接受人类修改的草稿，再从审查节点恢复。恢复前重新检查当前用户权限。

### 练习5：替换生产 Checkpointer（困难）

选择适合的持久化后端，实现会话过期和删除，并编写两个用户不能互读状态的集成测试。

## 10.20 思考题与答案

### 题1：什么时候普通函数比 LangGraph 更合适

**答案**：流程短、无循环、无需暂停恢复时，普通函数更直接，也减少框架依赖。

### 题2：为什么状态里不应保存模型客户端

**答案**：客户端通常不可序列化，还包含连接和认证信息。它会破坏持久化，并可能让密钥进入状态存储。

### 题3：最大循环次数到了，能否把结果标成成功

**答案**：不能。它只能说明系统停止了。应检查业务完成条件，否则标成 `needs_human` 或失败。

### 题4：人类确认节点应展示什么

**答案**：展示动作类型、目标对象、关键参数、影响范围和取消选项，而不是只有一个脱离上下文的“继续”按钮。

### 题5：为什么 `thread_id` 既要唯一又要鉴权

**答案**：唯一只能避免无意串线，不能阻止越权读取。攻击者若猜到别人的 ID，仍可能访问状态，因此后端必须验证会话所有者。

### 题6：检查点是否等同于业务数据库

**答案**：不是。检查点服务于流程恢复，业务数据库保存订单、用户等权威事实。两者生命周期和一致性要求不同，不能把检查点当唯一业务真相。

## 10.21 小结

LangGraph 让循环、分支和状态变化成为显式结构。可靠的图不只要“能跑”，还要能停止、能恢复、能隔离用户，并在高风险动作前把控制权交还给人。

---

> 上一章：[第9章：LangChain 快速上手](../09-langchain/README.md)  
> 下一章：[第11章：Dify 可视化构建](../11-dify/README.md)
