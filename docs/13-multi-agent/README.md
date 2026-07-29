# 第13章：多 Agent 协作系统

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 13.1 为什么会出现多 Agent？

学到这里，很多人都会自然冒出一个想法：

> 既然一个 Agent 已经能规划、检索、用工具、反思，那为什么还要多个 Agent？

这是个好问题。  
答案不是"因为更高级"，而是因为**有些任务天然适合分工**。

### 一个单 Agent 容易遇到的问题

当一个 Agent 既要：

- 理解任务
- 搜集资料
- 写内容
- 检查质量
- 决定下一步

它很容易出现下面这些问题：

1. **角色冲突**：自己写、自己夸、自己审，很难真正挑毛病
2. **上下文过载**：做的事情太多，历史对话越来越长
3. **决策混乱**：检索、规划、写作、审查混在一起
4. **错误放大**：前面的小偏差会一路传递下去

### 多 Agent 的核心价值

多 Agent 的本质，不是"搞很多模型一起跑"，而是：

> **把不同职责拆开，让每个 Agent 专注做一件事。**

比如：

- 研究员 Agent：负责找资料
- 写作者 Agent：负责整理成稿
- 审稿 Agent：负责找问题

这就像一个小团队，而不是一个人同时扮演所有角色。

---

## 13.2 多 Agent 不是什么？

这里我要先泼一盆冷水：**多 Agent 不是万能升级包。**

很多人一做 Agent 就想上多 Agent，理由通常是：

- 听起来更高级
- 架构图更酷
- 论文里经常这么画

但真实情况是：

### 单 Agent 足够的场景

- 简单问答
- 单轮工具调用
- 小型 RAG
- 固定模板生成

这些场景，单 Agent 通常更直接、更快、更便宜。

### 什么时候真的该上多 Agent？

当你出现这些信号时，再认真考虑：

1. **任务可以自然拆成多个角色**
2. **不同阶段需要明显不同的思维方式**
3. **一个 Agent 处理全流程时质量明显下降**
4. **你确实需要交叉审查或分工协作**

一句话总结：

> **多 Agent 应该是为了解决问题，而不是为了显得架构先进。**

---

## 13.3 最常见的三种多 Agent 模式

### 模式一：主管 - 执行者（Supervisor-Worker）

这是最常见、也最容易理解的一种。

```text
主 Agent 负责理解任务和分派工作
  ↓
子 Agent 负责执行具体子任务
  ↓
主 Agent 汇总结果
```

适合：

- 任务拆解明确
- 主流程由一个中心控制
- 需要统一汇总输出

例如：

- 主 Agent 拆任务
- 检索 Agent 查资料
- 计算 Agent 做数据处理
- 写作 Agent 输出答案

### 模式二：流水线（Pipeline）

这种更像工厂装配线：

```text
Agent A 做完
  ↓
结果交给 Agent B
  ↓
再交给 Agent C
```

适合：

- 流程顺序固定
- 每一步职责非常明确

例如：

- 提取关键信息
- 改写成结构化要点
- 润色成正式报告

### 模式三：辩论 / 审查（Debate / Review）

这种模式的重点不是分工，而是**交叉检验**。

例如：

- Agent A 先给方案
- Agent B 负责挑错
- Agent C 做最终仲裁

适合：

- 高风险决策
- 需要高准确性
- 希望减少单点幻觉

---

## 13.4 一个最小可运行的多 Agent 示例

这一节我们写一个最简单的"研究员 + 写作者 + 审稿人"系统。

### 目标

用户提一个问题后：

1. 研究员 Agent 先列出要点
2. 写作者 Agent 根据要点写初稿
3. 审稿 Agent 检查初稿的问题
4. 最终输出"初稿 + 审稿意见"

### 代码示例

```python
# multi_agent_demo.py
"""
最小多 Agent 协作示例
作者：青松与桑叶
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    通用 LLM 调用函数
    """
    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def researcher_agent(question: str) -> str:
    """
    研究员 Agent：提炼问题相关要点
    """
    system_prompt = (
        "你是一名研究员。"
        "请围绕用户问题提炼出清晰、结构化的研究要点。"
        "只输出要点，不写完整文章。"
    )
    return call_llm(system_prompt, question)


def writer_agent(question: str, notes: str) -> str:
    """
    写作者 Agent：根据研究要点写出初稿
    """
    system_prompt = (
        "你是一名技术写作者。"
        "请根据研究要点写一份条理清晰、通俗易懂的回答。"
    )
    user_prompt = f"问题：{question}\n\n研究要点：\n{notes}"
    return call_llm(system_prompt, user_prompt)


def reviewer_agent(question: str, draft: str) -> str:
    """
    审稿 Agent：检查初稿的问题
    """
    system_prompt = (
        "你是一名严格的审稿人。"
        "请检查回答是否存在以下问题：\n"
        "1. 是否偏题\n"
        "2. 是否有遗漏\n"
        "3. 是否逻辑不清\n"
        "4. 是否存在明显事实风险\n"
        "请给出简洁明确的审稿意见。"
    )
    user_prompt = f"问题：{question}\n\n待审稿内容：\n{draft}"
    return call_llm(system_prompt, user_prompt)


def run_multi_agent(question: str):
    """
    多 Agent 主流程
    """
    print("=" * 60)
    print("用户问题：", question)
    print("=" * 60)

    print("\n【研究员 Agent】")
    notes = researcher_agent(question)
    print(notes)

    print("\n【写作者 Agent】")
    draft = writer_agent(question, notes)
    print(draft)

    print("\n【审稿 Agent】")
    review = reviewer_agent(question, draft)
    print(review)

    return {
        "notes": notes,
        "draft": draft,
        "review": review,
    }


if __name__ == "__main__":
    run_multi_agent("请解释为什么很多 AI Agent 项目最终又回到了单 Agent + 工作流的设计。")
```

### 这个例子最重要的点是什么？

不是它有多复杂，而是你开始意识到：

- 每个 Agent 都有自己的角色提示词
- 每个 Agent 都只处理自己该处理的输入
- 主流程负责调度，而不是让所有角色混在一个对话里打架

---

## 13.5 多 Agent 设计的第一原则：角色边界要清晰

如果你真的想把多 Agent 做好，第一原则不是"多"，而是"边界清楚"。

### 一个坏例子

假设你设计了 3 个 Agent：

- Agent A：负责分析
- Agent B：负责综合分析
- Agent C：负责深度综合分析

看起来有 3 个角色，其实完全重叠。  
最后只会出现：

- 大量重复劳动
- 输出风格高度相似
- 协作价值很低

### 一个好例子

- 检索 Agent：只负责找资料
- 写作 Agent：只负责写内容
- 审稿 Agent：只负责挑问题

这三个角色天然互补，边界非常清楚。

### 一个很实用的判断标准

如果你删掉其中一个 Agent，系统会失去一项明确能力，那说明角色设计合理。  
如果删掉之后没什么区别，那这个 Agent 很可能只是"凑数"。

---

## 13.6 多 Agent 的通信方式

多 Agent 不是放几个名字就完事了，它们之间必须交换信息。

最常见的通信方式有两种：

### 方式一：结构化消息传递

比如传 JSON：

```python
{
    "task": "写一份总结",
    "constraints": ["简洁", "专业"],
    "research_notes": [...]
}
```

优点：

- 清晰
- 好解析
- 不容易丢信息

### 方式二：自然语言传递

比如：

```text
以下是研究员整理出的要点，请基于这些内容写成一份正式说明。
```

优点：

- 灵活
- 快速
- 适合小系统

### 实际建议

- 小项目、Demo：自然语言就够了
- 严肃项目、复杂系统：尽量结构化

因为系统一复杂，自然语言传递很容易出现：

- 信息丢失
- 约束失效
- 上下游理解不一致

---

## 13.7 多 Agent 最大的坑：成本和复杂度暴涨

这是我特别想强调的一点。

### 为什么多 Agent 不是越多越好？

因为每多一个 Agent，就意味着：

1. 多一次模型调用
2. 多一份 Prompt 维护成本
3. 多一段上下游协议设计
4. 多一个错误传播点

### 常见问题

#### 1. 成本上升

本来一个问题只调用一次模型，现在可能调用三次、五次，甚至更多。

#### 2. 延迟上升

每个 Agent 都要思考、输出、传递，系统自然会变慢。

#### 3. 调试变难

出了问题时，你要排查：

- 是研究员没找对资料？
- 是写作者误解了资料？
- 还是审稿人没指出问题？

#### 4. 责任边界模糊

如果设计不好，很容易演变成：

> Agent A 以为 Agent B 会做  
> Agent B 以为 Agent C 已经做了  
> 最后谁都没做

### 一个很现实的原则

> **能用单 Agent 解决的问题，不要上多 Agent。**

只有当单 Agent 明显扛不住，或者任务天然适合分工时，再考虑多 Agent。

---

## 13.8 多 Agent 和 LangGraph 怎么结合？

这是一个非常常见的实践方向。

你可以把：

- **LangGraph** 理解为流程骨架
- **多 Agent** 理解为节点里的角色分工

例如：

```text
START
  ↓
规划节点（主管 Agent）
  ↓
检索节点（检索 Agent）
  ↓
写作节点（写作者 Agent）
  ↓
审查节点（审稿 Agent）
  ├─ 通过 → END
  └─ 不通过 → 回到写作节点
```

这种组合非常自然，因为：

- LangGraph 负责控制流程
- 多 Agent 负责在每个阶段扮演不同角色

这通常比"让所有 Agent 自由聊天协商"更稳定，也更容易调试。

---

## 13.9 动手练习

### 练习1：做一个"研究员 + 写作者"双 Agent（难度：简单）

要求：

1. 用户给问题
2. 研究员先列要点
3. 写作者根据要点生成回答

目标：

感受多 Agent 的最小分工闭环。

### 练习2：加一个审稿 Agent（难度：中等）

要求：

1. 在上面的基础上新增审稿 Agent
2. 审稿 Agent 给出至少 3 条审稿意见
3. 你手动判断这些意见是否有价值

这个练习很重要，因为你会开始意识到：  
**不是加了审稿 Agent，质量就一定提升。关键在于审稿角色是否设计得足够严格。**

### 练习3：用 LangGraph 编排三个 Agent（难度：中等）

要求：

1. 用 LangGraph 把三类 Agent 串起来
2. 增加一个条件分支：
   - 审稿通过 → 结束
   - 审稿不通过 → 返回写作者重写

---

## 13.10 小结

这一章我们终于把多 Agent 从"听起来很酷的概念"变成了可以落地理解的系统设计。

| 知识点 | 核心内容 |
|------|------|
| 多 Agent 的价值 | 通过分工减少角色冲突和上下文过载 |
| 常见模式 | 主管-执行者、流水线、辩论/审查 |
| 设计原则 | 角色边界清晰，比数量更重要 |
| 通信方式 | 小系统可自然语言，大系统更适合结构化协议 |
| 主要代价 | 成本、延迟、调试复杂度都会上升 |

核心要点：

1. **多 Agent 的本质是分工，不是堆模型调用**
2. **角色边界越清晰，协作越有效**
3. **多 Agent 不等于更高级，很多场景单 Agent 反而更优**
4. **最稳妥的组合通常是：LangGraph 控流程，多 Agent 做分工**

---

## 下一章预告

恭喜你，走到这里，整套学习路线已经接近收尾了。

接下来我们不再学新的零散概念，而是进入最后一章：  
把前面所有能力真正串起来，做一个完整的毕业项目。

在 **第14章：智能研究助手** 中，我们会综合使用：

- Agent 规划
- 工具调用
- RAG 知识库
- 多 Agent 分工
- 工作流编排

最终做出一个真正像样、可以展示给别人看的 Agent 应用。

---

> 上一篇：[第12章：RAG 知识库 Agent](../12-rag-agent/README.md) | 下一篇：[第14章：智能研究助手](../14-capstone/README.md)
