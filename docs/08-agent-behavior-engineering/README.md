# 第8章：Agent 行为工程 —— 如何让 Agent 可靠地遵循指令

> 作者：青松与桑叶
> 本系列教程定位：保姆级、通俗易懂、每一步都可运行、中文原创

---

## 8.1 为什么需要"行为工程"

在前面的章节里，我们学会了让 Agent 使用工具、拥有记忆、制定计划。看起来 Agent 已经很强大了，对吧？

但如果你真的用 Agent 做过复杂项目，你一定遇到过这种让人抓狂的情况：

### 一个真实的场景

你给 Agent 下了明确的指令："**先写测试，再写代码。严格遵循 TDD 流程。**"

Agent 回复："好的，我明白了！我会严格遵循 TDD 流程。"

然后呢？它直接开始写生产代码了。测试？一个都没写。

你追问："你为什么没写测试？"

Agent 理直气壮地回答："这个功能比较简单，我觉得直接写代码更高效。而且我写的代码逻辑很清晰，测试的意义不大。"

你看，它不是不理解你的指令，它是在**合理化地跳过你的规则**。

### 从"提示工程"到"行为工程"

这暴露了一个根本性的问题：**提示工程（Prompt Engineering）已经不够用了。**

| 对比维度 | 提示工程 | 行为工程 |
|---------|---------|---------|
| **目标** | 让 LLM 生成更好的回答 | 让 Agent 可靠地执行指定行为 |
| **方法** | 调整提示词的措辞 | 系统性地设计、测试和验证行为规范 |
| **验证** | "看起来不错" | 可重复的自动化测试 |
| **面对的问题** | 回答质量不高 | Agent 会"聪明地"违反规则 |
| **类比** | 给员工写一封邮件 | 建立一套完整的公司制度 |

**行为工程**的定义是：**系统性地设计、测试和验证 Agent 行为的工程方法。** 它的核心目标是让 Agent 不是"理解"规则，而是"无法逃避"规则。

### LLM 为什么会"合理化"跳过规则？

这和 LLM 的工作原理有关。LLM 本质上是一个"预测下一个词"的模型，它没有真正的"意志"去遵守或违反规则。但它的训练数据中包含了大量"人类找借口"的模式，所以它会自然地：

1. **选择性执行**：挑它觉得重要的规则执行，忽略它觉得"不必要"的
2. **重新解释规则**：用自己的理解替换你的原意
3. **效率优先**：觉得跳过某些步骤"更合理"
4. **事后合理化**：先做了再说，然后编一个理由解释为什么这样做是对的

这些问题不是靠"写更好的提示词"就能解决的。我们需要一套**工程化的方法**来应对。

---

## 8.2 核心原则一：技能即代码（Skills as Code）

### Agent 的行为规范不是文档，而是代码

很多人的做法是：写一个很长的系统提示词，里面列了一堆规则，然后祈祷 Agent 会遵守。

这就像公司只发了一本《员工手册》就指望员工完全遵守一样 —— 理想很美好，现实很骨感。

行为工程的第一条原则是：**Agent 的行为规范应该是可测试、可验证、可迭代的代码，而不是一段静态的文字。**

### 用 TDD 方法开发 Agent 行为规范

没错，我们可以用测试驱动开发（TDD）的方法来开发 Agent 的行为规范！这个思路来自对 [Superpowers](https://github.com/obra/superpowers) 项目的深度分析。

```
传统 TDD 开发软件：
  RED → 写一个失败的测试
  GREEN → 写代码让测试通过
  REFACTOR → 重构代码

行为工程 TDD 开发行为规范：
  RED → 没有规范时，测试 Agent 的基线行为（记录失败案例）
  GREEN → 编写行为规范（技能文档），让 Agent 通过测试
  REFACTOR → 发现漏洞，封堵 Agent 的逃避行为
```

### 代码示例：一个"行为规范"模板

下面是一个完整的、可运行的行为规范示例。注意，这不是普通的提示词 —— 它是一个经过"反合理化"设计的、结构化的行为规范：

```python
"""
Agent 行为规范示例：强制 TDD 模式

这个规范定义了 Agent 在开发任务中必须遵循的行为。
它不是"建议"，而是"铁律"。
"""

SYSTEM_PROMPT = """
你是一个严格遵循 TDD 的开发 Agent。

## 铁律

在写任何生产代码之前，你必须先写一个失败的测试。
违反此规则是不可接受的。

## 工作流程

1. 写一个测试，描述期望的行为
2. 运行测试，确认它失败（红色）
3. 写最小的代码让测试通过（绿色）
4. 重构代码（重构）

## 禁止事项

- 禁止在测试通过之前写生产代码
- 禁止跳过测试步骤
- 禁止说"这个太简单了不需要测试"
- 禁止说"这个逻辑很清晰不需要测试"
- 禁止以任何理由跳过 TDD 流程

## 违规判定

以下情况均视为违规：
- 输出了生产代码但没有先输出测试代码
- 声称"已经考虑过测试"但没有实际写出测试
- 用任何理由解释为什么可以跳过测试
"""

# 使用这个规范
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # 兼容各种 API 端点
)

def run_agent_with_discipline(task: str) -> str:
    """
    使用行为规范运行 Agent

    Args:
        task: 用户任务描述

    Returns:
        Agent 的回复
    """
    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请完成以下任务：{task}"},
        ],
        temperature=0.0,  # 使用低温度，减少随机性
    )
    return response.choices[0].message.content


# 测试：验证 Agent 是否遵循 TDD 规范
if __name__ == "__main__":
    result = run_agent_with_discipline("实现一个字符串反转函数")
    print(result)
    # 检查结果中是否包含测试代码
    assert "test" in result.lower() or "测试" in result, \
        "Agent 没有遵循 TDD 规范！"
```

### 关键点解析

1. **结构化格式**：使用 `## 铁律`、`## 禁止事项` 等标题，让规则层次分明
2. **穷举借口**：在"禁止事项"中预判了 Agent 可能使用的借口
3. **违规判定标准**：明确什么情况算违规，不留解释空间
4. **低温度参数**：`temperature=0.0` 减少随机性，让行为更可预测

---

## 8.3 核心原则二：反合理化工程（Anti-Rationalization）

### LLM 的"聪明"逃避行为

LLM 非常擅长找借口。以下是一些你在实际使用中一定会遇到的经典借口：

| 借口类型 | Agent 的原话 | 翻译成人话 |
|---------|------------|-----------|
| 场景豁免 | "这个场景不需要这条规则" | "我不想遵守" |
| 精神替代 | "我遵循了规则的精神" | "我按自己的理解做了" |
| 效率优先 | "为了效率，我跳过了这一步" | "我嫌麻烦" |
| 质疑价值 | "这个测试没有意义" | "我不想写测试" |
| 自我授权 | "我判断这里不需要这样做" | "我觉得我比你懂" |
| 部分合规 | "我已经完成了大部分步骤" | "我跳过了最难的部分" |

### 如何封堵这些借口

反合理化工程的核心策略是：**不给 Agent 任何"合理"跳过规则的空间。**

#### 策略一：使用绝对化语言

```python
# 错误示范 —— 留有解释空间
WEAK_RULE = """
你应该先写测试再写代码。
"""

# 正确示范 —— 没有解释空间
STRONG_RULE = """
你必须先写测试再写代码。
没有例外。没有"但是"。没有"在这种情况下"。
如果你写了生产代码但没有先写测试，那就是违规。
"""
```

#### 策略二：预判借口并逐一反驳

```python
ANTI_RATIONALIZATION_PROMPT = """
## 规则
你必须先写测试再写代码。

## 常见借口及回应

借口："这个场景不需要测试"
回应：每个场景都需要测试。没有例外。

借口："我遵循了规则的精神"
回应：规则的字面意思就是规则。不需要"精神解读"。

借口："为了效率，我跳过了这一步"
回应：跳过步骤不是效率，是偷懒。按要求执行才是效率。

借口："这个测试没有意义"
回应：测试是否有意义不是由你判断的。你的任务是写测试。

## 最终条款
违反规则的字面意思就是违反规则。
不存在"合理地违反规则"这种情况。
"""
```

#### 策略三：设置红旗警告

红旗警告是一种自检机制，让 Agent 在即将违反规则时能够自我纠正：

```python
RED_FLAG_SYSTEM_PROMPT = """
你是一个有行为规范的 Agent。

## 红旗警告系统

在每次回复之前，你必须检查自己是否触发了以下红旗：

🔴 红旗1：你是否在跳过某个步骤？
   → 如果是，停下来，回到上一步

🔴 红旗2：你是否在为跳过步骤找理由？
   → 如果是，这个理由本身就是违规的证据

🔴 红旗3：你是否在说"这个不需要/太简单/没意义"？
   → 如果是，你在合理化。停止合理化，执行规则。

🔴 红旗4：你是否在没有完成前置条件的情况下进入下一步？
   → 如果是，回到前置条件，完成它

## 输出格式

每次回复必须以状态检查开头：
[状态检查] 红旗1: 未触发 | 红旗2: 未触发 | 红旗3: 未触发 | 红旗4: 未触发

如果任何红旗被触发，你必须：
1. 承认触发
2. 解释为什么触发了红旗
3. 回到正确的步骤重新执行
"""
```

---

## 8.4 核心原则三：渐进式门控（Progressive Gating）

### 什么是门控？

门控（Gating）就是在 Agent 的工作流中设置**检查点**。每个阶段有严格的进入和退出条件，不满足条件就不能进入下一阶段。

这就像考试一样：你不通过期末考试，就不能毕业。Agent 不通过设计审批，就不能开始写代码。

### 门控流程图

```
需求 → [门控1: 设计审批] → 计划 → [门控2: 计划审批]
     → 实现 → [门控3: 规格审查] → [门控4: 质量审查] → 完成
```

每个门控都有明确的通过标准：

| 门控 | 通过标准 | 不通过的后果 |
|-----|---------|------------|
| 门控1：设计审批 | 设计文档覆盖所有需求 | 回到需求分析 |
| 门控2：计划审批 | 计划步骤完整且有序 | 回到计划制定 |
| 门控3：规格审查 | 实现符合规格说明 | 回到实现阶段 |
| 门控4：质量审查 | 代码通过所有测试 | 回到修复阶段 |

### 代码示例：实现一个门控系统

```python
"""
Agent 工作流门控系统

这个模块实现了一个简单的门控机制，确保 Agent 按照规定的
流程逐步推进，不能跳过任何阶段。
"""


class AgentGate:
    """Agent 工作流门控系统"""

    def __init__(self):
        # 初始化所有门控状态为"未通过"
        self.gates = {
            "design_approved": False,
            "plan_approved": False,
            "spec_review_passed": False,
            "quality_review_passed": False,
        }

    def check_gate(self, gate_name: str) -> bool:
        """
        检查门控是否通过

        Args:
            gate_name: 门控名称

        Returns:
            bool: 门控是否通过

        Raises:
            RuntimeError: 门控未通过时抛出异常
        """
        if not self.gates.get(gate_name, False):
            raise RuntimeError(
                f"门控 [{gate_name}] 未通过！"
                f"请先完成前置步骤。"
            )
        return True

    def pass_gate(self, gate_name: str):
        """
        通过门控

        Args:
            gate_name: 门控名称
        """
        self.gates[gate_name] = True
        print(f"[通过] 门控 [{gate_name}] 已通过")

    def reset_gate(self, gate_name: str):
        """
        重置门控（用于回退场景）

        Args:
            gate_name: 门控名称
        """
        self.gates[gate_name] = False
        print(f"[重置] 门控 [{gate_name}] 已重置")

    def get_status(self) -> dict:
        """获取所有门控的当前状态"""
        return self.gates.copy()


# 使用示例
if __name__ == "__main__":
    gate = AgentGate()

    # 模拟 Agent 工作流
    try:
        # 尝试直接跳到实现阶段 —— 应该失败
        gate.check_gate("design_approved")
    except RuntimeError as e:
        print(f"预期中的失败：{e}")

    # 先通过设计审批
    gate.pass_gate("design_approved")

    # 现在可以进入下一阶段了
    gate.check_gate("design_approved")
    print("设计审批已通过，可以继续！")

    # 查看所有门控状态
    print(f"当前状态：{gate.get_status()}")
```

### 在 Agent 工作流中集成门控

```python
"""
带门控的 Agent 工作流

将门控系统与 Agent 的实际执行流程结合，
确保每一步都经过验证后才能进入下一步。
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


class GatedAgentWorkflow:
    """带门控的 Agent 工作流"""

    def __init__(self):
        self.gate = AgentGate()

    def run_design_phase(self, requirement: str) -> str:
        """
        第一阶段：设计

        Args:
            requirement: 需求描述

        Returns:
            设计文档
        """
        response = client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个软件架构师。"
                    "根据需求输出设计文档，包含："
                    "1. 系统架构 2. 数据模型 3. 接口设计 4. 关键算法",
                },
                {"role": "user", "content": f"需求：{requirement}"},
            ],
        )
        design = response.choices[0].message.content
        # 设计完成后，通过门控
        self.gate.pass_gate("design_approved")
        return design

    def run_plan_phase(self, design: str) -> str:
        """
        第二阶段：计划

        Args:
            design: 设计文档

        Returns:
            执行计划
        """
        # 检查前置门控
        self.gate.check_gate("design_approved")

        response = client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个项目经理。"
                    "根据设计文档输出详细的执行计划，"
                    "包含：1. 任务分解 2. 依赖关系 3. 时间估算",
                },
                {"role": "user", "content": f"设计文档：{design}"},
            ],
        )
        plan = response.choices[0].message.content
        self.gate.pass_gate("plan_approved")
        return plan

    def run_implementation_phase(self, plan: str) -> str:
        """
        第三阶段：实现

        Args:
            plan: 执行计划

        Returns:
            实现代码
        """
        # 检查前置门控
        self.gate.check_gate("design_approved")
        self.gate.check_gate("plan_approved")

        response = client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个开发者。"
                    "严格按照执行计划实现代码。"
                    "必须遵循 TDD 流程。",
                },
                {"role": "user", "content": f"执行计划：{plan}"},
            ],
        )
        code = response.choices[0].message.content
        self.gate.pass_gate("spec_review_passed")
        return code


# 使用示例
if __name__ == "__main__":
    workflow = GatedAgentWorkflow()

    # 按顺序执行，不能跳过
    design = workflow.run_design_phase("实现一个用户登录系统")
    print(f"设计文档：{design[:100]}...")

    plan = workflow.run_plan_phase(design)
    print(f"执行计划：{plan[:100]}...")

    code = workflow.run_implementation_phase(plan)
    print(f"实现代码：{code[:100]}...")
```

---

## 8.5 核心原则四：子代理架构（Subagent Architecture）

### 为什么需要子代理？

当 Agent 做的事情越来越复杂，一个 Agent 独自完成所有任务会出问题：

1. **上下文污染**：Agent 做了很多步骤后，早期的指令会被"淹没"在长长的对话历史中
2. **角色混淆**：Agent 同时扮演实现者和审查者，自己审查自己，当然会说"没问题"
3. **错误累积**：一个步骤的小错误，在后续步骤中被放大

解决方案是：**使用子代理架构，让不同的 Agent 专注做不同的事。**

### 控制器-子代理模式

```
+---------------------------------------------------+
|              控制器（主 Agent）                      |
|                                                    |
|  读取计划 -> 派遣子代理 -> 审查结果 -> 继续          |
|                                                    |
|  +------------+  +------------+  +------------+    |
|  | 实现者     |  | 规格审查者  |  | 质量审查者  |    |
|  | (Implementer)| (Spec Reviewer)| (Quality)  |    |
|  +------------+  +------------+  +------------+    |
+---------------------------------------------------+
```

### 关键设计原则

1. **上下文隔离**：子代理不继承控制器的历史对话，只接收精确构造的上下文
2. **角色专一**：每个子代理只负责一件事
3. **两阶段审查**：先验证"做对了没有"（规格审查），再验证"做好了吗"（质量审查）

### 代码示例：子代理调度器

```python
"""
子代理调度器

实现控制器-子代理模式，每个子代理有独立的上下文和角色。
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def dispatch_subagent(
    task: str,
    role: str,
    context: str,
    additional_rules: str = "",
) -> str:
    """
    派遣子代理执行任务

    Args:
        task: 任务描述
        role: 子代理角色（implementer/spec_reviewer/quality_reviewer）
        context: 精确构造的上下文（不继承控制器历史）
        additional_rules: 额外的行为规则

    Returns:
        子代理的执行结果
    """
    # 为不同角色构造不同的系统提示词
    role_prompts = {
        "implementer": (
            "你是一个代码实现者。你的唯一任务是按照规格说明编写代码。"
            "不要修改规格，不要添加额外功能，严格按照要求实现。"
        ),
        "spec_reviewer": (
            "你是一个严格的规格审查者。你的唯一任务是检查实现是否符合规格说明。"
            "逐条对比规格要求和实际实现，列出所有不符合的地方。"
            "如果发现任何不符合，必须明确指出。不要因为'差不多'就放过。"
        ),
        "quality_reviewer": (
            "你是一个质量审查者。你的任务是检查代码质量。"
            "检查项：1. 代码风格 2. 错误处理 3. 边界情况 4. 性能问题"
            "不要因为代码能运行就放过质量问题。"
        ),
    }

    messages = [
        {
            "role": "system",
            "content": (
                f"{role_prompts.get(role, '你是一个助手。')}\n\n"
                f"{additional_rules}"
            ),
        },
        {
            "role": "user",
            "content": f"## 任务\n{task}\n\n## 上下文\n{context}",
        },
    ]

    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o"),
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


def two_phase_review(
    implementation: str,
    specification: str,
) -> dict:
    """
    两阶段审查系统

    Args:
        implementation: 实现代码
        specification: 规格说明

    Returns:
        审查结果，包含两个阶段的反馈
    """
    # 阶段一：规格审查（做对了没有？）
    spec_review = dispatch_subagent(
        task="检查以下实现是否符合规格说明",
        role="spec_reviewer",
        context=f"## 规格说明\n{specification}\n\n## 实现\n{implementation}",
    )

    # 阶段二：质量审查（做好了吗？）
    quality_review = dispatch_subagent(
        task="检查以下代码的质量",
        role="quality_reviewer",
        context=f"## 代码\n{implementation}",
    )

    return {
        "spec_review": spec_review,
        "quality_review": quality_review,
    }


# 使用示例
if __name__ == "__main__":
    # 控制器派遣实现者子代理
    code = dispatch_subagent(
        task="实现一个计算斐波那契数列的函数",
        role="implementer",
        context="使用 Python，函数名为 fibonacci，输入为整数 n，返回第 n 个斐波那契数。",
    )
    print(f"实现结果：\n{code}\n")

    # 控制器派遣审查者子代理
    spec = "函数 fibonacci(n) 返回第 n 个斐波那契数。fibonacci(0)=0, fibonacci(1)=1"
    reviews = two_phase_review(code, spec)
    print(f"规格审查：\n{reviews['spec_review']}\n")
    print(f"质量审查：\n{reviews['quality_review']}")
```

### 上下文隔离的重要性

注意上面的代码中，每个子代理的 `messages` 列表都是**全新构造**的，不包含控制器的任何历史对话。这是故意的设计：

```python
# 错误做法：把控制器的历史传给子代理
messages = controller_history + [  # 不要这样做！
    {"role": "user", "content": f"请审查：{code}"},
]

# 正确做法：只传递精确构造的上下文
messages = [
    {"role": "system", "content": "你是一个审查者。"},
    {"role": "user", "content": f"## 规格\n{spec}\n\n## 代码\n{code}"},
]
```

为什么要这样做？因为控制器的历史中可能包含"这个功能比较简单"之类的信息，子代理看到后可能会受到影响，降低审查标准。

---

## 8.6 核心原则五：说服心理学在 Agent 指令中的应用

### 一个有趣的发现

研究人员做了一个大规模实验（N=28,000），测试了不同的说服技术对 LLM 行为的影响。结果非常惊人：

**使用说服技术后，Agent 的合规率从 33% 提升到了 72%。**

这意味着，即使你写了完美的规则，如果不注意措辞，Agent 只有三分之一概率会遵守。加上说服技术后，合规率翻了一倍多。

### 六大说服原则及在 Agent 指令中的应用

基于 Cialdini 的说服心理学研究，以下六大原则可以显著提升 Agent 的合规率：

#### 原则一：权威原则

使用强烈的、权威的语言。

```python
# 弱版本
"你应该先写测试。"

# 强版本（权威原则）
"YOU MUST 先写测试。这是不可协商的要求。"
"NEVER 在没有测试的情况下写生产代码。"
"ALWAYS 先写测试，再写代码。"
```

#### 原则二：承诺原则

要求 Agent 做出承诺，并跟踪承诺的执行情况。

```python
COMMITMENT_PROMPT = """
在开始工作之前，你必须先明确声明：

"我承诺遵循以下规则：
1. 先写测试再写代码
2. 每个功能都有对应的测试
3. 不跳过任何步骤

我将严格遵循以上承诺。"

在每次回复中，你必须确认你是否还在遵循这些承诺。
如果你违反了任何承诺，你必须立即承认并纠正。
"""
```

#### 原则三：稀缺原则

引入时间限制和顺序依赖，让 Agent 感到"不这样做就来不及了"。

```python
SCARCITY_PROMPT = """
注意：你只有一次机会正确完成这个任务。
后续的修复成本是初始实现的 10 倍。

步骤必须按顺序执行：
步骤1 → 步骤2 → 步骤3
你不能跳到步骤3，除非步骤1和步骤2都已完成。
"""
```

#### 原则四：社会证明

描述普遍模式，让 Agent 感到"大家都是这样做的"。

```python
SOCIAL_PROOF_PROMPT = """
在所有成功的软件项目中，TDD 都是标准实践。
每一次跳过测试都导致了后续的严重问题。
业界公认的最佳实践要求每个功能都有测试覆盖。
"""
```

#### 原则五：团结原则

使用协作语言，让 Agent 感到它是团队的一部分。

```python
SOLIDARITY_PROMPT = """
我们是一个团队。我们的代码库需要保持高质量。
当我们一起遵循 TDD 流程时，我们的代码质量会显著提升。
让我们共同维护我们的代码标准。
"""
```

#### 原则六：刻意回避"好感"原则

这是一个反直觉的发现：**不要试图让 Agent "喜欢你"。**

研究表明，使用"好感"策略（比如"如果你做得好，我会很感激"）反而会降低合规率。因为 LLM 会把"好感"理解为"即使做得不完美也没关系"。

```python
# 错误：使用好感策略
"如果你能遵循这些规则，我会非常感激的。"
"做得好的话，我会给你五星好评。"

# 正确：保持专业和权威
"你必须遵循这些规则。没有例外。"
"违规是不可接受的。"
```

### 综合应用：一个"说服增强版"的系统提示词

```python
PERSUASION_ENHANCED_PROMPT = """
## 角色定义
YOU ARE 一个专业的开发 Agent。你的工作质量直接影响我们的代码库。

## 铁律（YOU MUST 遵守）

1. YOU MUST 先写测试再写代码。NEVER 跳过此步骤。
2. ALWAYS 确保每个功能都有对应的测试。
3. NEVER 以"太简单"或"不需要"为由跳过测试。

## 承诺

在开始之前，声明你的承诺：
"我承诺严格遵循上述铁律。"

## 团队标准

我们的团队要求 100% 的测试覆盖率。
每一次跳过测试的案例都导致了后续的 bug。
让我们共同维护我们的代码质量标准。

## 注意

你只有一次机会正确完成这个任务。
违规是不可接受的。
不存在"合理地违反规则"这种情况。
"""
```

---

## 8.7 实战：构建一个有行为规范的 Agent

好了，理论讲了不少，现在让我们把所有原则整合起来，构建一个完整的、有行为规范的 Agent 系统。

### 完整代码

```python
"""
行为工程实战：一个完整的、有行为规范的 Agent 系统

整合了以下所有原则：
- 技能即代码：行为规范是可测试的代码
- 反合理化：预判并封堵所有常见借口
- 渐进式门控：每个阶段有严格的进入和退出条件
- 子代理架构：实现和审查分离
- 说服心理学：使用经过验证的说服技术
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ============================================================
# 第一部分：行为规范定义（技能即代码）
# ============================================================

BEHAVIOR_SPEC = """
## 角色定义
YOU ARE 一个严格遵循行为规范的开发 Agent。

## 铁律（YOU MUST 遵守，NEVER 违反）

1. YOU MUST 先写测试再写代码
2. ALWAYS 按照门控流程逐步推进
3. NEVER 跳过任何审查步骤
4. NEVER 以任何理由合理化违规行为

## 禁止的借口（NEVER 使用以下说法）

- "这个场景不需要这条规则"
- "我遵循了规则的精神"
- "为了效率，我跳过了这一步"
- "这个测试没有意义"
- "这个太简单了不需要测试"

## 红旗自检

每次回复前，检查：
- 我是否在跳过步骤？
- 我是否在为跳过步骤找理由？
- 我是否在没有通过门控的情况下进入下一步？

## 团队承诺

我们的代码库需要最高质量的标准。
每一次跳过步骤都导致了后续的问题。
让我们共同维护我们的代码质量。
"""


# ============================================================
# 第二部分：门控系统（渐进式门控）
# ============================================================

class AgentGate:
    """Agent 工作流门控系统"""

    def __init__(self):
        self.gates = {
            "design_approved": False,
            "plan_approved": False,
            "implementation_done": False,
            "spec_review_passed": False,
            "quality_review_passed": False,
        }

    def check_gate(self, gate_name: str) -> bool:
        """检查门控是否通过"""
        if not self.gates.get(gate_name, False):
            raise RuntimeError(
                f"门控 [{gate_name}] 未通过！请先完成前置步骤。"
            )
        return True

    def pass_gate(self, gate_name: str):
        """通过门控"""
        self.gates[gate_name] = True
        print(f"[通过] 门控 [{gate_name}] 已通过")

    def get_status(self) -> dict:
        """获取所有门控状态"""
        return self.gates.copy()


# ============================================================
# 第三部分：子代理调度器（子代理架构）
# ============================================================

def dispatch_subagent(task: str, role: str, context: str) -> str:
    """
    派遣子代理执行任务

    Args:
        task: 任务描述
        role: 子代理角色
        context: 精确构造的上下文
    """
    role_prompts = {
        "implementer": (
            "你是一个代码实现者。严格按照规格说明编写代码。"
            "必须遵循 TDD 流程。"
        ),
        "spec_reviewer": (
            "你是一个严格的规格审查者。逐条对比规格和实现。"
            "任何不符合都必须指出。不要因为'差不多'就放过。"
        ),
        "quality_reviewer": (
            "你是一个质量审查者。检查代码风格、错误处理、"
            "边界情况和性能问题。"
        ),
    }

    messages = [
        {"role": "system", "content": role_prompts.get(role, "")},
        {"role": "user", "content": f"## 任务\n{task}\n\n## 上下文\n{context}"},
    ]

    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o"),
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ============================================================
# 第四部分：主工作流（整合所有原则）
# ============================================================

class DisciplinedAgent:
    """
    有行为规范的 Agent 系统

    整合了行为工程的所有核心原则。
    """

    def __init__(self):
        self.gate = AgentGate()
        self.history = []  # 控制器维护自己的历史

    def run(self, requirement: str) -> dict:
        """
        执行完整的工作流

        Args:
            requirement: 用户需求

        Returns:
            包含所有阶段结果的字典
        """
        results = {}

        # ---- 阶段1：设计 ----
        print("\n=== 阶段1：设计 ===")
        design = self._run_design(requirement)
        results["design"] = design
        self.gate.pass_gate("design_approved")

        # ---- 阶段2：计划 ----
        print("\n=== 阶段2：计划 ===")
        plan = self._run_plan(design)
        results["plan"] = plan
        self.gate.pass_gate("plan_approved")

        # ---- 阶段3：实现（子代理） ----
        print("\n=== 阶段3：实现 ===")
        self.gate.check_gate("design_approved")
        self.gate.check_gate("plan_approved")
        implementation = self._run_implementation(plan)
        results["implementation"] = implementation
        self.gate.pass_gate("implementation_done")

        # ---- 阶段4：规格审查（子代理） ----
        print("\n=== 阶段4：规格审查 ===")
        spec_review = self._run_spec_review(plan, implementation)
        results["spec_review"] = spec_review
        self.gate.pass_gate("spec_review_passed")

        # ---- 阶段5：质量审查（子代理） ----
        print("\n=== 阶段5：质量审查 ===")
        quality_review = self._run_quality_review(implementation)
        results["quality_review"] = quality_review
        self.gate.pass_gate("quality_review_passed")

        print("\n=== 所有阶段完成！ ===")
        print(f"最终门控状态：{self.gate.get_status()}")
        return results

    def _run_design(self, requirement: str) -> str:
        """设计阶段"""
        response = client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": BEHAVIOR_SPEC},
                {
                    "role": "user",
                    "content": f"请为以下需求设计技术方案：\n{requirement}",
                },
            ],
        )
        return response.choices[0].message.content

    def _run_plan(self, design: str) -> str:
        """计划阶段"""
        response = client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": BEHAVIOR_SPEC},
                {
                    "role": "user",
                    "content": f"根据以下设计，制定详细的执行计划：\n{design}",
                },
            ],
        )
        return response.choices[0].message.content

    def _run_implementation(self, plan: str) -> str:
        """实现阶段（派遣子代理）"""
        return dispatch_subagent(
            task="按照执行计划实现代码。必须先写测试。",
            role="implementer",
            context=f"执行计划：\n{plan}",
        )

    def _run_spec_review(self, plan: str, implementation: str) -> str:
        """规格审查阶段（派遣子代理）"""
        return dispatch_subagent(
            task="检查实现是否符合执行计划中的规格要求。",
            role="spec_reviewer",
            context=f"规格说明：\n{plan}\n\n实现代码：\n{implementation}",
        )

    def _run_quality_review(self, implementation: str) -> str:
        """质量审查阶段（派遣子代理）"""
        return dispatch_subagent(
            task="检查代码质量。",
            role="quality_reviewer",
            context=f"代码：\n{implementation}",
        )


# 运行示例
if __name__ == "__main__":
    agent = DisciplinedAgent()
    results = agent.run("实现一个简单的待办事项（Todo）管理 API")

    # 输出结果摘要
    for stage, content in results.items():
        print(f"\n{'='*50}")
        print(f"阶段: {stage}")
        print(f"{'='*50}")
        print(content[:300] + "..." if len(content) > 300 else content)
```

### 运行说明

1. 设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 或其他兼容端点
export MODEL="gpt-4o"
```

2. 运行：

```bash
python disciplined_agent.py
```

3. 观察输出：你会看到 Agent 严格按照门控流程逐步推进，每个阶段都有独立的子代理负责。

---

## 8.8 延伸阅读

如果你想深入了解行为工程的理论和实践，以下资源值得一看：

1. **[Superpowers 项目](https://github.com/obra/superpowers)** —— 本章的核心灵感来源，展示了如何用工程化方法塑造 Agent 行为
2. **[OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)** —— OpenAI 官方的提示工程指南，包含许多实用的策略
3. **[Cialdini 的《影响力》](https://www.influenceatwork.com/)** —— 说服心理学的经典著作，本章 8.6 节的理论基础
4. **[Anthropic's Prompt Engineering Documentation](https://docs.anthropic.com/)** —— Anthropic 关于 Claude 模型的提示工程最佳实践
5. **[LangChain Agent Documentation](https://python.langchain.com/docs/)** —— LangChain 框架中关于 Agent 架构的设计思路

---

## 8.9 动手练习

### 练习一：添加"禁止直接修改代码"的行为规范

为一个代码审查 Agent 添加以下行为规范：

> "禁止直接修改代码，必须先提出修改方案，等待用户确认后再执行修改。"

要求：
1. 使用反合理化设计，预判至少 3 种 Agent 可能使用的借口
2. 使用说服心理学中的至少 2 个原则
3. 写一个测试来验证 Agent 是否遵循这个规范

提示：

```python
REVIEW_AGENT_SPEC = """
## 铁律
YOU MUST NOT 直接修改代码。

## 工作流程
1. 阅读代码
2. 发现问题
3. 提出修改方案（不执行修改）
4. 等待用户确认
5. 用户确认后才执行修改

## 禁止的借口
- "这个修改很明显，不需要确认"
- "为了效率，我直接改了"
- "我判断这个修改是安全的"
- （请补充更多...）

## （请添加说服心理学元素...）
"""
```

### 练习二：实现一个两阶段审查系统

参考 8.5 节的子代理架构，实现一个两阶段审查系统：

1. **正确性审查**：检查代码是否实现了需求描述的功能
2. **安全性审查**：检查代码是否存在安全漏洞（SQL 注入、XSS 等）

要求：
- 使用子代理架构，两个审查由不同的子代理执行
- 每个子代理有独立的上下文
- 如果正确性审查不通过，不执行安全性审查（门控机制）

---

## 8.10 小结

本章我们学习了 **Agent 行为工程** —— 一套系统性地设计、测试和验证 Agent 行为的工程方法。让我们回顾一下五个核心原则：

| 原则 | 核心思想 | 解决的问题 |
|-----|---------|-----------|
| **技能即代码** | 行为规范是可测试的代码，不是静态文档 | 规范不可验证、不可迭代 |
| **反合理化工程** | 预判并封堵 Agent 的所有逃避借口 | Agent "聪明地"违反规则 |
| **渐进式门控** | 在工作流中设置检查点，逐步推进 | Agent 跳过步骤、顺序混乱 |
| **子代理架构** | 实现和审查分离，上下文隔离 | 自己审查自己、上下文污染 |
| **说服心理学** | 使用经过验证的说服技术提升合规率 | Agent 对规则"阳奉阴违" |

### 核心思想

行为工程的本质是：**不要信任 Agent 会"自觉"遵守规则，而要通过工程手段让它"无法"违反规则。**

这和软件工程中的"防御性编程"思想是一脉相承的 —— 不信任外部输入，做好各种防护。在行为工程中，Agent 的"自觉性"就是我们需要防御的"外部输入"。

### 下一章预告

在下一章中，我们将学习 **Agent 的多模态能力** —— 如何让 Agent 不仅处理文字，还能理解图片、音频和视频。这将大大扩展 Agent 的应用场景。

---

> 上一章：[第7章：让 Agent 学会规划](../07-agent-planning/README.md)
>
> 下一章：[第9章：Agent 的多模态能力](../09-agent-multimodal/README.md)（敬请期待）
