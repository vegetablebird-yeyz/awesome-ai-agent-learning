"""第13章：带真实门控、共享状态和成本预算的离线多 Agent 团队。

“真实门控”指路由由普通 Python 条件决定，而不是让模型口头说“已通过”。
所有 Agent 都是确定性实现，便于无 Key 运行；换成 LLM 后仍应保留协调器。

运行：
    python examples/13_multi_agent.py
    python examples/13_multi_agent.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class Status(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RESEARCHED = "researched"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"
    REJECTED = "rejected"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Message:
    message_id: str
    task_id: str
    sender: str
    receiver: str
    kind: str
    payload: Mapping[str, Any]
    created_at: float


@dataclass(frozen=True)
class Usage:
    agent: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float


@dataclass
class SharedState:
    """协调器拥有的唯一事实源，Agent 只能返回补丁。"""

    task_id: str
    task: str
    status: Status = Status.CREATED
    plan: list[str] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    draft: str = ""
    review_issues: list[str] = field(default_factory=list)
    revision_count: int = 0
    step_count: int = 0
    spent: float = 0.0
    messages: list[Message] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def public_snapshot(self) -> dict[str, Any]:
        """日志和 Agent 输入只拿必要字段，不暴露内部对象引用。"""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "status": self.status.value,
            "plan": copy.deepcopy(self.plan),
            "evidence": copy.deepcopy(self.evidence),
            "draft": self.draft,
            "review_issues": list(self.review_issues),
            "revision_count": self.revision_count,
            "spent": round(self.spent, 6),
        }


@dataclass(frozen=True)
class AgentResult:
    sender: str
    receiver: str
    kind: str
    patch: Mapping[str, Any]
    text: str
    usage: Usage


class BudgetExceeded(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    pass


class CostMeter:
    """教学用成本估算器；真实项目读取供应商 usage 字段。"""

    PRICES = {
        "planner": (0.0000004, 0.0000012),
        "researcher": (0.0000005, 0.0000015),
        "writer": (0.0000005, 0.0000015),
        "reviewer": (0.0000004, 0.0000012),
    }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # 中文近似每字一个 token；英文粗略每四字符一个。
        chinese = len("".join(c for c in text if "\u4e00" <= c <= "\u9fff"))
        others = max(0, len(text) - chinese)
        return max(1, chinese + others // 4)

    def measure(self, agent: str, prompt: str, output: str) -> Usage:
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(output)
        input_price, output_price = self.PRICES[agent]
        cost = input_tokens * input_price + output_tokens * output_price
        return Usage(agent, input_tokens, output_tokens, cost)


class BaseAgent:
    name = "base"

    def __init__(self, meter: CostMeter) -> None:
        self.meter = meter

    def make_result(
        self,
        receiver: str,
        kind: str,
        patch: Mapping[str, Any],
        prompt: str,
        text: str,
    ) -> AgentResult:
        return AgentResult(
            sender=self.name,
            receiver=receiver,
            kind=kind,
            patch=patch,
            text=text,
            usage=self.meter.measure(self.name, prompt, text),
        )


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, snapshot: Mapping[str, Any]) -> AgentResult:
        task = str(snapshot["task"])
        plan = [
            "定义可靠知识助手的验收标准",
            "收集引用、拒答和人工确认的证据",
            "形成带来源草稿并独立审查",
        ]
        if "安全" in task or "可靠" in task:
            plan.insert(1, "确认工具权限和提示注入边界")
        text = "；".join(f"{index}. {step}" for index, step in enumerate(plan, 1))
        return self.make_result(
            receiver="coordinator",
            kind="plan",
            patch={"plan": plan},
            prompt=task,
            text=text,
        )


class ResearcherAgent(BaseAgent):
    name = "researcher"

    KNOWLEDGE = (
        {
            "id": "kb#citation",
            "text": "知识助手的关键结论应附可复核来源。",
            "keywords": {"知识", "可靠", "来源", "引用"},
        },
        {
            "id": "kb#refusal",
            "text": "资料不足时必须拒答，不能根据模型记忆猜测。",
            "keywords": {"知识", "可靠", "资料", "拒答"},
        },
        {
            "id": "kb#approval",
            "text": "发送、删除和付费等高风险动作必须等待人工确认。",
            "keywords": {"可靠", "安全", "工具", "人工确认"},
        },
        {
            "id": "kb#isolation",
            "text": "每个 Agent 只获得完成职责所需的最小上下文和工具。",
            "keywords": {"可靠", "安全", "权限", "agent"},
        },
    )

    def run(self, snapshot: Mapping[str, Any]) -> AgentResult:
        prompt = json.dumps(
            {"task": snapshot["task"], "plan": snapshot["plan"]},
            ensure_ascii=False,
        )
        task = str(snapshot["task"]).lower()
        evidence = []
        for item in self.KNOWLEDGE:
            if any(keyword.lower() in task for keyword in item["keywords"]):
                evidence.append({"id": item["id"], "text": item["text"]})
        # 教学任务固定保留至少三条；真实检索器应靠阈值决定。
        if len(evidence) < 3:
            evidence = [
                {"id": item["id"], "text": item["text"]}
                for item in self.KNOWLEDGE[:3]
            ]
        text = "\n".join(f"- {item['text']} [{item['id']}]" for item in evidence)
        return self.make_result(
            receiver="coordinator",
            kind="evidence",
            patch={"evidence": evidence},
            prompt=prompt,
            text=text,
        )


class WriterAgent(BaseAgent):
    name = "writer"

    def run(self, snapshot: Mapping[str, Any]) -> AgentResult:
        evidence = list(snapshot["evidence"])
        issues = list(snapshot["review_issues"])
        prompt = json.dumps(
            {"task": snapshot["task"], "evidence": evidence, "issues": issues},
            ensure_ascii=False,
        )
        lines = [f"研究结论：{snapshot['task']}"]
        for item in evidence:
            lines.append(f"- {item['text']} [{item['id']}]")
        if issues:
            lines.append("本稿已按审查意见补充明确的限制与人工确认边界。")
        lines.append("边界：证据之外的信息不作推断，高风险动作不自动执行。")
        draft = "\n".join(lines)
        return self.make_result(
            receiver="coordinator",
            kind="draft",
            patch={"draft": draft},
            prompt=prompt,
            text=draft,
        )


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def run(self, snapshot: Mapping[str, Any]) -> AgentResult:
        draft = str(snapshot["draft"])
        evidence = list(snapshot["evidence"])
        prompt = json.dumps(
            {"draft": draft, "allowed_sources": [item["id"] for item in evidence]},
            ensure_ascii=False,
        )
        issues: list[str] = []
        allowed_sources = {item["id"] for item in evidence}
        cited_sources = set(re.findall(r"\[([^\[\]]+)\]", draft))
        if not draft.strip():
            issues.append("草稿为空")
        if not cited_sources:
            issues.append("缺少引用")
        if cited_sources - allowed_sources:
            issues.append("存在未召回引用")
        if "不作推断" not in draft:
            issues.append("缺少资料边界")
        if "人工确认" not in draft and "不自动执行" not in draft:
            issues.append("缺少高风险动作边界")

        approved = not issues
        text = "审查通过" if approved else "审查退回：" + "；".join(issues)
        return self.make_result(
            receiver="coordinator",
            kind="review",
            patch={"approved": approved, "review_issues": issues},
            prompt=prompt,
            text=text,
        )


class Coordinator:
    """唯一允许改变共享状态和调用 Agent 的组件。"""

    ALLOWED_TRANSITIONS = {
        Status.CREATED: {Status.PLANNED, Status.BUDGET_EXCEEDED, Status.FAILED},
        Status.PLANNED: {Status.RESEARCHED, Status.BUDGET_EXCEEDED, Status.FAILED},
        Status.RESEARCHED: {
            Status.NEEDS_REVISION,
            Status.APPROVED,
            Status.BUDGET_EXCEEDED,
            Status.FAILED,
        },
        Status.NEEDS_REVISION: {
            Status.RESEARCHED,
            Status.REJECTED,
            Status.BUDGET_EXCEEDED,
            Status.FAILED,
        },
    }

    PATCH_FIELDS = {
        "planner": {"plan"},
        "researcher": {"evidence"},
        "writer": {"draft"},
        "reviewer": {"approved", "review_issues"},
    }

    def __init__(
        self,
        budget: float = 0.01,
        max_steps: int = 10,
        max_revisions: int = 1,
    ) -> None:
        self.budget = budget
        self.max_steps = max_steps
        self.max_revisions = max_revisions
        meter = CostMeter()
        self.agents = {
            "planner": PlannerAgent(meter),
            "researcher": ResearcherAgent(meter),
            "writer": WriterAgent(meter),
            "reviewer": ReviewerAgent(meter),
        }

    @staticmethod
    def task_id(task: str) -> str:
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:10]
        return f"task-{digest}"

    def transition(self, state: SharedState, target: Status) -> None:
        allowed = self.ALLOWED_TRANSITIONS.get(state.status, set())
        if target not in allowed:
            raise ProtocolError(f"非法状态迁移：{state.status.value} -> {target.value}")
        state.status = target

    def check_gate(self, state: SharedState, next_agent: str) -> None:
        """调用前的硬门控；不满足就根本不会调用下一个 Agent。"""
        if state.step_count >= self.max_steps:
            raise ProtocolError("达到最大步骤数")
        if state.spent >= self.budget:
            raise BudgetExceeded("任务预算已耗尽")
        requirements: dict[str, Callable[[SharedState], bool]] = {
            "planner": lambda s: s.status == Status.CREATED,
            "researcher": lambda s: s.status == Status.PLANNED and bool(s.plan),
            "writer": lambda s: s.status in {
                Status.RESEARCHED,
                Status.NEEDS_REVISION,
            } and bool(s.evidence),
            "reviewer": lambda s: s.status == Status.RESEARCHED and bool(s.draft),
        }
        if next_agent not in requirements or not requirements[next_agent](state):
            raise ProtocolError(
                f"门控拒绝：status={state.status.value}, next={next_agent}"
            )

    def validate_result(self, result: AgentResult) -> None:
        if result.sender not in self.PATCH_FIELDS:
            raise ProtocolError(f"未知发送者：{result.sender}")
        unexpected = set(result.patch) - self.PATCH_FIELDS[result.sender]
        if unexpected:
            raise ProtocolError(f"{result.sender} 越权修改字段：{sorted(unexpected)}")
        if result.receiver != "coordinator":
            raise ProtocolError("Agent 结果必须返回协调器")

    def charge(self, state: SharedState, usage: Usage) -> None:
        projected = state.spent + usage.estimated_cost
        if projected > self.budget:
            raise BudgetExceeded(
                f"预计费用 {projected:.6f} 超过预算 {self.budget:.6f}"
            )
        state.spent = projected
        state.usage.append(usage)

    def apply(self, state: SharedState, result: AgentResult) -> None:
        self.validate_result(result)
        self.charge(state, result.usage)
        state.step_count += 1
        message = Message(
            message_id=f"msg-{state.step_count:03d}",
            task_id=state.task_id,
            sender=result.sender,
            receiver=result.receiver,
            kind=result.kind,
            payload=dict(result.patch),
            created_at=time.time(),
        )
        state.messages.append(message)

        if result.sender == "planner":
            state.plan = list(result.patch["plan"])
            self.transition(state, Status.PLANNED)
        elif result.sender == "researcher":
            state.evidence = copy.deepcopy(list(result.patch["evidence"]))
            self.transition(state, Status.RESEARCHED)
        elif result.sender == "writer":
            state.draft = str(result.patch["draft"])
            # 返工写完后重新进入可审查状态。
            if state.status == Status.NEEDS_REVISION:
                self.transition(state, Status.RESEARCHED)
        elif result.sender == "reviewer":
            state.review_issues = list(result.patch["review_issues"])
            if bool(result.patch["approved"]):
                self.transition(state, Status.APPROVED)
            elif state.revision_count < self.max_revisions:
                state.revision_count += 1
                self.transition(state, Status.NEEDS_REVISION)
            else:
                self.transition(state, Status.REJECTED)

    def invoke(self, state: SharedState, agent_name: str) -> None:
        self.check_gate(state, agent_name)
        result = self.agents[agent_name].run(state.public_snapshot())
        self.apply(state, result)

    def run(self, task: str) -> SharedState:
        state = SharedState(task_id=self.task_id(task), task=task)
        try:
            self.invoke(state, "planner")
            self.invoke(state, "researcher")
            self.invoke(state, "writer")
            self.invoke(state, "reviewer")
            while state.status == Status.NEEDS_REVISION:
                self.invoke(state, "writer")
                self.invoke(state, "reviewer")
        except BudgetExceeded as exc:
            state.errors.append(str(exc))
            if state.status not in {Status.APPROVED, Status.REJECTED}:
                self.transition(state, Status.BUDGET_EXCEEDED)
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            state.errors.append(str(exc))
            if state.status not in {Status.APPROVED, Status.REJECTED}:
                self.transition(state, Status.FAILED)
        return state


def print_report(state: SharedState) -> None:
    print(f"任务：{state.task}")
    print(f"task_id={state.task_id} status={state.status.value}")
    print(f"steps={state.step_count} revisions={state.revision_count}")
    print(f"estimated_cost={state.spent:.6f} 元（教学估算）")
    print("\n调用链：")
    for message in state.messages:
        print(
            f"- {message.message_id} {message.sender} -> {message.receiver} "
            f"kind={message.kind}"
        )
    print("\n最终草稿：")
    print(state.draft or "(无)")
    print("\n审查问题：", state.review_issues or "无")
    if state.errors:
        print("错误：", state.errors)


def self_test() -> None:
    normal = Coordinator(budget=0.01).run("总结可靠知识助手的安全设计原则")
    assert normal.status == Status.APPROVED
    assert normal.draft
    assert normal.spent > 0
    assert len(normal.messages) >= 4

    tiny_budget = Coordinator(budget=0.000001).run("总结可靠知识助手")
    assert tiny_budget.status == Status.BUDGET_EXCEEDED
    assert tiny_budget.errors

    coordinator = Coordinator()
    state = SharedState(task_id="test", task="测试非法路由")
    try:
        coordinator.invoke(state, "reviewer")
    except ProtocolError:
        pass
    else:
        raise AssertionError("reviewer 不应绕过规划和研究门控")

    malicious = AgentResult(
        sender="researcher",
        receiver="coordinator",
        kind="evidence",
        patch={"draft": "越权篡改"},
        text="bad",
        usage=Usage("researcher", 1, 1, 0.0),
    )
    try:
        coordinator.apply(state, malicious)
    except ProtocolError:
        pass
    else:
        raise AssertionError("共享状态字段越权应被拒绝")
    print("SELF-TEST: 通过（门控、状态权限、预算、终止）")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第13章多 Agent 协调器")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--budget", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
        return
    state = Coordinator(budget=args.budget).run("总结可靠知识助手的安全设计原则")
    print_report(state)


if __name__ == "__main__":
    main()
