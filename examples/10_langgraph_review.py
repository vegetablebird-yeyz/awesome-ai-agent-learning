"""第10章完整示例：状态、条件分支、循环与内存检查点。

该程序完全离线，不调用 LLM。规则节点让每次运行结果可重复，适合观察
LangGraph 的控制流，而不是把注意力放到模型输出差异上。
"""

from __future__ import annotations

from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    """所有节点共享的可序列化状态。"""

    topic: str
    audience: str
    draft: str
    feedback: str
    revision_count: int
    max_revisions: int
    approved: bool
    needs_human: bool
    status: str


def write_draft(state: ReviewState) -> dict:
    """生成第一版草稿，只返回发生变化的字段。"""
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


def review_draft(state: ReviewState) -> dict:
    """用确定规则审查草稿，真实项目可替换为模型或人工审查。"""
    missing: list[str] = []
    if "例如" not in state["draft"]:
        missing.append("具体例子")
    if "安全" not in state["draft"]:
        missing.append("安全边界")

    if not missing:
        return {
            "approved": True,
            "feedback": "包含例子和安全边界，可以发布。",
            "status": "approved",
            "needs_human": False,
        }

    return {
        "approved": False,
        "feedback": "请补充：" + "、".join(missing),
        "status": "review_failed",
    }


def revise_draft(state: ReviewState) -> dict:
    """根据反馈逐轮补充内容，展示状态驱动的循环。"""
    revised = state["draft"]
    if "具体例子" in state["feedback"] and "例如" not in revised:
        revised += "例如：输入问题，检索资料，再附来源回答。"
    elif "安全边界" in state["feedback"] and "安全" not in revised:
        revised += "安全要求：证据不足时拒答，并对日志脱敏。"
    else:
        revised += "请人工检查未识别的审查意见。"

    return {
        "draft": revised,
        "revision_count": state["revision_count"] + 1,
        "status": "revised",
    }


def mark_for_human(state: ReviewState) -> dict:
    """达到上限后明确标记人工处理，不能伪装成成功。"""
    return {
        "needs_human": True,
        "approved": False,
        "status": "needs_human",
        "feedback": (
            f"已修订 {state['revision_count']} 次仍未通过，"
            "请人工审查后决定是否发布。"
        ),
    }


def route_after_review(
    state: ReviewState,
) -> Literal["revise", "human", "finish"]:
    """条件分支只能返回固定标签。"""
    if state["approved"]:
        return "finish"
    if state["revision_count"] >= state["max_revisions"]:
        return "human"
    return "revise"


def build_graph(checkpointer: MemorySaver | None = None):
    """注册节点和边，并选择是否开启检查点。"""
    graph = StateGraph(ReviewState)
    graph.add_node("write", write_draft)
    graph.add_node("review", review_draft)
    graph.add_node("revise", revise_draft)
    graph.add_node("human", mark_for_human)

    graph.add_edge(START, "write")
    graph.add_edge("write", "review")
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
    graph.add_edge("human", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(
    topic: str,
    audience: str = "Python 初学者",
    max_revisions: int = 3,
) -> ReviewState:
    """集中构造完整初始状态，避免调用方漏字段。"""
    if not topic.strip():
        raise ValueError("topic 不能为空")
    if max_revisions < 0 or max_revisions > 10:
        raise ValueError("max_revisions 必须在 0 到 10 之间")
    return {
        "topic": topic,
        "audience": audience,
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "approved": False,
        "needs_human": False,
        "status": "new",
    }


def print_result(result: ReviewState) -> None:
    """以稳定格式输出业务验收信息。"""
    print(f"状态：{result['status']}")
    print(f"草稿：{result['draft']}")
    print(f"审查：{result['feedback']}")
    print(f"修订次数：{result['revision_count']}")
    print(f"通过：{result['approved']}")
    print(f"需要人工：{result['needs_human']}")


def run_checkpoint_demo() -> None:
    """演示 thread_id 隔离和检查点历史读取。"""
    memory = MemorySaver()
    app = build_graph(checkpointer=memory)

    thread_a = f"tutorial-{uuid4().hex}"
    config_a = {
        "configurable": {"thread_id": thread_a},
        "recursion_limit": 20,
    }
    result_a = app.invoke(
        initial_state("如何构建可靠的 RAG"),
        config=config_a,
    )

    print("=== 会话 A ===")
    print_result(result_a)

    snapshot = app.get_state(config_a)
    print(f"检查点 next：{snapshot.next}")
    print(f"检查点状态：{snapshot.values['status']}")

    history = list(app.get_state_history(config_a))
    print(f"检查点数量：{len(history)}")

    # 第二个 thread_id 从独立初始状态运行，不能读到会话 A。
    thread_b = f"tutorial-{uuid4().hex}"
    config_b = {
        "configurable": {"thread_id": thread_b},
        "recursion_limit": 20,
    }
    result_b = app.invoke(
        initial_state("如何设计工具 Agent", max_revisions=0),
        config=config_b,
    )

    print("\n=== 会话 B（修订上限为 0）===")
    print_result(result_b)
    if result_b["topic"] == result_a["topic"]:
        raise AssertionError("不同 thread_id 的状态发生串线")


def main() -> None:
    run_checkpoint_demo()


if __name__ == "__main__":
    main()
