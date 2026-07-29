"""第9章完整示例：LCEL 问答链与 LangChain 工具 Agent。

运行方式：
    python examples/09_langchain_quickstart.py chain
    python examples/09_langchain_quickstart.py agent "第10章讲什么？再算 18*24"

示例工具只读取内置课程目录，并使用 AST 计算四则运算，不访问网络或文件。
"""

from __future__ import annotations

import ast
import json
import operator
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI


COURSES = {
    4: "手写第一个 Agent",
    5: "让 Agent 拥有记忆",
    6: "让 Agent 使用工具",
    7: "让 Agent 学会规划",
    8: "Agent 行为工程",
    9: "LangChain 快速上手",
    10: "LangGraph 状态机",
    11: "Dify 可视化构建",
    12: "RAG 知识库 Agent",
    13: "多 Agent 协作",
    14: "毕业项目",
}

MAX_AGENT_ROUNDS = 6
MAX_TOOL_RESULT_CHARS = 2000


def require_env(name: str) -> str:
    """读取必需环境变量，缺失时给出可操作的错误。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请参考 .env.example")
    return value


def build_model() -> ChatOpenAI:
    """集中创建模型，避免链和 Agent 使用不同配置。"""
    return ChatOpenAI(
        api_key=require_env("API_KEY"),
        base_url=os.getenv("BASE_URL") or None,
        model=require_env("MODEL"),
        temperature=0,
        timeout=30,
        max_retries=2,
    )


def build_chain(model: ChatOpenAI | None = None):
    """把提示模板、模型和字符串解析器连成 LCEL 管道。"""
    model = model or build_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是耐心的 Python 助教。只回答学习问题；"
                "不知道时明确说不知道，不编造资料。",
            ),
            ("human", "请用不超过 120 字解释：{question}"),
        ]
    )
    return prompt | model | StrOutputParser()


@tool
def lookup_course(chapter: int) -> str:
    """查询第4到第14章的课程标题；chapter 必须是整数章号。"""
    if chapter < 4 or chapter > 14:
        return json.dumps(
            {"ok": False, "error": "章号必须在 4 到 14 之间"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"ok": True, "chapter": chapter, "title": COURSES[chapter]},
        ensure_ascii=False,
    )


ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(node: ast.AST) -> int | float:
    """递归计算白名单 AST，拒绝名称、调用、属性等危险节点。"""
    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("指数绝对值不能超过 10")
        result = ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if abs(result) > 1_000_000_000:
            raise ValueError("结果绝对值过大")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
        return ALLOWED_UNARY_OPERATORS[type(node.op)](
            evaluate_expression(node.operand)
        )
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


@tool
def safe_calculate(expression: str) -> str:
    """计算只含数字、括号和基本运算符的表达式，不执行 Python 代码。"""
    if not expression or len(expression) > 100:
        return json.dumps(
            {"ok": False, "error": "表达式长度必须在 1 到 100 之间"},
            ensure_ascii=False,
        )
    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate_expression(tree)
        return json.dumps(
            {"ok": True, "expression": expression, "result": result},
            ensure_ascii=False,
        )
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return json.dumps(
            {"ok": False, "error": str(exc)},
            ensure_ascii=False,
        )


TOOLS: list[BaseTool] = [lookup_course, safe_calculate]
TOOL_MAP = {item.name: item for item in TOOLS}


def execute_tool_call(tool_call: dict[str, Any]) -> str:
    """校验工具名并执行一次调用，统一输出为有限长度的字符串。"""
    name = tool_call.get("name", "")
    selected = TOOL_MAP.get(name)
    if selected is None:
        result = json.dumps(
            {"ok": False, "error": f"未知工具：{name}"},
            ensure_ascii=False,
        )
    else:
        try:
            result = str(selected.invoke(tool_call.get("args", {})))
        except Exception as exc:  # 工具错误要回传模型，而不是让循环崩溃
            result = json.dumps(
                {"ok": False, "error": f"工具执行失败：{exc}"},
                ensure_ascii=False,
            )
    return result[:MAX_TOOL_RESULT_CHARS]


def run_tool_agent(
    question: str,
    model: ChatOpenAI | None = None,
    max_rounds: int = MAX_AGENT_ROUNDS,
) -> str:
    """显式实现 bind_tools Agent 循环，便于观察每一轮消息。"""
    if not question.strip():
        raise ValueError("问题不能为空")
    if len(question) > 4000:
        raise ValueError("问题不能超过 4000 字符")

    model_with_tools = (model or build_model()).bind_tools(TOOLS)
    messages = [
        SystemMessage(
            content=(
                "你是课程助教。查询章名或计算时必须使用工具；"
                "不得虚构工具结果；工具返回错误时向用户解释。"
            )
        ),
        HumanMessage(content=question),
    ]

    for round_number in range(1, max_rounds + 1):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return str(response.content)

        print(f"[第 {round_number} 轮] 模型请求 {len(response.tool_calls)} 个工具")
        for tool_call in response.tool_calls:
            result = execute_tool_call(tool_call)
            print(f"  - {tool_call['name']}({tool_call.get('args', {})})")
            print(f"    -> {result}")
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    raise RuntimeError(f"达到最大轮数 {max_rounds}，Agent 仍未结束")


def run_chain_demo() -> None:
    """运行最小 LCEL 示例。"""
    question = "LCEL 中的竖线符号有什么作用？"
    answer = build_chain().invoke({"question": question})
    print(f"问题：{question}\n回答：{answer}")


def run_agent_demo(question: str) -> None:
    """运行工具 Agent 示例。"""
    answer = run_tool_agent(question)
    print(f"\n问题：{question}\n最终回答：{answer}")


def main() -> None:
    load_dotenv()
    mode = sys.argv[1] if len(sys.argv) > 1 else "agent"
    if mode == "chain":
        run_chain_demo()
        return
    if mode == "agent":
        question = (
            " ".join(sys.argv[2:])
            if len(sys.argv) > 2
            else "第10章讲什么？再帮我计算 18*24。"
        )
        run_agent_demo(question)
        return
    raise SystemExit("用法：python examples/09_langchain_quickstart.py [chain|agent] [问题]")


if __name__ == "__main__":
    main()
