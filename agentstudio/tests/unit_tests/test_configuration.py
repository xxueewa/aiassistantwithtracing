import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langgraph.graph import END

from agents.calculator.graph import add, divide, multiply, should_continue, subtract


def test_calculator_tools() -> None:
    assert add.invoke({"a": 3, "b": 4}) == 7
    assert subtract.invoke({"a": 10, "b": 3}) == 7
    assert multiply.invoke({"a": 3, "b": 4}) == 12
    assert divide.invoke({"a": 12, "b": 4}) == 3


def test_should_continue_routes_to_end_when_no_tool_calls() -> None:
    from langchain_core.messages import AIMessage

    state = {"messages": [AIMessage(content="The answer is 7.")]}
    assert should_continue(state) == END


def test_should_continue_routes_to_tool_when_tool_call_present() -> None:
    from langchain_core.messages import AIMessage

    msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 3, "b": 4}, "type": "tool_call"}],
    )
    state = {"messages": [msg]}
    assert should_continue(state) == "tool"
