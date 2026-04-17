"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.tools import tool
from typing import Any, Dict

from langchain_core.messages import SystemMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import SecretStr
from typing_extensions import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
import operator
import os

apiKey = SecretStr(os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=apiKey)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """

    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts `a` and `b`."""
    return a - b

tools =[add, subtract, multiply, divide]
# tools_by_name = {tool.name: tool for tool in tools}
tool_node = ToolNode(tools=tools)
model_with_tools = llm.bind_tools(tools)

class Context(TypedDict):
    """Context parameters for the calculator.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    session: str


# @dataclass
# class State:
#     """Input state for the calculator.
#
#     Defines the initial structure of incoming data.
#     See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
#     """
#     messages: str
#     llm_calls: 0

@dataclass
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # operator.add is used to concatenate lists [MSG1] -> [MSG1, MSG2]
    llm_calls: int

def llm_call(state: dict) -> Dict[str, Any]:
    """
    This is a description of my tool.
    It takes two parameters: param1 (string) and param2 (integer).
    """
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool(state: dict):
    """
    This is a description of my tool.
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tool_node.tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action, last message is built with ToolMessage
    if last_message.tool_calls:
        return "tool"

    # Otherwise, we stop (reply to the user)
    return END

# Define the graph
agent = StateGraph(MessagesState)

agent.add_node("llm_call", llm_call)
agent.add_node("tool", tool)

agent.add_edge(START, "llm_call")
agent.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool", END]
)

agent.add_edge("tool", "llm_call")
graph = agent.compile(name="calculator")
graph.invoke({"user_input":"My"})
