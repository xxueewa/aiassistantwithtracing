"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.tools import tool, ToolRuntime
from typing import Any, Dict

from langchain_core.messages import SystemMessage, ToolMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langmem.short_term import SummarizationNode, RunningSummary
from pydantic import SecretStr
from typing_extensions import TypedDict, Annotated, Literal
import operator
import os

apiKey = SecretStr(os.environ["OPENAI_API_KEY"])

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=apiKey)
llm_with_summary = llm.bind(max_tokens=128)

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

class State(MessagesState):
    # extend from MessageState: messages # operator.add is used to concatenate lists [MSG1] -> [MSG1, MSG2]
    context: dict[str, RunningSummary]

class LLMMessagesState(TypedDict):
    context: dict[str, RunningSummary]
    summarized_messages: list[AnyMessage]

initial_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "Summarize the conversation below. Focus on key decisions, "
        "user preferences, and important facts. Be concise.\n\n"
        "Conversation:\n{messages}"
    )
])

existing_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "Update the existing summary with the new messages. "
        "Keep key decisions, user preferences, and important facts. "
        "Discard the outdated information from existing_summary.\n\n"
        "Existing summary:\n{existing_summary}\n\n"
        "New messages:\n{messages}"
    )
])

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=llm_with_summary,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
    initial_summary_prompt=initial_summary_prompt,  # ← first summary
    existing_summary_prompt=existing_summary_prompt,  # ← update summary
)

def llm_call(state: LLMMessagesState) -> Dict[str, Any]:
    """
    This is a description of my tool.
    It takes two parameters: param1 (string) and param2 (integer).
    """
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs. Your name is @$%#$^"
                    )
                ]
                + state["summarized_messages"]
            )
        ]
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
agent = StateGraph(State)

agent.add_node("summarization", summarization_node)
agent.add_node("llm_call", llm_call)
agent.add_node("tool", tool_node)

agent.add_edge(START, "summarization")
agent.add_edge("summarization", "llm_call")

agent.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool", END]
)

agent.add_edge("tool", "summarization")

graph = agent.compile(name="calculator")