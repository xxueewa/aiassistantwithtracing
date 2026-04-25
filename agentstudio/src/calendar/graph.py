from dataclasses import dataclass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timezone
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from typing_extensions import TypedDict, Annotated, Literal
from typing import Any, Dict
import operator
import os

# NSDate = Foundation.NSDate
# NSCalendar = Foundation.NSCalendar
# store = EventKit.EKEventStore.alloc().init()
# store.requestAccessToEntityType_completion_(0, lambda granted, error: None)

# creds = Credentials.from_authorized_user_file("token.json")
# service = build("calendar", "v3", credentials=creds)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@dataclass
class ContextSchema:
    system_timezone: str
    convert_timezone: str

@dataclass
class event:
    summary: str
    start: dict
    end: dict

@tool
def calender_tool(start: str, end: str) -> list:
    """ Fetch User's Calendar Events In the Given Time Range

    Args:
        start: First string
        end: Second string
    """
    # start_date = NSDate.dateFromString_(start)
    # end_date = NSDate.dateFromString_(end)
    # predicate = store.predicateForEventsWithStartDate_endDate_calendars_(start_date, end_date, None)
    # events = store.eventsMatchingPredicate_(predicate)
    return []


@tool
def check_availability_tool(time_zone: str, calender_events: list, new_event: dict):
    """
    Provides a function to check availability based on a time zone.
    This function initiates a process to determine availability.

    Args:
        time_zone: Time zone string
        calender_events: List of events on the user's calendar
        new_event: Dictionary containing information about the new event, {start_time, end_time, title, time_zone}
    """
    print("start checking your availability")

apiKey = SecretStr(os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=apiKey)

tools =[calender_tool, check_availability_tool]
tool_node = ToolNode(tools=tools)
model_with_tools = llm.bind_tools(tools)

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
                        content="You are an assistant to help booking new meetins for me."
                                "Make sure to convert all the date and time to Time format: ISO 8601 with timezone (required)."
                                "Use tools to retrieve my existing calendar meetings and check availability."
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
    It takes two parameters: param1 (string) and param2 (integer).
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
graph = agent.compile(name="calendar_assistant")