import csv
import os
import pytest

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from openevals.llm import create_llm_as_judge
from langgraph_sdk import get_sync_client
from langgraph_sdk.client import SyncRunsClient
from openevals.prompts import TOOL_SELECTION_PROMPT, CORRECTNESS_PROMPT

# https://github.com/langchain-ai/openevals/tree/main
CSV_PATH = "tool_selection_test_set.csv"
agent_client = get_sync_client(url=os.getenv("LANGGRAPH_URL", "http://localhost:2024"))
runs_client: SyncRunsClient = agent_client.runs

# Instantiate once so the LLM client is not recreated per example
tool_selection_evaluator = create_llm_as_judge(
    prompt=TOOL_SELECTION_PROMPT,
    model="openai:o3-mini",
    feedback_key="tool_selection_llm",
)


def target(inputs: dict) -> dict:
    thread = agent_client.threads.create()
    run = runs_client.create(
        thread_id = thread["thread_id"],
        assistant_id = "assistant",
        input = {"messages": [{"role": "human", "content": inputs["input"]}]},
    )
    runs_client.join(thread["thread_id"], run["run_id"])
    thread_state = agent_client.threads.get_state(thread["thread_id"])
    messages = thread_state["values"]["messages"]

    tool_call_names = [
        tool_call["name"]
        for message in messages
        for tool_call in (message.get("tool_calls") or [])
    ]
    return {
        "output": tool_call_names[0] if tool_call_names else None,
        "messages": messages,
    }


@pytest.mark.langsmith
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    return tool_selection_evaluator(
        inputs=inputs,
        outputs=outputs["messages"],
        reference_outputs=reference_outputs,
    )


if __name__ == "__main__":
    ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    # Create dataset only if it doesn't already exist
    datasets = list(ls_client.list_datasets(dataset_name="assistant_tool_selection"))
    if datasets:
        dataset = datasets[0]
    else:
        dataset = ls_client.create_dataset(
            dataset_name="assistant_tool_selection",
            description="The dataset used to test the tool selection accuracy.",
        )
        inputs = []
        outputs = []
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                inputs.append({"input": row["input"].strip()})
                outputs.append({"output": row["output"].strip()})
        ls_client.create_examples(
            dataset_id=dataset.id,
            inputs=inputs,
            outputs=outputs,
        )

    experiment_results = ls_client.evaluate(
        target,
        data="assistant_tool_selection",
        evaluators=[correctness_evaluator],
        experiment_prefix="assistant_experiment_tool_selection",
        max_concurrency=2,
    )


    print(experiment_results)
    print(experiment_results.to_pandas())