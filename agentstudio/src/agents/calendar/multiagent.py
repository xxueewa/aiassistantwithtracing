"""
# supervisor.py
from agents.researcher import researcher_graph
from agents.writer import writer_graph

def supervisor_node(state):
    result = researcher_graph.invoke(state)
    ...

supervisor = StateGraph(...)
supervisor.add_node("researcher", researcher_graph)  # subgraph
supervisor.add_node("writer", writer_graph)          # subgraph



"""