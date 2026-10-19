"""Entry point for ``langgraph dev`` / LangGraph Studio / LangGraph Platform (see ``langgraph.json``).

The graph is compiled once, when the server imports this module. ``langgraph dev`` calls a graph
*factory* on its async event loop for every request and rejects blocking calls there, and building the
graph does blocking work (creating the research-cache folder, reading config files). Importing happens
outside the loop, so it is allowed. Agent nodes are synchronous, and LangGraph runs those in a worker
thread, so their network and disk I/O is fine.

In Studio, invoke with input ``{"user_input": "..."}``.
"""

from src.workflow.langgraph_workflow import make_studio_graph

graph = make_studio_graph()
