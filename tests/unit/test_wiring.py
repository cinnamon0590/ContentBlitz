from src.agents import AgentDeps
from src.workflow import langgraph_workflow as lw


def test_deps_from_config_wires_real_clients(config):
    config.secrets.serpapi_api_key = "k"
    deps = AgentDeps.from_config(config)
    assert [p.name for p in deps.search.providers] == ["serpapi", "tavily", "perplexity"]
    assert deps.search.available_providers == ["serpapi"]
    assert deps.images.chain[0] == "gpt-image-1" and deps.validator.pass_threshold == 70
    assert deps.output_dir.exists()


def test_studio_graph_factory(monkeypatch, config):
    monkeypatch.setattr("src.core.config.get_config", lambda: config)
    graph = lw.make_studio_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"query_handler", "research_agent", "blog_writer", "linkedin_writer", "image_generator",
            "content_strategist", "quality_gate", "finalize"} <= nodes


def test_studio_module_compiles_the_graph_at_import(monkeypatch, config):
    """langgraph dev rejects blocking calls on its event loop, so the graph is built at import, not per request."""
    import importlib
    import json
    from pathlib import Path

    monkeypatch.setattr("src.core.config.get_config", lambda: config)
    import src.workflow.studio as studio

    studio = importlib.reload(studio)
    assert "research_agent" in studio.graph.get_graph().nodes
    manifest = json.loads((Path(__file__).resolve().parents[2] / "langgraph.json").read_text())
    assert manifest["graphs"]["contentblitz"] == "./src/workflow/studio.py:graph"
