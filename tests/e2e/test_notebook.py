"""Execute the walkthrough notebook top-to-bottom in offline mode (skips the Colab-only pip cell)."""

import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "contentblitz_walkthrough.ipynb"


def test_notebook_executes_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTENTBLITZ_OFFLINE", "1")
    nb = copy.deepcopy(nbformat.read(NOTEBOOK, as_version=4))
    nb.cells = [c for c in nb.cells if not (c.cell_type == "code" and c.source.lstrip().startswith("# Colab only"))]
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "all routes behaved as expected" in text
