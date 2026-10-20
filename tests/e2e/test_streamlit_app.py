"""UI smoke tests: drive the real Streamlit app with an offline ContentBlitz instance."""

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from src.core.workflow import ContentBlitz
from tests.conftest import default_roles

APP = str(Path(__file__).resolve().parents[2] / "src" / "web_app" / "streamlit_app.py")


@pytest.fixture
def ui(config, make_deps):
    def _make(roles=None):
        at = AppTest.from_file(APP, default_timeout=60)
        at.session_state["app"] = ContentBlitz(config, deps=make_deps(roles or default_roles("campaign")),
                                               checkpointer=InMemorySaver())
        at.session_state["thread_id"] = "ui-test"
        return at

    return _make


def test_empty_app_renders_all_tabs(ui):
    at = ui().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 Chat", "📝 Content Dashboard", "🔎 Research", "🖼️ Images",
                                          "📦 Export", "⚙️ Settings"]
    assert any("OPENAI_API_KEY" in w.value for w in at.warning)
    assert any("Nothing to preview" in i.value for i in at.info)


def test_campaign_turn_populates_every_tab(ui):
    at = ui().run()
    at.chat_input[0].set_value("Create a full campaign about our bakery newsletter").run()
    assert not at.exception
    state = at.session_state["app"].state("ui-test")
    assert state["blog"] and state["linkedin"] and state["image"]
    markdown = " ".join(m.value for m in at.markdown)
    assert "SEO Blog Post" in markdown and "LinkedIn Post" in markdown
    assert at.metric and "Quality score" in [m.label for m in at.metric]  # dashboard
    assert any("Sources (3)" in s.value for s in at.header)  # research panel
    labels = [b.label for b in at.button]
    assert "Save edits & re-score" in " ".join(labels)


def test_example_button_and_settings_save(ui):
    at = ui(default_roles("general")).run()
    at.button[1].click().run()  # first example prompt (button 0 is "New conversation")
    assert not at.exception
    assert len(at.session_state["app"].history("ui-test")) == 2
    name_input = next(t for t in at.text_input if t.label == "Business name")
    name_input.set_value("Bean There Café")
    next(b for b in at.button if b.label == "Save brand profile").click().run()
    assert at.session_state["brand"]["name"] == "Bean There Café"


def test_learn_brand_voice_confirms_after_the_rerun(ui):
    at = ui(default_roles("general")).run()
    sample = next(t for t in at.text_area if t.label.startswith("Paste 2–3 paragraphs"))
    sample.set_value("We bake with love every morning. Come say hi.")
    next(b for b in at.button if b.label == "Analyze voice").click().run()
    assert not at.exception
    assert at.session_state["brand"]["tone"] == "warm"
    assert any("Learned tone: **warm**" in s.value for s in at.success)  # still visible after st.rerun()


def test_dashboard_editor_follows_stored_content(ui):
    """A hook swap updates the stored post and the editor.

    The stale-editor bug this guards against lived in the browser's widget state, which AppTest does not model,
    so it was verified in Chrome; this test covers the server side of the same flow.
    """
    at = ui(default_roles("linkedin")).run()
    at.chat_input[0].set_value("Write a LinkedIn post about fast replies").run()
    app = at.session_state["app"]
    post = app.state("ui-test")["linkedin"]
    editor = next(t for t in at.text_area if t.label == "Edit content")
    assert editor.value.startswith(post["hook"])

    new_hook = next(o["text"] for o in post["hook_options"] if o["text"] != post["hook"])
    radio = next(r for r in at.radio if r.label == "Opening line")
    radio.set_value(next(i for i, o in enumerate(post["hook_options"]) if o["text"] == new_hook)).run()
    next(b for b in at.button if b.label == "Use this hook").click().run()
    assert not at.exception
    assert app.state("ui-test")["linkedin"]["text"].startswith(new_hook)
    editor = next(t for t in at.text_area if t.label == "Edit content")
    assert editor.value.startswith(new_hook)  # the editor shows the swapped post, not the stale one
