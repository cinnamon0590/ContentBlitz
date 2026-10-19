import pytest
from pydantic import BaseModel

from src.integrations.openai_client import (
    LLMService,
    ModerationClient,
    ProviderUnavailableError,
    _extract_json,
    create_chat_model,
    message_text,
)
from src.utils.observability import CostTracker
from tests.conftest import FailingChatModel, FakeChatModel


class Pick(BaseModel):
    choice: str
    score: int = 0


def test_text_and_tracking(config):
    tracker = CostTracker()
    llm = LLMService(config, tracker=tracker, chat_models={"smart": FakeChatModel(responder=lambda m: "hello")})
    assert llm.text("sys", "user") == "hello"
    assert llm.provider == "fake" and llm.describe("fast") == "fake:fake-fast"
    assert tracker.summary()["llm"]["fake"]["calls"] == 1
    # tier not injected reuses the injected model
    assert llm.model("fast") is llm.model("smart")


def test_structured_via_json_fallback(config):
    llm = LLMService(config, chat_models={"fast": FakeChatModel(
        responder=lambda m: 'Sure! ```json\n{"choice": "blog", "score": 3}\n```')})
    result = llm.structured(Pick, "sys", "user")
    assert result == Pick(choice="blog", score=3)


def test_structured_returns_none_on_garbage(config):
    llm = LLMService(config, chat_models={"fast": FakeChatModel(responder=lambda m: "no json here")})
    assert llm.structured(Pick, "sys", "user") is None


def test_fallback_model_used_when_primary_fails(config):
    llm = LLMService(config, chat_models={"smart": FailingChatModel()},
                     fallback_models={"smart": [FakeChatModel(responder=lambda m: "from fallback")]})
    assert llm.text("s", "u") == "from fallback"


def test_all_candidates_fail_raises(config):
    llm = LLMService(config, chat_models={"smart": FailingChatModel()})
    with pytest.raises(RuntimeError, match="All LLM candidates failed"):
        llm.text("s", "u")


def test_bind_tools_with_fallbacks(config):
    llm = LLMService(config, chat_models={"smart": FakeChatModel(responder=lambda m: "x")},
                     fallback_models={"smart": [FakeChatModel(responder=lambda m: "y")]})
    assert llm.bind_tools([]).invoke("hi").content == "x"


def test_create_chat_model_requires_keys(config):
    with pytest.raises(ProviderUnavailableError):
        create_chat_model("openai", "gpt-4o", 0, config)
    with pytest.raises(ProviderUnavailableError):
        create_chat_model("anthropic", "x", 0, config)
    with pytest.raises(ProviderUnavailableError):
        create_chat_model("gemini", "x", 0, config)
    with pytest.raises(ProviderUnavailableError):
        create_chat_model("nope", "x", 0, config)


def test_create_openai_model_with_key(config):
    config.secrets.openai_api_key = "sk-test"
    model = create_chat_model("openai", "gpt-4o-mini", 0.1, config)
    assert model.model_name == "gpt-4o-mini"
    llm = LLMService(config)
    assert llm.provider == "openai" and llm.model_name("smart") == "gpt-4o"
    assert len(llm.fallbacks("smart")) == 1


def test_extract_json_variants():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('prefix {"a": [1]} suffix') == {"a": [1]}
    assert _extract_json("noise [1, 2] noise") == [1, 2]
    with pytest.raises(ValueError):
        _extract_json("nothing")


def test_message_text_handles_content_blocks():
    from langchain_core.messages import AIMessage

    assert message_text(AIMessage(content=[{"type": "text", "text": "a"}, "b"])) == "ab"


class _FakeModerations:
    def __init__(self, flagged):
        self.flagged = flagged

    def create(self, model, input):
        class Cats:
            def model_dump(self_inner):
                return {"violence": self.flagged, "hate": False}

        class Item:
            flagged = self.flagged
            categories = Cats()

        class Resp:
            results = [Item()]

        return Resp()


class _FakeOpenAI:
    def __init__(self, flagged=False, boom=False):
        self.moderations = _FakeModerations(flagged)
        if boom:
            self.moderations.create = lambda **k: (_ for _ in ()).throw(RuntimeError("down"))


def test_moderation_flagged_and_clean():
    assert ModerationClient("k", client=_FakeOpenAI(flagged=True)).check("bad") == (True, ["violence"])
    assert ModerationClient("k", client=_FakeOpenAI(flagged=False)).check("fine") == (False, [])


def test_moderation_fails_open_and_disabled():
    assert ModerationClient("k", client=_FakeOpenAI(boom=True)).check("x") == (False, [])
    assert ModerationClient(None).check("x") == (False, [])
    assert ModerationClient("k", enabled=False, client=_FakeOpenAI(True)).check("x") == (False, [])
