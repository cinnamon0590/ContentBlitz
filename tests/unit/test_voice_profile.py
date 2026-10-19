"""Richer brand voice: persona, emoji level and sentence length — learned, prompted and checked."""

from src.core import prompts
from src.utils.content_optimization import emoji_level, voice_metrics
from src.utils.quality_validation import brand_voice_check, learn_brand_voice
from src.workflow.state_management import BrandProfile

SHORT = "We bake at dawn. Come early. Bread sells out fast. Say hi to Sam! 🥐 Fresh daily. 🍞"


def test_voice_metrics_and_emoji_level():
    m = voice_metrics(SHORT + " #Bakery #Fresh")
    assert m["emoji_count"] == 2 and 2 <= m["avg_sentence_words"] <= 4
    assert voice_metrics("") == {"avg_sentence_words": 0.0, "emoji_per_100_words": 0.0, "emoji_count": 0}
    assert [emoji_level(v) for v in (0, 1.5, 6)] == ["none", "light", "heavy"]


def test_emoji_rules_follow_brand_level():
    heavy_text = "Fresh bread today 🥐🍞🎉 come by soon 😀 we open early ☕"
    for level, penalized in (("none", True), ("light", True), ("heavy", False)):
        _, suggestions, _ = brand_voice_check(heavy_text, BrandProfile(emoji_level=level))
        assert any("emoji" in s.lower() for s in suggestions) is penalized, level


def test_sentence_length_rule():
    long_text = ("Our bakery has served the neighborhood for many years with bread that we shape by hand every "
                 "single morning before the sun comes up and the first customers arrive at the door. ") * 3
    score, suggestions, metrics = brand_voice_check(long_text, BrandProfile(avg_sentence_words=8))
    assert score <= 85 and "Shorten sentences" in suggestions[-1] and metrics["avg_sentence_words"] > 20
    _, suggestions, _ = brand_voice_check(long_text, BrandProfile(avg_sentence_words=0))
    assert not any("sentences to match" in s for s in suggestions)
    _, suggestions, _ = brand_voice_check(SHORT, BrandProfile(avg_sentence_words=20, emoji_level="heavy"))
    assert "Vary and lengthen" in suggestions[-1]


def test_learn_brand_voice_measures_style(make_llm):
    llm = make_llm(lambda m: '{"tone": "warm", "archetype": "Storyteller"}')
    profile = learn_brand_voice(SHORT, BrandProfile(), llm, prompts.BRAND_VOICE_LEARNER)
    assert profile.voice_archetype == "storyteller" and profile.emoji_level == "heavy"
    assert 2 <= profile.avg_sentence_words <= 4
    odd = learn_brand_voice("Plain text with no emoji at all.", BrandProfile(voice_archetype="educator"),
                            make_llm(lambda m: '{"archetype": "pirate"}'), prompts.BRAND_VOICE_LEARNER)
    assert odd.voice_archetype == "educator" and odd.emoji_level == "none"  # invalid persona ignored


def test_brand_block_includes_persona_and_style():
    block = prompts.brand_block(BrandProfile(voice_archetype="data-driven", emoji_level="none",
                                             avg_sentence_words=12))
    assert "data-driven — anchor every point in evidence" in block
    assert "about 12 words" in block and "never use emojis" in block
    default = prompts.brand_block(BrandProfile())
    assert "none (match the tone above)" in default and "mix short and medium sentences" in default
