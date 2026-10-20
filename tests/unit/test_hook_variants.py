"""LinkedIn hook variants: generated in several styles, scored, best first, swappable."""

from src.agents.linkedin_writer import HookIdea, clean_hook, make_linkedin_writer, rank_hooks, swap_hook
from src.utils.content_optimization import linkedin_metrics
from src.utils.quality_validation import score_hook
from src.workflow.state_management import LinkedInPost
from tests.conftest import LINKEDIN_BODY, default_roles

STATE = {"request": {"intent": "linkedin", "topic": "fast replies", "refine_target": "linkedin"},
         "linkedin": {"text": "old"}}


def writer_reply(hook, options):
    reply = dict(default_roles()["linkedin_writer"])
    reply.update(hook=hook, body=LINKEDIN_BODY, hook_options=options)
    return reply


def test_score_hook_rules():
    assert score_hook("Most small shops lose leads to slow replies.") == (100.0, [])
    score, issues = score_hook("In today's fast world, speed matters for everyone in business.")
    assert score == 60 and "generic opener" in issues[0]
    assert score_hook("Hi")[0] == 70 and score_hook("x" * 300)[0] == 60
    assert score_hook("Want to boost sales? Try a seasonal special.")[0] == 80  # stock opener
    assert score_hook("Searches for caramel apples jumped 1,800%!")[0] == 90  # exclamation
    assert score_hook("y" * 180)[0] == 85  # long but still above the fold


def test_rank_hooks_cleans_dedupes_and_sorts():
    options = rank_hooks("In today's market, replies matter more than ever.", [
        HookIdea(style="hot take", text="Your follow-up email is too late [2] #Sales https://x.example"),
        HookIdea(style="question", text="in today's market, replies matter more than ever."),  # duplicate
        HookIdea(style="", text="   "),
    ])
    assert [o.style for o in options] == ["hot take", "drafted"]
    assert options[0].text == "Your follow-up email is too late" and options[0].score == 100
    assert options[1].issues and clean_hook("Line one\nline two") == "Line one"


def test_writer_leads_with_best_hook_and_keeps_alternatives(make_deps):
    roles = default_roles()
    roles["linkedin_writer"] = writer_reply("In today's world, replies matter.", [
        {"style": "story", "text": "Last spring a customer waited for our reply."},
        {"style": "hot take", "text": "Slow replies cost you more than ads do."},
    ])
    out = make_linkedin_writer(make_deps(roles))(STATE)
    post = LinkedInPost.model_validate(out["linkedin"])
    assert post.text.startswith("Last spring a customer waited for our reply.\n\n")
    assert post.hook == "Last spring a customer waited for our reply."
    assert [o.style for o in post.hook_options] == ["story", "hot take", "drafted"]
    assert any("Led with the 'story' hook" in n for n in post.provenance.notes)
    assert linkedin_metrics(post.text)["hook"] == post.hook


def test_invented_result_in_a_hook_never_leads(make_deps):
    roles = default_roles()
    roles["linkedin_writer"] = writer_reply("Most small shops lose leads to slow replies.", [
        {"style": "story", "text": "A bakery owner tripled her sales with 3 quick replies a day."},
    ])
    post = LinkedInPost.model_validate(make_linkedin_writer(make_deps(roles))(STATE)["linkedin"])
    story = next(o for o in post.hook_options if o.style == "story")
    assert post.hook == "Most small shops lose leads to slow replies."
    assert story.score == 70 and "tripled" in story.issues[0] and "3" in story.issues[0]


def test_unsupported_claims_ignore_years_and_accept_sourced_numbers():
    research = "Email reaches 4.73 billion users; replies within 5 minutes doubled conversions in 2025."
    assert score_hook("Email will reach 4.73 billion users by 2026.", research) == (100.0, [])
    assert score_hook("Fast replies doubled conversions.", research)[0] == 100
    assert score_hook("Fast replies quadrupled conversions.", research)[0] == 70
    assert score_hook("Fast replies quadrupled conversions.")[0] == 100  # no research given: not checked


def test_writer_keeps_drafted_hook_on_a_tie(make_deps):
    roles = default_roles()
    roles["linkedin_writer"] = writer_reply("Most small shops lose leads to slow replies.",
                                            [{"style": "question", "text": "How fast do you really reply?"}])
    post = make_linkedin_writer(make_deps(roles))(STATE)["linkedin"]
    assert post["hook"] == "Most small shops lose leads to slow replies."
    assert not any("Led with" in n for n in post["provenance"]["notes"])


def test_swap_hook_replaces_opening_and_keeps_format(make_deps):
    post = LinkedInPost.model_validate(make_linkedin_writer(make_deps())(STATE)["linkedin"])
    swapped = swap_hook(post, "Slow replies cost you more than ads do. [1]")
    assert swapped.text.startswith("Slow replies cost you more than ads do.\n\n")
    assert post.hook not in swapped.text and swapped.hook == "Slow replies cost you more than ads do."
    assert swapped.hashtags == post.hashtags and swapped.text.endswith(" ".join(post.hashtags))
    assert len(swapped.text) <= 1600 and swapped.text.count("#SmallBusiness") == 1

    edited = post.model_copy(update={"text": "My edited opener\n\n" + post.text.split("\n\n", 1)[1]})
    assert swap_hook(edited, "New opener here today").text.startswith("New opener here today\n\nLast spring")
    bare = LinkedInPost(text="Only a hook", hook="Only a hook")
    assert swap_hook(bare, "Replacement hook line").text == "Replacement hook line"
