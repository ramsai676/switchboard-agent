"""Tests for the parts that decide who gets interrupted.

No network here — the Caspian gateway is exercised separately by demo/live_check.py.
These cover the logic that would silently rot: retrieval confidence, routing order,
escalation exclusion, and the deflection metric.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from switchboard.brain import Brain            # noqa: E402
from switchboard.knowledge import KnowledgeBase  # noqa: E402
from switchboard.router import escalation_chain, rank_experts  # noqa: E402
from switchboard.store import Store            # noqa: E402


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture()
def staffed(store):
    store.add_expert("Priya", "email", "priya@example.com", ["billing", "refunds"])
    store.add_expert("Marco", "email", "marco@example.com", ["kubernetes", "deploys"])
    store.add_expert("Ada", "discord", "ada#1234", ["kubernetes", "networking"])
    return store


# ---- retrieval ---------------------------------------------------------

def test_knowledge_returns_nothing_when_empty(store):
    assert KnowledgeBase(store).best("how do I issue a refund?") is None


def test_knowledge_matches_a_paraphrase(store):
    store.learn(
        "How do I issue a refund for a duplicate charge?",
        "Use the Refunds tab, pick 'duplicate', it clears in 3 days.",
        "refunds", "Priya",
    )
    hit = KnowledgeBase(store).best("how do I refund a duplicate charge")
    assert hit is not None
    assert "Refunds tab" in hit.answer_text


def test_knowledge_refuses_an_unrelated_question(store):
    """The important negative case: a populated KB must not answer off-topic
    questions, or no human ever gets paged to fill the real gap."""
    store.learn(
        "How do I issue a refund for a duplicate charge?",
        "Use the Refunds tab.", "refunds", "Priya",
    )
    assert KnowledgeBase(store).best("how do I roll back a kubernetes deploy") is None


# ---- routing -----------------------------------------------------------

def test_exact_skill_owner_ranks_first(staffed):
    ranked = rank_experts(staffed, "our k8s deploy is stuck", "kubernetes")
    assert ranked
    assert ranked[0].expert.name in {"Marco", "Ada"}
    assert all(c.expert.name != "Priya" for c in ranked)


def test_experts_with_no_connection_to_topic_are_never_paged(staffed):
    ranked = rank_experts(staffed, "how do I issue a refund", "refunds")
    assert [c.expert.name for c in ranked] == ["Priya"]


def test_load_balancing_prefers_the_less_paged_expert(staffed):
    marco = next(e for e in staffed.experts() if e.name == "Marco")
    for _ in range(6):
        staffed.bump_expert(marco.id, paged=True, answered=True)

    ranked = rank_experts(staffed, "kubernetes question", "kubernetes")
    assert ranked[0].expert.name == "Ada", "should spread load to the quieter expert"


def test_escalation_excludes_already_paged_experts(staffed):
    first = escalation_chain(staffed, "kubernetes question", "kubernetes")[0]
    nxt = escalation_chain(
        staffed, "kubernetes question", "kubernetes", exclude={first.expert.id}
    )
    assert nxt
    assert all(c.expert.id != first.expert.id for c in nxt)


def test_escalation_runs_out_cleanly(staffed):
    everyone = {e.id for e in staffed.experts()}
    assert escalation_chain(staffed, "kubernetes question", "kubernetes",
                            exclude=everyone) == []


# ---- classification falls back without a key ---------------------------

def test_local_classifier_picks_the_overlapping_tag():
    brain = Brain(api_key=None)
    assert brain.classify("our kubernetes cluster is down",
                          ["billing", "kubernetes", "design"]) == "kubernetes"


def test_local_classifier_returns_none_when_nothing_matches():
    brain = Brain(api_key=None)
    assert brain.classify("what's for lunch", ["billing", "kubernetes"]) is None


# ---- metrics -----------------------------------------------------------

def test_deflection_rate_tracks_memory_answers(store):
    q1 = store.add_question("q one", "email", None, None, "asker", None)
    store.resolve_question(q1, "a", "memory", "answered_from_memory")
    q2 = store.add_question("q two", "email", None, None, "asker", None)
    store.resolve_question(q2, "a", "exp_1", "resolved")

    stats = store.stats()
    assert stats["questions_total"] == 2
    assert stats["deflected"] == 1
    assert stats["deflection_rate"] == 0.5


def test_page_lifecycle_is_recoverable_after_restart(store, tmp_path):
    """State must survive a restart — an escalation timer that forgets its pages
    is worse than none."""
    e = store.add_expert("Priya", "email", "priya@example.com", ["billing"])
    q = store.add_question("q", "email", None, None, "asker", "billing")
    p = store.add_page(q, e.id, "email", "conv_123")
    store.close()

    reopened = Store(tmp_path / "test.db")
    assert reopened.open_page_for_expert_conversation("conv_123")["id"] == p
    assert reopened.open_questions()[0]["id"] == q
    reopened.close()
