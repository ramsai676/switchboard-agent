"""Deciding which human to interrupt, and in what order.

Ranking blends three things:

  skill match   — does this person actually know the area?
  reliability   — have they answered when paged before?
  load          — spread pages around instead of burning out the one person
                  who always replies.

That last term matters more than it looks. A router that only optimises for
"most likely to answer" converges on paging the same helpful person forever,
which is exactly how internal help channels die.
"""

from __future__ import annotations

from dataclasses import dataclass

from .brain import stems
from .store import Expert, Store


@dataclass
class Candidate:
    expert: Expert
    score: float
    reason: str


def _skill_affinity(expert: Expert, skill: str | None, question: str) -> float:
    """1.0 for an exact tag hit, partial credit for word overlap with the question."""
    if skill and skill in expert.skills:
        return 1.0

    question_words = stems(question)
    best = 0.0
    for tag in expert.skills:
        tag_words = stems(tag.replace("-", " "))
        if not tag_words:
            continue
        overlap = len(question_words & tag_words) / len(tag_words)
        best = max(best, overlap)
    return best


def rank_experts(
    store: Store, question: str, skill: str | None, exclude: set[str] | None = None
) -> list[Candidate]:
    exclude = exclude or set()
    experts = [e for e in store.experts() if e.id not in exclude]
    if not experts:
        return []

    busiest = max((e.paged_count for e in experts), default=0)

    candidates: list[Candidate] = []
    for expert in experts:
        affinity = _skill_affinity(expert, skill, question)
        if affinity <= 0.0:
            continue  # never page someone with no plausible connection to the topic

        # Normalised inverse load: 1.0 when least-paged, 0.0 when most-paged.
        load_relief = 1.0 - (expert.paged_count / busiest) if busiest else 1.0

        score = (0.60 * affinity) + (0.25 * expert.reliability) + (0.15 * load_relief)

        if affinity >= 1.0:
            reason = f"owns “{skill}”"
        else:
            reason = f"adjacent to {', '.join(expert.skills[:2])}"
        if expert.reliability >= 0.8 and expert.paged_count:
            reason += f", answers {expert.reliability:.0%} of pages"

        candidates.append(Candidate(expert, round(score, 4), reason))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def escalation_chain(
    store: Store, question: str, skill: str | None, depth: int = 3,
    exclude: set[str] | None = None,
) -> list[Candidate]:
    """Who to try, in order, as each one times out."""
    return rank_experts(store, question, skill, exclude)[:depth]
