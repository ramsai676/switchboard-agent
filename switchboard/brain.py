"""Inference for Switchboard.

Two jobs, both small and both explicitly fallible:

  1. classify(question)     -> which skill does this need?
  2. compose(...)           -> write the reply text for a given channel

Featherless.ai is the default backend (the buildathon ships credit for it) and it
speaks the OpenAI chat-completions shape, so any OpenAI-compatible endpoint works
by pointing FEATHERLESS_BASE_URL somewhere else.

If no key is configured the agent does NOT fall over — it drops to a deterministic
local backend. That matters for two reasons: a judge can clone the repo and run it
with zero secrets, and a paging agent that goes silent because an inference bill
lapsed is worse than one that routes on keyword overlap.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("switchboard.brain")

DEFAULT_MODEL = os.environ.get("FEATHERLESS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_BASE_URL = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def stem(word: str) -> str:
    """Crudest possible suffix stripping.

    It exists for one specific failure: a question asking about a "refund" must
    reach the expert whose skill tag is "refunds". Plain token overlap scores
    that pair at zero and the agent then reports that nobody covers the topic —
    which is how it looked before this was added. Consistency matters more than
    linguistic correctness here; both sides go through the same function, so
    "kubernetes" -> "kubernet" is harmless as long as it always happens."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def stems(text: str) -> set[str]:
    return {stem(t) for t in tokens(text)}


class Brain:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: float = 45.0) -> None:
        self.api_key = api_key or os.environ.get("FEATHERLESS_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or DEFAULT_BASE_URL
        self._timeout = timeout
        self.online = bool(self.api_key)
        if not self.online:
            logger.warning(
                "No FEATHERLESS_API_KEY set — running on the deterministic local backend. "
                "Routing still works; phrasing will be templated."
            )

    # ---- low level ------------------------------------------------------

    def _chat(self, system: str, user: str, max_tokens: int = 400,
              temperature: float = 0.3) -> str | None:
        if not self.online:
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                logger.warning("inference %s: %s", response.status_code, response.text[:200])
                return None
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("inference call failed; falling back to local backend")
            return None

    # ---- job 1: which skill does this question need? --------------------

    def classify(self, question: str, skills: list[str]) -> str | None:
        """Pick the skill tag best matching the question, or None if nothing fits."""
        if not skills:
            return None

        answer = self._chat(
            system=(
                "You route questions to the right subject-matter expert. "
                "Reply with ONLY a JSON object: {\"skill\": \"<one tag from the list>\"}. "
                "If no tag genuinely fits, use {\"skill\": null}. No prose."
            ),
            user=f"Tags: {json.dumps(skills)}\n\nQuestion: {question}",
            max_tokens=60,
            temperature=0.0,
        )
        if answer:
            picked = self._extract_skill(answer, skills)
            if picked is not None:
                return picked
        return self._classify_local(question, skills)

    @staticmethod
    def _extract_skill(raw: str, skills: list[str]) -> str | None:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                value = json.loads(match.group(0)).get("skill")
                if value is None:
                    return None
                if isinstance(value, str) and value in skills:
                    return value
            except (ValueError, AttributeError):
                pass
        # Model ignored the format but may still have named a tag.
        for skill in skills:
            if skill.lower() in raw.lower():
                return skill
        return None

    @staticmethod
    def _classify_local(question: str, skills: list[str]) -> str | None:
        """Token-overlap routing. Crude, but it never returns a tag that shares
        nothing with the question, which is the failure that actually hurts —
        paging the wrong human trains them to ignore the agent."""
        q = stems(question)
        best, best_score = None, 0
        for skill in skills:
            score = len(q & stems(skill.replace("-", " ")))
            if score > best_score:
                best, best_score = skill, score
        return best

    # ---- job 2: write the outgoing text ---------------------------------

    def compose_page(self, question: str, expert_name: str, channel_guide: str) -> str:
        """The message that lands in an expert's inbox. Must be short: this is an
        interruption, and a long interruption gets ignored."""
        written = self._chat(
            system=(
                "You draft short internal pages to subject-matter experts. "
                "2 sentences maximum. Lead with the question. No greeting, no sign-off, "
                "no apology for interrupting. Plain text.\n\n" + channel_guide
            ),
            user=f"Expert: {expert_name}\nQuestion from a colleague: {question}",
            max_tokens=160,
        )
        return written or (
            f"{expert_name} — you're the closest match for this one:\n\n"
            f"“{question}”\n\n"
            "Reply here with the answer and I'll relay it and remember it."
        )

    def compose_relay(self, question: str, answer: str, expert_name: str) -> str:
        """The answer going back to whoever asked."""
        written = self._chat(
            system=(
                "You relay an expert's answer back to the person who asked. "
                "Keep the expert's meaning exactly; tighten the wording only. "
                "Do not add caveats they did not make. Plain text, no preamble."
            ),
            user=f"Question: {question}\nExpert ({expert_name}) said: {answer}",
            max_tokens=400,
        )
        body = written or answer
        return f"{body}\n\n— answered by {expert_name}"

    def answer_from_knowledge(self, question: str, known_q: str, known_a: str) -> str:
        """Re-serve a learned answer, adapted to how this asker phrased it."""
        written = self._chat(
            system=(
                "Answer the user's question using ONLY the supplied known answer. "
                "Adapt the phrasing to their question. If the known answer does not "
                "actually cover it, reply exactly: INSUFFICIENT. Plain text."
            ),
            user=f"Known Q: {known_q}\nKnown A: {known_a}\n\nUser asks: {question}",
            max_tokens=400,
        )
        if not written or written.strip().upper().startswith("INSUFFICIENT"):
            return known_a
        return written
