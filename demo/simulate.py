"""Run the whole Switchboard lifecycle against a fake gateway.

This exercises the real agent code — routing, paging, escalation, relay, learning,
deflection — with the Caspian transport swapped for an in-memory double. Nothing
here is mocked *inside* the agent; only the network edge is replaced.

    python demo/simulate.py

Useful for judges (no secrets, no channels, runs in a second) and as the
regression harness for the paging logic.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from switchboard.agent import Switchboard  # noqa: E402
from switchboard.brain import Brain        # noqa: E402
from switchboard.store import Store        # noqa: E402

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
BLUE, GREEN, YELLOW, MAGENTA = "\033[34m", "\033[32m", "\033[33m", "\033[35m"


class FakeMessage:
    """Stands in for caspian_sdk.Message."""

    def __init__(self, gateway, mid, conversation_id, channel, text, sender):
        self.id = mid
        self.conversation_id = conversation_id
        self.connection_id = f"conn_{channel}"
        self.customer_id = "cust_demo"
        self.agent_id = "agent_demo"
        self.channel = channel
        self.sender = sender
        self.subject = None
        self.text = text
        self.html = None
        self.media = []
        self._gateway = gateway

    def reply(self, text=None, html=None, blocks=None, media=None):
        self._gateway.outbound(self.conversation_id, self.channel, text, blocks)
        return {"id": "msg_reply"}

    def react(self, emoji):
        print(f"      {DIM}· reacted {emoji}{RESET}")
        return {"reacted": True}

    def typing(self):
        return None


class FakeGateway:
    """In-memory stand-in for CommClient. Same surface the agent actually uses."""

    def __init__(self):
        self._counter = 0
        self.conversations: dict[str, str] = {}   # conversation_id -> channel
        self.message_channel: dict[str, str] = {}
        self._on_message = None
        self._on_interaction = None

    # -- surface the agent calls --
    def on_message(self, fn):
        self._on_message = fn
        return fn

    def on_interaction(self, fn):
        self._on_interaction = fn
        return fn

    def channel_guide(self, channel):
        return ""

    def connect_email(self, username=None, **kw):
        return {"id": "conn_email", "status": "active",
                "address": f"{username}@agents.trycaspianai.com"}

    def connect_telegram(self, bot_token, **kw):
        return {"id": "conn_telegram", "status": "active"}

    def _next(self, prefix):
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    def initiate(self, connection_id, recipient, text):
        """Cold-start — the capability the whole design rests on."""
        channel = connection_id.replace("conn_", "")
        conversation_id = self._next("conv")
        self.conversations[conversation_id] = channel
        print(f"  {MAGENTA}→ PAGE{RESET} {recipient} {DIM}via {channel}{RESET}")
        print(textwrap.indent(textwrap.fill(text, 72), "      "))
        return {"conversation_id": conversation_id}

    def send_message(self, conversation_id, text=None, html=None, blocks=None, media=None):
        self.outbound(conversation_id, self.conversations.get(conversation_id, "?"), text, blocks)
        return {"id": self._next("msg")}

    def reply(self, message_id, text=None, html=None, blocks=None, media=None):
        channel = self.message_channel.get(message_id, "?")
        self.outbound(message_id, channel, text, blocks)
        return {"id": self._next("msg")}

    def outbound(self, where, channel, text, blocks=None):
        if text:
            print(f"  {BLUE}← SENT{RESET} {DIM}[{channel}]{RESET}")
            print(textwrap.indent(textwrap.fill(text, 72), "      "))
        if blocks:
            labels = [
                b["text"] for block in blocks for b in block.get("buttons", [])
            ] or ["(rich blocks)"]
            print(f"      {DIM}buttons: {' | '.join(labels)}{RESET}")

    def listen(self, **kw):
        raise NotImplementedError("simulation drives events directly")

    # -- test driver --
    def deliver(self, board, channel, text, sender_name, conversation_id=None):
        conversation_id = conversation_id or self._next("conv")
        self.conversations.setdefault(conversation_id, channel)
        mid = self._next("msg")
        self.message_channel[mid] = channel
        print(f"\n  {GREEN}✉ {sender_name}{RESET} {DIM}[{channel}]{RESET}: {text}")
        board.handle_message(
            FakeMessage(self, mid, conversation_id, channel, text,
                        {"name": sender_name, "address": sender_name})
        )
        return conversation_id


def banner(step, title):
    print(f"\n{BOLD}{'─' * 76}{RESET}")
    print(f"{BOLD}  {step}. {title}{RESET}")
    print(f"{BOLD}{'─' * 76}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=None,
        help="write to this database instead of a throwaway one "
             "(use switchboard.db to populate the dashboard)",
    )
    args = parser.parse_args()

    db = Path(args.db) if args.db else Path(tempfile.mkdtemp()) / "sim.db"
    if args.db and db.exists():
        db.unlink()  # replay from clean state so the metrics tell the real story
    store = Store(db)
    gateway = FakeGateway()

    board = Switchboard(client=gateway, store=store, brain=Brain(api_key=None),
                        page_timeout=1.0)
    board.connections = {"email": "conn_email", "telegram": "conn_telegram"}

    store.add_expert("Priya", "email", "priya@example.com", ["billing", "refunds"])
    store.add_expert("Marco", "email", "marco@example.com", ["kubernetes", "deploys"])
    store.add_expert("Ada", "email", "ada@example.com", ["kubernetes", "networking"])

    # ---------------------------------------------------------------
    banner(1, "A question nobody has asked before")
    print(f"  {DIM}Sam asks on Telegram. Switchboard knows nothing, so it finds the"
          f" owner\n  and cold-starts a conversation with them over email.{RESET}")
    q1 = "How do I issue a refund for a duplicate charge?"
    gateway.deliver(board, "telegram", q1, "Sam")

    # ---------------------------------------------------------------
    banner(2, "The expert answers on their own channel")
    page = store.pages_for_question(store.open_questions()[0]["id"])[-1]
    gateway.deliver(
        board, "email",
        "Refunds tab, pick 'duplicate', it clears in 3 working days. "
        "Anything over 500 needs a manager to approve.",
        "priya@example.com", conversation_id=page["conversation_id"],
    )

    # ---------------------------------------------------------------
    banner(3, "The same question again — from a different person, different channel")
    print(f"  {DIM}No human is paged this time. This is the compounding step.{RESET}")
    gateway.deliver(board, "email", "how do i refund a duplicate charge?", "Dana")

    # ---------------------------------------------------------------
    banner(4, "Escalation: the first expert declines")
    q2 = "Our kubernetes deploy is stuck rolling out, what do I check?"
    gateway.deliver(board, "telegram", q2, "Sam")

    open_q = store.open_questions()[0]
    first_page = store.pages_for_question(open_q["id"])[-1]
    first_name = store.expert(first_page["expert_id"]).name
    print(f"\n  {YELLOW}{first_name} declines — Switchboard moves down the chain.{RESET}")
    gateway.deliver(board, "email", "not me, I don't touch the cluster",
                    first_name, conversation_id=first_page["conversation_id"])

    second_page = store.pages_for_question(open_q["id"])[-1]
    second_name = store.expert(second_page["expert_id"]).name
    gateway.deliver(
        board, "email",
        "Check `kubectl rollout status` first, then look for a failing readiness "
        "probe — that's what wedges it 9 times out of 10.",
        second_name, conversation_id=second_page["conversation_id"],
    )

    # ---------------------------------------------------------------
    banner(5, "Where it ends up")
    stats = store.stats()
    for label, key in [
        ("questions asked", "questions_total"),
        ("answered from memory", "deflected"),
        ("needed a human", "human_resolved"),
        ("things learned", "knowledge_items"),
        ("human interruptions avoided", "human_pages_avoided"),
    ]:
        print(f"    {label:32} {stats[key]}")
    print(f"    {'deflection rate':32} {stats['deflection_rate']:.0%}")

    print(f"\n  {DIM}Every answer above was given by a human exactly once. The next"
          f"\n  person to ask gets it instantly, on whatever channel they use.{RESET}\n")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
