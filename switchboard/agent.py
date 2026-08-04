"""The Switchboard agent.

One `on_message` handler serves two completely different populations:

  * someone asking a question (any channel)
  * an expert answering a page we sent them (their channel)

Which one you are is decided by whether the conversation you're speaking in has
an open page attached to it. That is the whole trick, and it is why this needs
Caspian rather than a per-platform bot: the asker and the expert are structurally
on different channels, and the agent has to cold-start the conversation with the
expert rather than wait to be spoken to.
"""

from __future__ import annotations

import logging
import threading
import time

from caspian_sdk import CommClient, CommError, Message, blocks

from .brain import Brain
from .knowledge import KnowledgeBase
from .router import escalation_chain
from .store import Store

logger = logging.getLogger("switchboard.agent")

# How long an expert has before we escalate to the next name in the chain.
PAGE_TIMEOUT_SECONDS = float(90)
ESCALATION_TICK_SECONDS = 10.0

DECLINE_PREFIXES = ("not me", "no idea", "dunno", "don't know", "dont know", "pass", "nope")


class Switchboard:
    def __init__(
        self,
        client: CommClient | None = None,
        store: Store | None = None,
        brain: Brain | None = None,
        page_timeout: float = PAGE_TIMEOUT_SECONDS,
    ) -> None:
        self.client = client or CommClient()
        self.store = store or Store()
        self.brain = brain or Brain()
        self.kb = KnowledgeBase(self.store)
        self.page_timeout = page_timeout
        self.connections: dict[str, str] = {}   # channel -> connection_id
        self._guides: dict[str, str] = {}
        self._stop = threading.Event()

        self.client.on_message(self.handle_message)
        self.client.on_interaction(self.handle_interaction)

    # ---- setup ----------------------------------------------------------

    def connect_email(self, username: str = "switchboard") -> dict:
        inbox = self.client.connect_email(username=username)
        self.connections["email"] = inbox["id"]
        logger.info("email channel live at %s", inbox.get("address"))
        self.store.log("channel.connected", channel="email", address=inbox.get("address"))
        return inbox

    def connect_telegram(self, bot_token: str) -> dict:
        conn = self.client.connect_telegram(bot_token=bot_token)
        self.connections["telegram"] = conn["id"]
        logger.info("telegram channel live (%s)", conn["id"])
        self.store.log("channel.connected", channel="telegram")
        return conn

    def install_discord(self, display_name: str = "Switchboard") -> dict:
        conn = self.client.install_discord(display_name=display_name)
        self.connections["discord"] = conn["id"]
        logger.info("discord: authorize at %s", conn.get("authorize_url"))
        self.store.log("channel.connected", channel="discord",
                       authorize_url=conn.get("authorize_url"))
        return conn

    def guide(self, channel: str) -> str:
        """Caspian's own per-channel etiquette guide, fed into the drafting prompt
        so a page reads native on Slack and native in an inbox."""
        if channel not in self._guides:
            try:
                self._guides[channel] = self.client.channel_guide(channel)
            except CommError:
                self._guides[channel] = ""
        return self._guides[channel]

    # ---- inbound --------------------------------------------------------

    def handle_message(self, message: Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return

        page = self.store.open_page_for_expert_conversation(message.conversation_id)
        if page:
            self.handle_expert_reply(message, page, text)
        else:
            self.handle_question(message, text)

    # ---- path 1: somebody asked something -------------------------------

    def handle_question(self, message: Message, text: str) -> None:
        sender = message.sender or {}
        asker = sender.get("name") or sender.get("address") or sender.get("id") or "someone"

        # Already known? Answer instantly and page nobody.
        hit = self.kb.best(text)
        if hit:
            answer = self.brain.answer_from_knowledge(text, hit.question_text, hit.answer_text)
            self.store.bump_reuse(hit.knowledge_id)
            qid = self.store.add_question(
                text, message.channel, message.id, message.conversation_id, asker, None
            )
            self.store.resolve_question(qid, answer, "memory", "answered_from_memory")
            credit = f" (learned from {hit.taught_by})" if hit.taught_by else ""
            message.reply(f"{answer}\n\n— from what I already know{credit}")
            logger.info("deflected %s from memory (score %.2f)", qid, hit.score)
            return

        # Unknown. Work out the topic, then who owns it.
        skills = sorted({s for e in self.store.experts() for s in e.skills})
        skill = self.brain.classify(text, skills)
        qid = self.store.add_question(
            text, message.channel, message.id, message.conversation_id, asker, skill
        )

        chain = escalation_chain(self.store, text, skill)
        if not chain:
            message.reply(
                "I don't know this yet and I don't have anyone registered who covers it. "
                "Add an expert for this topic and ask me again — I'll learn it permanently."
            )
            self.store.resolve_question(qid, "", "none", "abandoned")
            return

        first = chain[0]
        message.reply(
            f"I don't know that one yet. Paging {first.expert.name} "
            f"on {first.expert.channel} — {first.reason}. I'll relay the answer here."
        )
        self.page_expert(qid, first.expert, text)

    # ---- paging ---------------------------------------------------------

    def page_expert(self, question_id: str, expert, question_text: str) -> None:
        """Cold-start a conversation with a human on their own channel."""
        connection_id = self.connections.get(expert.channel)
        if not connection_id:
            logger.error("no %s connection; cannot page %s", expert.channel, expert.name)
            return

        body = self.brain.compose_page(question_text, expert.name, self.guide(expert.channel))

        try:
            started = self.client.initiate(connection_id, expert.address, body)
        except CommError:
            logger.exception("could not page %s on %s", expert.name, expert.channel)
            return

        conversation_id = (
            started.get("conversation_id")
            or (started.get("conversation") or {}).get("id")
            or started.get("id")
        )
        page_id = self.store.add_page(question_id, expert.id, expert.channel, conversation_id)

        # Follow up with buttons where the channel supports them; on email this
        # degrades to clean text automatically, so it is safe to always send.
        if conversation_id:
            try:
                self.client.send_message(
                    conversation_id,
                    text="Can you take this one?",
                    blocks=[
                        blocks.buttons([
                            {"text": "I'll answer", "value": f"accept:{page_id}"},
                            {"text": "Not my area", "value": f"decline:{page_id}"},
                        ])
                    ],
                )
            except CommError:
                logger.debug("buttons unsupported on %s; text page stands", expert.channel)

        logger.info("paged %s (%s) for %s", expert.name, expert.channel, question_id)

    # ---- path 2: an expert replied to a page ----------------------------

    def handle_expert_reply(self, message: Message, page: dict, text: str) -> None:
        question = self.store.question(page["question_id"])
        if not question or question["status"] != "paging":
            message.reply("Thanks — this one's already been resolved.")
            return

        if text.lower().startswith(DECLINE_PREFIXES):
            self.store.set_page_status(page["id"], "declined")
            message.reply("No problem — passing it on.")
            self.escalate(question, exclude_page=page)
            return

        expert = self.store.expert(page["expert_id"])
        expert_name = expert.name if expert else "an expert"

        self.store.set_page_status(page["id"], "answered")
        self.store.bump_expert(page["expert_id"], answered=True)

        relay = self.brain.compose_relay(question["text"], text, expert_name)
        self.store.resolve_question(question["id"], relay, page["expert_id"], "resolved")

        # The compounding step — this is why the agent needs humans less over time.
        self.store.learn(question["text"], text, question["skill"], expert_name)

        self.relay_to_asker(question, relay)
        message.reply(
            "Relayed, thank you. I've learned this one — nobody will be paged for it again."
        )
        message.react("✅")

    def relay_to_asker(self, question: dict, answer: str) -> None:
        try:
            if question.get("asker_message_id"):
                self.client.reply(question["asker_message_id"], text=answer)
            elif question.get("asker_conversation_id"):
                self.client.send_message(question["asker_conversation_id"], text=answer)
        except CommError:
            logger.exception("could not relay answer for %s", question["id"])

    # ---- buttons --------------------------------------------------------

    def handle_interaction(self, interaction) -> None:
        value = interaction.value or ""
        action, _, page_id = value.partition(":")
        page = self.store.page(page_id)
        if not page:
            return

        if action == "accept":
            self.store.set_page_status(page_id, "accepted")
            interaction.reply("Great — send the answer whenever you're ready.")
        elif action == "decline":
            self.store.set_page_status(page_id, "declined")
            interaction.reply("Understood, passing it to someone else.")
            question = self.store.question(page["question_id"])
            if question and question["status"] == "paging":
                self.escalate(question, exclude_page=page)

    # ---- escalation -----------------------------------------------------

    def escalate(self, question: dict, exclude_page: dict | None = None) -> None:
        """Move to the next name in the chain, skipping everyone already tried."""
        tried = {p["expert_id"] for p in self.store.pages_for_question(question["id"])}
        if exclude_page:
            tried.add(exclude_page["expert_id"])

        chain = escalation_chain(self.store, question["text"], question["skill"], exclude=tried)
        if not chain:
            self.store.resolve_question(question["id"], "", "none", "abandoned")
            self.relay_to_asker(
                question,
                "I paged everyone who might know this and didn't get an answer. "
                "Worth asking around directly.",
            )
            return

        nxt = chain[0]
        self.store.log("question.escalated", question_id=question["id"],
                       to=nxt.expert.name, channel=nxt.expert.channel)
        self.page_expert(question["id"], nxt.expert, question["text"])

    def _escalation_loop(self) -> None:
        while not self._stop.wait(ESCALATION_TICK_SECONDS):
            try:
                for page in self.store.stale_pages(self.page_timeout):
                    question = self.store.question(page["question_id"])
                    if not question or question["status"] != "paging":
                        continue
                    self.store.set_page_status(page["id"], "timed_out")
                    logger.info("page %s timed out; escalating", page["id"])
                    self.escalate(question, exclude_page=page)
            except Exception:
                logger.exception("escalation sweep failed; continuing")

    # ---- run ------------------------------------------------------------

    def run(self) -> None:
        watcher = threading.Thread(target=self._escalation_loop, daemon=True,
                                   name="switchboard-escalation")
        watcher.start()
        logger.info("Switchboard listening on %s", ", ".join(self.connections) or "no channels")
        try:
            self.client.listen(ack=None, concurrency="queue")
        finally:
            self._stop.set()
