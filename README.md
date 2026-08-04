# Switchboard

**An agent that pages humans, then replaces them.**

Built for the [Caspian Buildathon](https://caspian.devpost.com/) — *build agents that can reach anyone*.

---

## The idea

Most "ask the AI" agents answer from a knowledge base somebody else wrote. When
they don't know something, they apologise.

Switchboard doesn't have a knowledge base. It **builds one**, by going and finding
the human who knows — on whatever channel that person actually lives on — and
never asking them the same thing twice.

```
Sam asks on Telegram    →  Switchboard doesn't know
                        →  works out this is a "refunds" question
                        →  cold-starts an email to Priya, who owns refunds
Priya replies by email  →  answer relayed back to Sam on Telegram
                        →  and learned, permanently

Dana asks the same thing on email, a week later
                        →  answered instantly. Priya is never interrupted again.
```

The interesting metric isn't accuracy. It's **deflection rate** — the share of
questions answered without waking a human — and it climbs on its own as the agent
runs. The agent is working itself out of a job.

---

## Why this needs Caspian specifically

The two-channel rule isn't decoration here, it's the whole architecture. The
asker and the expert are *structurally* on different channels:

- The person asking uses whatever they already have open — Telegram, Slack, email.
- The expert must be **reached where they are**, without having messaged the bot
  first. That's a cold start, and it's the capability everything rests on.

A per-platform bot can't do this. It can only talk to people who come to it. The
whole point of Switchboard is that it goes to *them*.

Caspian capabilities used:

| Capability | Used for |
| --- | --- |
| `initiate` (email, discord) | Cold-starting a conversation with an expert who never messaged us |
| `interactions` (telegram, discord, slack) | "I'll answer" / "Not my area" buttons on a page |
| `reactions` | ✅ on an expert's message once their answer is relayed |
| `blocks` | Rich page layout that degrades to clean text on email |
| `channel_guide()` | Feeds Caspian's own per-channel etiquette into the drafting prompt, so a page reads native in an inbox and native in Slack |
| one `on_message` handler | Serves askers and experts both — see below |

### The one-handler trick

There's a single `on_message`. Whether you're an asker or an expert is decided by
whether the conversation you're speaking in has an open page attached to it:

```python
page = store.open_page_for_expert_conversation(message.conversation_id)
if page:  handle_expert_reply(...)   # you're answering something we asked you
else:     handle_question(...)       # you're asking us something
```

---

## Running it

```bash
pip install -r requirements.txt

# Mint a free sandbox key — no signup, no card, no account.
curl -s -X POST https://api.trycaspianai.com/v1/projects/sandbox \
     -H 'Content-Type: application/json' -d '{"name":"switchboard"}'
# → put the api_key in .env as CASPIAN_API_KEY

python demo/seed.py                       # register your experts
python run.py --telegram <BOTFATHER_TOKEN> # email + telegram
```

In another terminal:

```bash
python -m switchboard.dashboard           # http://127.0.0.1:8765
```

### See it work without any of that

```bash
python demo/simulate.py
```

Runs the entire lifecycle — routing, paging, decline, escalation, relay, learning,
deflection — against an in-memory gateway. No keys, no channels, no network.
Nothing inside the agent is stubbed; only the transport is swapped.

---

## Cost

**Zero.** Sandbox keys are free and unauthenticated. Email, Telegram, Discord and
Slack are free Caspian channels. Inference is optional — set `FEATHERLESS_API_KEY`
to use a model, or leave it unset and the agent runs on a deterministic local
backend. Routing still works either way; only the phrasing is templated.

That fallback isn't a demo shortcut. An agent whose job is paging humans should
not go silent because an inference bill lapsed.

---

## How it decides things

**Who to page** — `router.py` scores every expert on three terms:

```
0.60 × skill affinity  +  0.25 × reliability  +  0.15 × load relief
```

That last term matters more than it looks. A router that only optimises for
"most likely to answer" converges on paging the same helpful person forever,
which is exactly how internal help channels die. Anyone with zero plausible
connection to the topic is never paged at all — paging the wrong human teaches
them to ignore the agent.

**Whether it already knows** — `knowledge.py` does TF-IDF cosine over learned
questions, with a confidence floor of 0.45. Below that it pages a human instead
of guessing. Answering confidently from a weak match is the worst outcome
available to this agent: the asker gets a wrong answer *and* no human is ever
paged to correct it.

**When to give up on someone** — an unanswered page escalates to the next name in
the chain after 90s, skipping everyone already tried. State lives in SQLite, so
an agent restarted mid-page still knows who it was waiting on.

---

## Layout

```
switchboard/
  agent.py       Caspian wiring — the one handler, paging, escalation, buttons
  router.py      who to interrupt, and in what order
  knowledge.py   retrieval over what humans have already taught it
  brain.py       inference (Featherless / OpenAI-compatible / local fallback)
  store.py       SQLite state — survives a restart mid-page
  dashboard.py   live ops view (stdlib http.server + SSE, no web framework)
demo/
  seed.py        register a team
  simulate.py    whole lifecycle against a fake gateway
tests/           routing, retrieval, escalation, metrics
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers the parts that would silently rot: retrieval confidence (including the
negative case — a populated knowledge base must *not* answer off-topic questions),
routing order, load balancing, escalation exclusion, and restart recovery.

---

## Honest limitations

- **Retrieval is lexical, not semantic.** TF-IDF with light stemming. A question
  phrased with entirely different vocabulary than the learned one will miss and
  page a human again. That fails safe, but it does mean deflection climbs slower
  than an embedding-based store would.
- **Expert registry is manual.** Skills are hand-tagged in `demo/seed.py`. Mining
  them from commit history or ticket ownership is the obvious next step.
- **No answer expiry.** A learned answer is served forever. Real deployments need
  a staleness signal — "is this still true?" — or the agent confidently repeats
  last quarter's policy.
- **`initiate` is channel-dependent.** Email and Discord support cold start;
  Telegram does not, so a Telegram expert must have messaged the bot once before
  they can be paged there.

## License

MIT — see [LICENSE](LICENSE).
