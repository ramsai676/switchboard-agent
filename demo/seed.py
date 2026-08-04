"""Seed a demo team into Switchboard.

Edit the addresses to real inboxes you control before recording a demo — the
agent genuinely cold-emails these people, so placeholder addresses will bounce.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from switchboard.store import Store  # noqa: E402

TEAM = [
    # name,     channel,    address,                     skills
    ("Priya",   "email",    "priya@example.com",         ["billing", "refunds", "invoices"]),
    ("Marco",   "email",    "marco@example.com",         ["kubernetes", "deploys", "infra"]),
    ("Ada",     "telegram", "@ada_dev",                  ["kubernetes", "networking"]),
    ("Rin",     "email",    "rin@example.com",           ["onboarding", "hr", "policy"]),
]


def main() -> None:
    store = Store()
    existing = {e.name for e in store.experts()}

    for name, channel, address, skills in TEAM:
        if name in existing:
            print(f"  = {name} already registered")
            continue
        store.add_expert(name, channel, address, skills)
        print(f"  + {name:8} {channel:9} {', '.join(skills)}")

    print(f"\n  {len(store.experts())} experts registered.")
    print("  Skills covered:",
          ", ".join(sorted({s for e in store.experts() for s in e.skills})))
    store.close()


if __name__ == "__main__":
    main()
