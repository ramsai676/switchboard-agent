"""Start Switchboard.

    python run.py                      # email only
    python run.py --telegram <token>   # email + telegram  (satisfies the 2-channel rule)
    python run.py --discord            # email + discord   (prints an authorize URL)

Every channel used here is on Caspian's free tier — no card, no credit, no signup.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from switchboard import Store, Switchboard
from switchboard.brain import Brain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Switchboard agent.")
    parser.add_argument("--telegram", metavar="BOT_TOKEN",
                        help="Telegram bot token from @BotFather (free)")
    parser.add_argument("--discord", action="store_true",
                        help="install the shared Discord bot (free, prints authorize URL)")
    parser.add_argument("--username", default="switchboard",
                        help="mailbox name on agents.trycaspianai.com")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="seconds before a page escalates to the next expert")
    args = parser.parse_args()

    board = Switchboard(store=Store(), brain=Brain(), page_timeout=args.timeout)

    inbox = board.connect_email(username=args.username)
    print(f"\n  email    →  {inbox.get('address')}")

    token = args.telegram or os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        board.connect_telegram(token)
        print("  telegram →  connected")

    if args.discord:
        conn = board.install_discord()
        print(f"  discord  →  authorize: {conn.get('authorize_url')}")

    if len(board.connections) < 2:
        print(
            "\n  ! Only one channel is connected. The buildathon requires two.\n"
            "    Add one (both free):\n"
            "      --telegram <token>   token from @BotFather, ~60 seconds\n"
            "      --discord            one click, no bot to create\n"
        )

    experts = board.store.experts()
    print(f"\n  {len(experts)} expert(s) registered:")
    for expert in experts:
        print(f"    {expert.name:12} {expert.channel:9} {', '.join(expert.skills)}")
    if not experts:
        print("    none — run  python demo/seed.py  first")

    print("\n  Ctrl-C to stop.\n")
    try:
        board.run()
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
