"""Smoke test: prove the Caspian gateway is reachable and a channel provisions."""

from caspian_sdk import CommClient

client = CommClient()

print("=== channels available ===")
for ch in client.channels():
    print(f"  {ch.get('channel'):12} caps={ch.get('capabilities')}")

print("\n=== connecting email ===")
inbox = client.connect_email(username="switchboard")
print(f"  status : {inbox.get('status')}")
print(f"  address: {inbox.get('address')}")
print(f"  id     : {inbox.get('id')}")

print("\n=== behavior prompt (first 400 chars) ===")
print(client.behavior_prompt()[:400])
