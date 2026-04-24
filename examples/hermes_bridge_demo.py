"""
Hermes Bridge Demo (NOT wired into Hermes).

Shows how the HermesBridgeExample could be used to:
1. Retrieve memories at session start
2. Store facts during the session
3. Generate memories at session end
"""

import asyncio

from memory_bank.bridge import HermesBridgeExample


async def main() -> None:
    bridge = HermesBridgeExample()

    # 1. Session start: retrieve context
    context = await bridge.on_session_start("jithendra")
    print("=== System Prompt Augmentation ===")
    print(context)
    print()

    # 2. During session: agent stores a fact
    await bridge.on_agent_fact(
        "jithendra",
        "User prefers gemini-3.1-pro-preview for reasoning tasks.",
    )
    print("=== Fact stored ===")
    print()

    # 3. Session end: generate memories from conversation
    session_events = [
        {"role": "user", "content": "Deploy my app to Cloud Run us-central1."},
        {"role": "model", "content": "Deploying to Cloud Run in us-central1..."},
        {"role": "user", "content": "Wait, use us-east1 instead."},
        {"role": "model", "content": "Updated to us-east1. Deployment complete."},
    ]

    print("=== Session-end memory generation ===")
    await bridge.on_session_end("jithendra", session_events)

    # 4. Recall
    print("\n=== Recall: deployment preferences ===")
    results = await bridge.recall("jithendra", "deployment region preference")
    for r in results:
        print(f"  - {r['fact']}")


if __name__ == "__main__":
    asyncio.run(main())
