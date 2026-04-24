"""
OpenClaw Bridge Demo (NOT wired into OpenClaw).

Shows how the OpenClawBridgeExample could store and retrieve
gateway-level user preferences.
"""

import asyncio

from memory_bank.bridge import OpenClawBridgeExample


async def main() -> None:
    bridge = OpenClawBridgeExample()

    # User sets preferences
    await bridge.on_agent_fact(
        "jithendra",
        "Primary model: google/gemini-3.1-pro-preview. Fallback: minimax/custom-minimax.",
    )
    await bridge.on_agent_fact(
        "jithendra",
        "Prefer direct, no-fluff responses. Max tokens: 4096.",
    )

    # Session start: OpenClaw loads preferences
    context = await bridge.on_session_start("jithendra")
    print("=== Gateway Context ===")
    print(context)
    print()

    # Recall specific preference
    print("=== Recall: model preference ===")
    results = await bridge.recall("jithendra", "preferred model")
    for r in results:
        print(f"  - {r['fact']}")


if __name__ == "__main__":
    asyncio.run(main())
