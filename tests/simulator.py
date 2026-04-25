"""Hermes Conversation Simulator for GCP Memory Bank.

Simulates the full Hermes conversation lifecycle with the Memory Bank plugin
active, showing:
  - Session-start memory retrieval
  - Per-turn prefetch and sync
  - Tool call integration
  - Session-end structured fact extraction
  - Cross-session persistence verification

Run: python simulator.py
"""

from __future__ import annotations

import json
import sys
import time
import threading
from typing import Any, Dict, List, Optional

# Ensure hermes-agent is importable
sys.path.insert(0, "/Users/jithendranara/.hermes/hermes-agent")

from plugins.memory import load_memory_provider


class HermesSimulator:
    """Simulates Hermes agent behavior with a memory provider."""

    def __init__(self, provider_name: str = "gcp-memory-bank", user_id: str = "sim-user"):
        self.provider = load_memory_provider(provider_name)
        self.provider.initialize(
            session_id=f"sim-{int(time.time())}",
            user_id=user_id,
            agent_identity="hermes",
            platform="telegram",
        )
        self.messages: List[Dict[str, str]] = []
        self.turn_count = 0
        self.memory_hits = 0
        self.memory_misses = 0

    def _inject_memories(self, user_query: str) -> str:
        """Simulate what Hermes does: queue_prefetch before LLM call, then prefetch()."""
        self.provider.queue_prefetch(user_query)
        time.sleep(1.5)  # Simulate LLM thinking time
        context = self.provider.prefetch(user_query)
        return context

    def _build_system_prompt(self) -> str:
        """Build system prompt with Memory Bank context block."""
        base = (
            "You are Hermes, a helpful AI assistant on a Mac Mini home server. "
            "You have access to persistent memory via GCP Memory Bank."
        )
        mem_block = self.provider.system_prompt_block()
        return f"{base}\n\n{mem_block}"

    def _simulate_llm_response(self, user_query: str, memory_context: str) -> str:
        """Simulate an LLM response given user query + injected memory context."""
        # In a real system this calls the LLM. Here we simulate how memory
        # would influence the response.
        q_lower = user_query.lower()

        # If memory context is present, simulate personalized response
        if memory_context and "User lives in" in memory_context:
            if "weather" in q_lower:
                self.memory_hits += 1
                return "I remember you live in Fort Wayne, Indiana. It's usually moderate there — mid-50s F this time of year."

        if memory_context and "golden retriever" in memory_context:
            if "dog" in q_lower or "pet" in q_lower:
                self.memory_hits += 1
                return "Max! Your golden retriever. How is he doing?"

        if memory_context and "python" in memory_context:
            if "learning" in q_lower or "study" in q_lower:
                self.memory_hits += 1
                return "You're learning Python and AI development — want me to suggest some GCP-focused projects?"

        self.memory_misses += 1
        return f"(Simulated generic response to: '{user_query}')"

    def turn(self, user_message: str) -> str:
        """Process one conversation turn."""
        self.turn_count += 1
        print(f"\n{'='*60}")
        print(f"TURN {self.turn_count}")
        print(f"{'='*60}")
        print(f"User: {user_message}")

        # Step 1: Memory prefetch (async, simulates Hermes behavior)
        print("\n[Memory Bank] Queueing prefetch...")
        mem_context = self._inject_memories(user_message)

        if mem_context:
            print("[Memory Bank] Retrieved context:")
            for line in mem_context.split("\n"):
                if line.strip():
                    print(f"  {line}")
        else:
            print("[Memory Bank] No relevant memories found.")

        # Step 2: Build system prompt (for logging)
        sys_prompt = self._build_system_prompt()
        print(f"\n[System prompt block] {sys_prompt.split(chr(10))[-1][:80]}")

        # Step 3: Simulate LLM response
        assistant_reply = self._simulate_llm_response(user_message, mem_context)
        print(f"\nAssistant: {assistant_reply}")

        # Step 4: Sync turn to memory (non-blocking)
        print("\n[Memory Bank] Syncing turn...")
        self.provider.sync_turn(user_message, assistant_reply)

        # Store for session-end extraction
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def tool_call(self, tool_name: str, args: dict) -> str:
        """Simulate a tool call through the memory provider."""
        print(f"\n[Tool call] {tool_name}({json.dumps(args)})")
        result = self.provider.handle_tool_call(tool_name, args)
        data = json.loads(result)
        print(f"[Tool result] {json.dumps(data, indent=2)[:300]}")
        return result

    def end_session(self) -> None:
        """Trigger session-end memory extraction."""
        print(f"\n{'='*60}")
        print("SESSION END — Extracting structured facts")
        print(f"{'='*60}")
        self.provider.on_session_end(self.messages)
        time.sleep(3)  # Let async extraction complete

        # Verify what was extracted
        print("\n[Memory Bank] Verifying extracted facts:")
        result = self.provider.handle_tool_call("memory_profile", {})
        data = json.loads(result)
        print(f"  Total memories: {data.get('count', 0)}")
        for line in data.get("result", "").split("\n"):
            if line.strip():
                print(f"    ✓ {line}")

    def stats(self) -> None:
        """Print simulation stats."""
        print(f"\n{'='*60}")
        print("SIMULATION STATS")
        print(f"{'='*60}")
        print(f"Total turns:      {self.turn_count}")
        print(f"Memory hits:      {self.memory_hits} (personalized responses)")
        print(f"Memory misses:    {self.memory_misses} (generic responses)")
        print(f"Hit rate:         {self.memory_hits / max(self.turn_count, 1) * 100:.0f}%")
        print(f"Engine:           {self.provider._engine_id}")


def run_simulation():
    """Run a full multi-session simulation."""
    print("\n" + "=" * 60)
    print("HERMES + GCP MEMORY BANK CONVERSATION SIMULATOR")
    print("=" * 60)

    # ---- Session 1: Getting to know the user ----
    sim = HermesSimulator(user_id="jith-sim")

    print("\n" + "-" * 60)
    print("SESSION 1: Getting to know the user")
    print("-" * 60)

    sim.turn("Hey, I'm Jithendra. I live in Fort Wayne, Indiana.")
    sim.turn("I have a golden retriever named Max.")
    sim.turn("I'm learning Python and AI development.")
    sim.turn("What should I work on today?")  # Generic, no memory hit yet

    # Simulate mid-session memory search
    sim.tool_call("memory_search", {"query": "pets", "top_k": 3})

    sim.end_session()
    sim.stats()

    # ---- Session 2: Cross-session recall ----
    print("\n\n" + "=" * 60)
    print("SESSION 2: Cross-session recall (new session, same user)")
    print("=" * 60)

    # Create a NEW simulator instance — simulates Hermes restart
    sim2 = HermesSimulator(user_id="jith-sim")
    time.sleep(2)

    # These should trigger memory hits because facts were stored in Session 1
    sim2.turn("What's the weather like where I live?")
    sim2.turn("How's my dog doing?")
    sim2.turn("Any AI project ideas for me?")
    sim2.turn("I just got a new laptop.")  # New fact, should extract

    sim2.end_session()
    sim2.stats()

    # ---- Session 3: Deduplication stress test ----
    print("\n\n" + "=" * 60)
    print("SESSION 3: Deduplication stress test")
    print("=" * 60)

    sim3 = HermesSimulator(user_id="jith-sim")

    # Repeat same facts — should deduplicate
    sim3.turn("I live in Fort Wayne, Indiana.")
    sim3.turn("I live in Fort Wayne, Indiana.")
    sim3.turn("My name is Jithendra.")

    sim3.end_session()

    # Verify no duplicates
    print("\n[Memory Bank] Checking for duplicates:")
    result = sim3.provider.handle_tool_call("memory_consolidate", {})
    print(f"  {json.loads(result)['result']}")

    # Final profile
    print("\n[Memory Bank] Final profile:")
    result = sim3.provider.handle_tool_call("memory_profile", {})
    data = json.loads(result)
    for line in data.get("result", "").split("\n"):
        if line.strip():
            print(f"    ✓ {line}")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
