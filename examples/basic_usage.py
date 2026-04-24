"""
Basic usage example for gcp-memory-bank.

Shows instance creation, memory generation, retrieval, and cleanup.
Run with:
    GOOGLE_CLOUD_PROJECT=your-project python basic_usage.py
"""

import asyncio
import os

from memory_bank import MemoryBankClient, MemoryBankConfig
from memory_bank.config import CustomTopicConfig, MemoryTopic
from memory_bank.models import ManagedTopicEnum, MemoryScope
from memory_bank.utils import build_scope


async def main() -> None:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")

    async with MemoryBankClient(project=project, location="us-central1") as client:
        # 1. Create instance with custom topics
        config = MemoryBankConfig(
            memory_topics=[
                MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PREFERENCES),
                MemoryTopic(
                        custom_memory_topic=CustomTopicConfig(
                            label="coding_style",
                        description="User's coding conventions and style preferences.",
                    )
                ),
            ]
        )
        engine_name = await client.create_instance(config, display_name="demo-instance")
        print(f"Created instance: {engine_name}")

        # 2. Generate memories from raw events
        from memory_bank.memory import MemoryManager

        manager = MemoryManager(client)
        scope = build_scope(user_id="demo-user")

        events = [
            {"role": "user", "content": "I prefer Python for most projects."},
            {"role": "model", "content": "Noted. I'll use Python for your tasks."},
            {"role": "user", "content": "Actually, I switched to Rust recently for systems work."},
        ]

        generated = await manager.generate_from_events(events, scope=scope)
        print(f"Generated {len(generated)} memories:")
        for mem in generated:
            print(f"  [{mem.action.value}] {mem.memory.fact}")

        # 3. Search memories
        results = await manager.search("programming language preference", scope=scope)
        print(f"\nSearch results:")
        for r in results:
            print(f"  - {r.memory.fact} (dist: {r.distance:.3f})")

        # 4. Retrieve all for scope
        all_memories = await manager.retrieve_by_scope(scope)
        print(f"\nAll memories for scope: {len(all_memories)}")

        # 5. Cleanup
        await client.delete_instance(force=True)
        print("\nInstance deleted.")


if __name__ == "__main__":
    asyncio.run(main())
