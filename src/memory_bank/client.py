"""
Core Memory Bank client.

Wraps the official vertexai SDK with async-first APIs, retry logic,
scoped isolation, and ergonomic Pythonic interfaces.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog
import vertexai
from google.api_core import operation as api_operation
from google.api_core.exceptions import GoogleAPICallError, NotFound, ResourceExhausted
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory_bank.config import MemoryBankConfig
from memory_bank.models import MemoryScope

logger = structlog.get_logger(__name__)


class ProvisioningError(Exception):
    """Raised when Memory Bank backend is still provisioning."""

    pass


class MemoryBankClient:
    """
    Async-first client for Google Memory Bank.

    Usage:
        async with MemoryBankClient() as client:
            engine = await client.create_instance(config)
            memories = await client.search(query="Python preference", scope=scope)
    """

    DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
    DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    DEFAULT_ENGINE_ID = os.environ.get("HERMES_MEMORY_ENGINE")

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        engine_id: Optional[str] = None,
    ):
        self.project = project or self.DEFAULT_PROJECT
        self.location = location or self.DEFAULT_LOCATION
        self._engine_id = engine_id or self.DEFAULT_ENGINE_ID
        self._client: Optional[vertexai.Client] = None
        self._engine_name: Optional[str] = None

        if not self.project:
            raise ValueError(
                "Google Cloud project must be set via argument or GOOGLE_CLOUD_PROJECT env var"
            )

    async def __aenter__(self) -> MemoryBankClient:
        await self._init()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # No persistent connection to close for vertexai SDK
        pass

    async def _init(self) -> None:
        """Initialize vertexai client (idempotent)."""
        if self._client is not None:
            return

        vertexai.init(project=self.project, location=self.location)
        self._client = vertexai.Client()

        if self._engine_id:
            self._engine_name = self._resolve_engine_name(self._engine_id)
        else:
            logger.info("memory_bank.auto_discover_instance")
            self._engine_name = await self._auto_discover()

        if self._engine_name:
            logger.info(
                "memory_bank.initialized",
                project=self.project,
                location=self.location,
                engine=self._engine_name,
            )
        else:
            logger.warning("memory_bank.no_instance_found")

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((ProvisioningError, ResourceExhausted)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=10, min=10, max=120),
        reraise=True,
    )
    async def create_instance(
        self,
        config: Optional[MemoryBankConfig] = None,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Create a new Agent Platform instance with Memory Bank enabled.

        Returns the fully-qualified engine name.
        """
        await self._init()
        cfg = (config or MemoryBankConfig()).to_api_dict()

        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(
            None,
            lambda: self._client.agent_engines.create(  # type: ignore[union-attr]
                display_name=display_name,
                **cfg,
            ),
        )

        self._engine_name = engine.api_resource.name
        logger.info(
            "memory_bank.instance_created",
            engine=self._engine_name,
            config=config.model_dump() if config else {},
        )
        return self._engine_name

    async def update_instance(self, config: MemoryBankConfig) -> str:
        """Update an existing instance's Memory Bank configuration."""
        await self._ensure_engine()
        cfg = config.to_api_dict()

        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(
            None,
            lambda: self._client.agent_engines.update(  # type: ignore[union-attr]
                name=self._engine_name,
                **cfg,
            ),
        )
        self._engine_name = engine.api_resource.name
        logger.info("memory_bank.instance_updated", engine=self._engine_name)
        return self._engine_name

    async def delete_instance(self, force: bool = True) -> None:
        """Delete the instance and all associated sessions/memories."""
        await self._ensure_engine()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.agent_engines.delete(  # type: ignore[union-attr]
                name=self._engine_name, force=force
            ),
        )
        logger.warning("memory_bank.instance_deleted", engine=self._engine_name)
        self._engine_name = None

    async def list_instances(self) -> List[Dict[str, str]]:
        """List all Agent Platform instances in the project."""
        await self._init()
        loop = asyncio.get_event_loop()
        pager = await loop.run_in_executor(
            None, lambda: self._client.agent_engines.list()  # type: ignore[union-attr]
        )
        return [
            {"name": e.name, "display_name": getattr(e, "display_name", "")}
            for e in pager
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        if not self._engine_name:
            raise RuntimeError("No Memory Bank instance is active. Call create_instance() first.")
        return self._engine_name

    @property
    def raw_client(self) -> vertexai.Client:
        if self._client is None:
            raise RuntimeError("Client not initialized")
        return self._client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_engine(self) -> None:
        if not self._engine_name:
            raise RuntimeError("No engine configured. Call create_instance() or set engine_id.")

    def _resolve_engine_name(self, engine_id: str) -> str:
        if engine_id.startswith("projects/"):
            return engine_id
        return (
            f"projects/{self.project}/locations/{self.location}"
            f"/reasoningEngines/{engine_id}"
        )

    async def _auto_discover(self) -> Optional[str]:
        """Find the most recently created instance."""
        instances = await self.list_instances()
        return instances[0]["name"] if instances else None

    async def _poll_lro(self, op: api_operation.Operation) -> Any:
        """Poll a long-running operation until complete."""
        loop = asyncio.get_event_loop()
        while not op.done():
            await asyncio.sleep(1.0)
            op = await loop.run_in_executor(
                None,
                lambda: self._client.agent_engines.operations.get(name=op.name),  # type: ignore[union-attr]
            )
        return op

    @staticmethod
    def _check_provisioning(err: GoogleAPICallError) -> None:
        msg = str(err).lower()
        if "provisioning" in msg or "not ready" in msg:
            raise ProvisioningError(str(err)) from err
