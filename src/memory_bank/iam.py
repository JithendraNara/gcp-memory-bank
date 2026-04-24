"""
IAM Conditions and scope-based security for Memory Bank.

Enforces memory isolation at the infrastructure level using
Google Cloud IAM Conditions.
"""

from __future__ import annotations

import subprocess
from typing import Optional

import structlog

from memory_bank.models import MemoryScope

logger = structlog.get_logger(__name__)


class MemoryBankIAM:
    """Helper for applying IAM Conditions to Memory Bank resources."""

    @staticmethod
    def build_condition_expression(scope: MemoryScope) -> str:
        """
        Build a CEL expression for IAM Conditions.

        Restricts access to memories matching the given scope.
        """
        parts = []
        for key, value in scope.to_dict().items():
            parts.append(
                f'resource.attributes.aiplatform.googleapis.com/memoryScope.{key}=="{value}"'
            )
        return " AND ".join(parts)

    @staticmethod
    def grant_scope_access(
        project_id: str,
        principal: str,
        scope: MemoryScope,
        role: str = "roles/aiplatform.memoryReader",
        dry_run: bool = False,
    ) -> str:
        """
        Grant a principal access ONLY to memories matching a scope.

        Uses gcloud CLI under the hood.
        """
        condition = MemoryBankIAM.build_condition_expression(scope)
        cmd = [
            "gcloud", "projects", "add-iam-policy-binding", project_id,
            f"--member={principal}",
            f"--role={role}",
            f"--condition=expression={condition},title=memory-scope-access,description=Restrict to specific memory scope",
        ]

        logger.info(
            "iam.grant_scope_access",
            project=project_id,
            principal=principal,
            role=role,
            scope=scope.to_dict(),
            dry_run=dry_run,
        )

        if dry_run:
            return " ".join(cmd)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("iam.gcloud_failed", stderr=result.stderr)
            raise RuntimeError(f"gcloud failed: {result.stderr}")
        return result.stdout

    @staticmethod
    def revoke_scope_access(
        project_id: str,
        principal: str,
        role: str = "roles/aiplatform.memoryReader",
    ) -> str:
        """Remove a principal's access."""
        cmd = [
            "gcloud", "projects", "remove-iam-policy-binding", project_id,
            f"--member={principal}",
            f"--role={role}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"gcloud failed: {result.stderr}")
        return result.stdout


# Predefined roles relevant to Memory Bank
ROLES = {
    "memory_reader": "roles/aiplatform.memoryReader",
    "memory_writer": "roles/aiplatform.memoryWriter",
    "user": "roles/aiplatform.user",
    "admin": "roles/aiplatform.admin",
}
