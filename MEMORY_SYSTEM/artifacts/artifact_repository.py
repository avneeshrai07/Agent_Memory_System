from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from MEMORY_SYSTEM.database.connect.connect import db_manager
from datetime import datetime, timezone
import json  

class ArtifactRepository:
    def __init__(self):
        pass



    async def create_artifact(self, *, artifact_type: str, summary: str, metadata: Optional[Dict[str, Any]], content_ref: str) -> Dict[str, Any]:
        try:
            pool = await db_manager.get_pool()
            artifact_id = uuid.uuid4()
            now = datetime.now(timezone.utc)

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agentic_memory_schema.artifacts (
                        artifact_id, artifact_type, summary, metadata, 
                        content_ref, created_at, last_updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    artifact_id, artifact_type, summary, 
                    json.dumps(metadata or {}),  # ← FIX: Convert dict to JSON string
                    content_ref, now, now
                )
            
            return {
                "artifact_id": str(artifact_id),
                "artifact_type": artifact_type,
                "summary": summary,
                "metadata": metadata or {},  # Return original dict for app use
                "content_ref": content_ref,
                "created_at": now,
                "last_updated_at": now,
            }
        except Exception as e:
            import traceback
            print(f"[ARTIFACT CREATE ERROR] {e}")
            print(f"[ARTIFACT TRACEBACK] {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create artifact: {str(e)}") from e



    async def get_artifact(
        self,
        *,
        artifact_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch artifact metadata by ID.
        """
        try:
            pool = await db_manager.get_pool()

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        artifact_id,
                        artifact_type,
                        summary,
                        metadata,
                        content_ref,
                        created_at,
                        last_updated_at
                    FROM {self.schema}.{self.table}
                    WHERE artifact_id = $1
                    """,
                    artifact_id,
                )

            if not row:
                return None

            return dict(row)

        except Exception as e:
            raise RuntimeError("Failed to fetch artifact") from e

    async def update_artifact(
        self,
        *,
        artifact_id: str,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update artifact metadata. Any provided field is updated.
        """
        try:
            pool = await db_manager.get_pool()
            now = datetime.utcnow()

            fields = []
            values = []
            idx = 1

            if summary is not None:
                fields.append(f"summary = ${idx}")
                values.append(summary)
                idx += 1

            if metadata is not None:
                fields.append(f"metadata = ${idx}")
                values.append(metadata)
                idx += 1

            if content_ref is not None:
                fields.append(f"content_ref = ${idx}")
                values.append(content_ref)
                idx += 1

            if not fields:
                raise ValueError("No fields provided to update")

            fields.append(f"last_updated_at = ${idx}")
            values.append(now)
            idx += 1

            values.append(artifact_id)

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    UPDATE {self.schema}.{self.table}
                    SET {", ".join(fields)}
                    WHERE artifact_id = ${idx}
                    RETURNING
                        artifact_id,
                        artifact_type,
                        summary,
                        metadata,
                        content_ref,
                        created_at,
                        last_updated_at
                    """,
                    *values,
                )

            if not row:
                raise KeyError(f"Artifact not found: {artifact_id}")

            return dict(row)

        except Exception as e:
            raise RuntimeError("Failed to update artifact") from e

    async def list_artifacts(
        self,
        *,
        artifact_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List artifacts (non-semantic, metadata only).
        """
        try:
            pool = await db_manager.get_pool()

            where_clause = ""
            params = []
            idx = 1

            if artifact_type:
                where_clause = f"WHERE artifact_type = ${idx}"
                params.append(artifact_type)
                idx += 1

            params.extend([limit, offset])

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        artifact_id,
                        artifact_type,
                        summary,
                        metadata,
                        content_ref,
                        created_at,
                        last_updated_at
                    FROM {self.schema}.{self.table}
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${idx} OFFSET ${idx + 1}
                    """,
                    *params,
                )

            return [dict(r) for r in rows]

        except Exception as e:
            raise RuntimeError("Failed to list artifacts") from e
