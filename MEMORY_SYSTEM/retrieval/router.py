# MEMORY_SYSTEM/retrieval/router.py

from MEMORY_SYSTEM.retrieval.retrieval_context import RetrievalContext


async def build_retrieval_context(
    *,
    route_intent,
    session_id: str,
    user_id: str,
    message_store,
    stm_store,
    artifact_store,
    artifact_id: str | None = None,
    campaign_id: str | None = None,
) -> RetrievalContext:
    try:
        route = route_intent.route

        # 1️⃣ Generation: STM only
        if route == "current_context":
            dcc = {
                "recent_messages": [],  # messages intentionally omitted
                "active_state": [
                    {
                        "type": r.state_type,
                        "content": r.statement,
                        "timestamp": r.created_at,
                    }
                    for r in await stm_store.fetch_active_records(
                        user_id=user_id,
                        limit=7,
                    )
                ],
            }

            return {
                "dcc": dcc,
                "artifact": None,
                "artifact_summaries": None,
            }

        # 2️⃣ Edit: Single artifact
        if route == "edit":
            if not artifact_id:
                raise ValueError("artifact_id required for edit route")

            artifact = await artifact_store.get_artifact(
                artifact_id=artifact_id
            )

            return {
                "dcc": None,
                "artifact": artifact,
                "artifact_summaries": None,
            }

        # 3️⃣ Reference: Summaries only
        if route == "reference":
            artifacts = await artifact_store.list_artifacts(
                limit=20
            )

            summaries = [
                {
                    "artifact_id": a["artifact_id"],
                    "artifact_type": a["artifact_type"],
                    "summary": a["summary"],
                    "metadata": a["metadata"],
                }
                for a in artifacts
            ]

            return {
                "dcc": None,
                "artifact": None,
                "artifact_summaries": summaries,
            }

        # 4️⃣ Semantic lookup (future-safe)
        if route == "semantic_lookup":
            artifacts = await artifact_store.list_artifacts(limit=20)

            summaries = [
                {
                    "artifact_id": a["artifact_id"],
                    "summary": a["summary"],
                    "summary_embedding": a["summary_embedding"],
                }
                for a in artifacts
            ]

            return {
                "dcc": None,
                "artifact": None,
                "artifact_summaries": summaries,
            }

        raise ValueError(f"Unknown route: {route}")

    except Exception as e:
        raise RuntimeError("Failed to build retrieval context") from e
