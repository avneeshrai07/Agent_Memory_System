async def execute_route(
    *,
    route: str,
    user_id: str,
    session_id: str,
    stm_store,
    artifact_store,
):
    """
    Executes retrieval based on route.
    Does NOT generate.
    Does NOT write memory.
    """

    if route == "current_context":
        # Load STM only
        stm_records = await stm_store.fetch_active_records(
            user_id=user_id,
            limit=7
        )

        return {
            "mode": "generation",
            "state": stm_records
        }

    if route == "edit":
        # For now, just signal edit mode
        return {
            "mode": "edit",
            "artifact_required": True
        }

    if route in ("reference", "semantic_lookup"):
        artifacts = await artifact_store.list_artifacts(limit=10)

        return {
            "mode": "reference",
            "summaries": [
                {
                    "artifact_id": a["artifact_id"],
                    "summary": a["summary"],
                    "metadata": a["metadata"]
                }
                for a in artifacts
            ]
        }

    raise ValueError(f"Unknown route: {route}")
