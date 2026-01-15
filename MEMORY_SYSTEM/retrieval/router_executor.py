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

    # -----------------------------
    # CURRENT CONTEXT (STM ONLY)
    # -----------------------------
    if route == "current_context":
        messages = await stm_store.get_recent_messages(
            session_id=session_id,
            limit=7
        )

        events = await stm_store.get_recent_events(
            session_id=session_id,
            limit=10
        )

        goals = await stm_store.get_goals(
            session_id=session_id
        )

        # Derived context — NOT stored
        stm_state = {
            "messages": messages,
            "events": events,
            "goals": goals,
        }

        return {
            "mode": "generation",
            "state": stm_state,
        }

    # -----------------------------
    # EDIT MODE (ARTIFACT REQUIRED)
    # -----------------------------
    if route == "edit":
        return {
            "mode": "edit",
            "artifact_required": True,
        }

    # -----------------------------
    # REFERENCE / LOOKUP
    # -----------------------------
    if route in ("reference", "semantic_lookup"):
        artifacts = await artifact_store.list_artifacts(limit=10)

        return {
            "mode": "reference",
            "summaries": [
                {
                    "artifact_id": a["artifact_id"],
                    "summary": a["summary"],
                    "metadata": a["metadata"],
                }
                for a in artifacts
            ],
        }

    # -----------------------------
    # UNKNOWN ROUTE
    # -----------------------------
    raise ValueError(f"Unknown route: {route}")
