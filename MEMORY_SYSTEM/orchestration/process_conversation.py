# MEMORY_SYSTEM/orchestration/process_conversation.py

async def process_conversation(
    conn,
    user_id: str,
    user_message: str,
    agent_response: str
):
    facts = await extract_facts(user_message, agent_response)

    for fact in facts:
        await store_fact(conn, user_id, fact)

    await update_persona(conn, user_id, facts)
