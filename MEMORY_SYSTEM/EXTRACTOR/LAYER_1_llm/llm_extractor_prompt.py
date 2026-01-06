from MEMORY_SYSTEM.EXTRACTOR.LAYER_1_llm.llm_extractor_schema import FactExtractionOutput
from MEMORY_SYSTEM.EXTRACTOR.LAYER_1_llm.llm_extractor import bedrock_llm_with_parser

async def extract_facts_from_conversation(user_message: str, agent_response: str):
    system_role = f"""
    You are an information extraction engine.

Your task is to analyze a conversation between a user and an AI agent and extract durable, reusable facts about the user, their context, preferences, constraints, and working patterns.

Rules:
- Extract only facts that are clearly implied or explicitly stated.
- Do NOT infer personal traits without evidence.
- Facts should be stable and reusable across future conversations.
- Ignore temporary or one-off requests unless they reveal a broader pattern.
- Each fact must fit exactly one of the allowed categories.
- Importance must be an integer from 1 to 10.
- Output must be valid JSON only.
- Do not include explanations, comments, or extra text outside JSON.

If no meaningful facts are found, return an empty JSON array: [].

"""
        
    prompt = f"""
    Extract key facts and patterns from this conversation.
    Return as JSON array with each fact containing:
    - topic: Main subject area
    - fact: The actual statement/observation
    - category: Type of fact (see categories below)
    - importance: 1-10 scale

    Categories:
    - technical_context: Technical facts ("PostgreSQL 500GB", "30s query latency")
    - user_preference: How user wants things ("I prefer JSON", "code-only")
    - problem_domain: What problem user works on ("database optimization")
    - expertise: What user knows ("expert in Python", "needs help with DevOps")
    - constraint: What user cannot do ("can't change schema", "2-week deadline")
    - learned_pattern: Pattern observed ("user ignores explanations", "always validates")

    CONVERSATION:
    User: {user_message}
    Agent: {agent_response}

    EXTRACT FACTS (JSON):
"""
    
    return await bedrock_llm_with_parser(system_role, prompt, FactExtractionOutput)
        
    