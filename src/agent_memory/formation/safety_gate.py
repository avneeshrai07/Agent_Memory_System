"""Deterministic safety gate (README Section 5, step 4).

Identity- and constraint-class candidates must additionally pass an
explicit-statement-or-N-repetitions check regardless of what the resolution
step (ADD/UPDATE/DELETE/NOOP) decided — this is what prevents one weakly
inferred extraction from silently becoming the user's identity, or from
superseding an existing identity/constraint fact.

Not an LLM call: this is exactly the kind of judgment that must stay
deterministic, since it's the last line of defense against a bad LLM call
elsewhere in the pipeline.
"""

from __future__ import annotations

from agent_memory.models import ExtractedCandidate

SAFETY_GATED_CATEGORIES = {"identity", "constraint"}
SAFETY_GATE_MIN_OBSERVATIONS = 2


def passes_safety_gate(candidate: ExtractedCandidate, existing_observation_count: int) -> bool:
    """existing_observation_count is 0 for a brand-new candidate (ADD), or
    the observation_count of the existing fact it would UPDATE/DELETE.

    Categories outside SAFETY_GATED_CATEGORIES always pass — this gate is
    specifically for information that would be costly to get wrong.
    """

    if candidate.category not in SAFETY_GATED_CATEGORIES:
        return True

    if candidate.explicit:
        return True

    return existing_observation_count + 1 >= SAFETY_GATE_MIN_OBSERVATIONS
