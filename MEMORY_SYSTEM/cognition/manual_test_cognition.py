import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR,  "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio

from MEMORY_SYSTEM.cognition.cognition_updater import run_cognition

async def main():
    signals = [
        {
            "category": "preference",
            "field": "response_length",
            "value": "short",
            "source": "explicit",
            "base_confidence": 0.85,
            "frequency": 1,
        }
    ]

    decisions = await run_cognition(signals)

    print("Cognition Decisions:")
    for d in decisions:
        print(d)

if __name__ == "__main__":
    asyncio.run(main())
