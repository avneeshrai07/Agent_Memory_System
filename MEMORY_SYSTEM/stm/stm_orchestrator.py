from MEMORY_SYSTEM.stm.stm_prompt import stm_intent_extractor_function
from MEMORY_SYSTEM.stm.stm_intent_gatekeeper import approve_stm_intent
from MEMORY_SYSTEM.stm.stm_repository import commit_stm_intent
from MEMORY_SYSTEM.retrieval.router_executor import execute_route
from MEMORY_SYSTEM.storage.message_store import MessageStore
from MEMORY_SYSTEM.storage.stm_store import STMStore
from MEMORY_SYSTEM.artifacts.artifact_store import ArtifactStore
from MEMORY_SYSTEM.artifacts.s3_client import ArtifactS3Client
from MEMORY_SYSTEM.artifacts.artifact_repository import ArtifactRepository
from datetime import datetime, timezone
import uuid
import os
from dotenv import load_dotenv
load_dotenv()
S3_bucket_name = os.getenv("S3_bucket_name")
artifact_repo = ArtifactRepository()
s3_client = ArtifactS3Client(bucket=S3_bucket_name)
message_store = MessageStore()
stm_store = STMStore()
artifact_store = ArtifactStore(
    repo=artifact_repo,
    s3_client=s3_client,
)
async def process_user_message(
    user_id: str,
    message: str
):
    """
    Single entry point for processing a user message.
    """

    try:
        print("\n[ORCHESTRATOR] New user message")
        print("[ORCHESTRATOR] Message:", message)

        # ----------------------------------
        # 1. LLM intent interpretation
        # ----------------------------------
        intent = await stm_intent_extractor_function(message)

        stm_intent = intent["stm"]
        route_intent = intent["route"]

        print("[ORCHESTRATOR] STM intent:", stm_intent)
        print("[ORCHESTRATOR] Route intent:", route_intent)

        # ----------------------------------
        # 2. Routing decision (ALWAYS)
        # ----------------------------------
        route = route_intent["route"]
        route_confidence = route_intent["confidence"]

        # ----------------------------------
        # 3. STM gatekeeper (CONDITIONAL)
        # ----------------------------------
        stm_entry = None

        if approve_stm_intent(stm_intent):
            print("[ORCHESTRATOR] STM write approved")

            stm_entry = await commit_stm_intent(
                user_id=user_id,
                intent=stm_intent
            )

            print("[ORCHESTRATOR] STM updated:", stm_entry["stm_id"])
        else:
            print("[ORCHESTRATOR] STM write rejected by gatekeeper")

        # ----------------------------------
        # 4. Return full orchestration result
        # ----------------------------------
        retrieval_result = await execute_route(
            route=route,
            user_id=user_id,
            session_id="session-1",  # or pass it properly
            stm_store=stm_store,
            artifact_store=artifact_store,
        )

        print("[ORCHESTRATOR] Retrieval mode:", retrieval_result["mode"])
        return {
            "route": route,
            "route_confidence": route_confidence,
            "stm_written": stm_entry is not None,
            "stm_entry": stm_entry,
            "raw_message": message
        }

    except Exception as e:
        print("[ORCHESTRATOR][ERROR]", str(e))
        raise


def should_create_artifact(route: str, response_text: str) -> bool:
    return (
        route == "current_context"
        and len(response_text.strip()) > 200
    )


import uuid
from datetime import datetime, timezone

async def post_model_response(user_id, route, route_confidence,  stm_written, response_text):
    print("\n[POST_MODEL] Entered post_model_response")
    print("[POST_MODEL] user_id:", user_id)
    print("[POST_MODEL] route:", route)
    print("[POST_MODEL] stm_written:", stm_written)
    print("[POST_MODEL] response length:", len(response_text))

    artifact = None

    try:
        # ----------------------------------
        # Artifact decision
        # ----------------------------------
        if not should_create_artifact(route, response_text):
            print("[POST_MODEL] Artifact creation skipped (rule not met)")

            return {
                "route": route,
                "route_confidence": route_confidence,
                "stm_written": stm_written,
                "artifact_created": False,
                "artifact": None,
                "response": response_text,
            }

        print("[POST_MODEL] Artifact creation approved")

        artifact_id = str(uuid.uuid4())
        print("[POST_MODEL] Generated artifact_id:", artifact_id)

        # ----------------------------------
        # 1️⃣ Upload full content to S3
        # ----------------------------------
        try:
            print("[POST_MODEL] Uploading content to S3...")
            content_ref = await s3_client.write_content(
                artifact_type="email",
                artifact_id=artifact_id,
                content=response_text,
            )
            print("[POST_MODEL] S3 upload successful:", content_ref)
        except Exception as s3_error:
            print("[POST_MODEL][ERROR] S3 upload failed:", str(s3_error))
            raise

        # ----------------------------------
        # 2️⃣ Persist artifact metadata (DB)
        # ----------------------------------
        try:
            print("[POST_MODEL] Writing artifact metadata to DB...")
            artifact = await artifact_repo.create_artifact(
            artifact_type="email",
            summary="Generated campaign email",
            metadata={
                "source": "llm",
                "created_by": user_id,
                "route": route,
            },
            content_ref=content_ref,
            # Remove artifact_id - method generates it automatically
        )

            print("[POST_MODEL] Artifact metadata persisted:", artifact)
        except Exception as db_error:
            print("[POST_MODEL][ERROR] Artifact DB write failed:", str(db_error))
            raise
        

        try:
            await stm_store.add_event(
            session_id="session-1",
            event_type="artifact_created",
            payload={
                "artifact_id": artifact["artifact_id"],
                "artifact_type": artifact["artifact_type"],
                "summary": artifact["summary"],
                "content_ref": artifact["content_ref"],
                "status": "draft"
            }
        )
        except Exception as stm_store_error:
            print("[POST_MODEL][ERROR] stm_store write failed:", str(stm_store_error))
            raise

    except Exception as fatal_error:
        print("[POST_MODEL][FATAL] Artifact creation aborted:", str(fatal_error))

    # ----------------------------------
    # Final response (always returned)
    # ----------------------------------
    print("[POST_MODEL] Returning response to caller")

    return {
        "route": route,
        "route_confidence": route_confidence,
        "stm_written": stm_written,
        "artifact_created": artifact is not None,
        "artifact": artifact,      # metadata only
        "response": response_text,
    }
