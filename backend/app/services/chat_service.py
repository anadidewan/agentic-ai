from app.services.agent_service import agent_service
from fastapi import HTTPException

from app.store.chat_store import (
    save_message,
    get_recent_messages,
)
from app.services.retrieval_service import hybrid_retrieve, graph_expand, get_graph_context
from app.services.llm_service import generate_answer, generate_direct_answer, generate_critique_answer, _call_gemini
from app.services.router_service import rewrite_and_route
from app.config import settings
import time
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)


def process_chat_message(session_id: str, user_message: str) -> dict:
    # Save current user message
    save_message(session_id, "user", user_message)

    # Load recent history
    history = get_recent_messages(session_id, limit=6)
    # Build retrieval query
    # routing_info = rewrite_and_route(history, user_message)
    # retrieval_query = routing_info["rewritten_query"]
    # routing = routing_info["mode"]
    # logger.debug("Retrieval query built | session=%s | query=%.120s", session_id, retrieval_query)

    


    try:
        
        logger.info("Agent-based chat processing | session=%s", session_id)
        agent_result = agent_service.run(user_message=user_message, history=history)
        answer = agent_result["answer"]


        save_message(session_id, "assistant", answer)

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": agent_result.get("sources", []),
            "retrieved_chunks": agent_result.get("retrieved_chunks", []),
            "mode": "agent",
            "routing_label": "agent",
        }

    except ValueError as e:
        logger.error("Chat processing ValueError | session=%s | error=%s", session_id, e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Chat processing failed | session=%s | error=%s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Save assistant reply
