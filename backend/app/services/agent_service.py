from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.verification_service import verification_service

from app.tools.document_tools import document_search
from app.tools.kg_tools import knowledge_graph_lookup
from app.tools.utility_tools import summarize_context, calculator
from app.tools.web_tools import web_search
from app.core.config import settings
from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

class AgentService:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.tools = [
            document_search,
            knowledge_graph_lookup,
            summarize_context,
            calculator,
            web_search,
        ]

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=(
                "You are a grounded research assistant. "
                "Prefer local document_search first for document-based questions. "
                "Use knowledge_graph_lookup when relationships between entities matter. "
                "Use calculator for arithmetic instead of mental math. "
                "Use web_search only when local documents are insufficient or the user asks for current external information. "
                "Do not fabricate citations or facts. "
                "Answer clearly and include the evidence you used."
            ),
        )
        
    def run(self, user_message: str, history: list[dict] | None = None) -> str:
        messages = []

        if history:
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        logger.info("Agent run started | message_len=%d", len(user_message))

        result = self.agent.invoke({"messages": messages})
        result_messages = result.get("messages", [])

        if not result_messages:
            logger.warning("Agent returned no messages")
            return "I could not generate a response."

        final_message = result_messages[-1]
        content = getattr(final_message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                return "\n".join(text_parts).strip()

        return str(content) if content is not None else "I could not generate a response."



agent_service = AgentService()