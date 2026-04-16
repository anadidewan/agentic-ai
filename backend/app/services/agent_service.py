from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.verification_service import verification_service

from app.tools.document_tools import document_search
from app.tools.kg_tools import knowledge_graph_lookup
from app.tools.utility_tools import summarize_context, calculator
from app.tools.web_tools import web_search
from app.core.config import settings


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
    def run(self, question: str) -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        messages = result.get("messages", [])
        if not messages:
            return "I could not generate a response."

        answer = messages[-1].content
        verification = verification_service.verify(question, answer)

        if verification.needs_retry:
            retry_prompt = (
                f"The previous answer was insufficient.\n"
                f"Feedback: {verification.feedback}\n"
                f"Question: {question}\n"
                f"Try again, use tools more effectively, and provide a grounded answer."
            )

            retry_result = self.agent.invoke(
                {"messages": [{"role": "user", "content": retry_prompt}]}
            )
            retry_messages = retry_result.get("messages", [])
            if retry_messages:
                return retry_messages[-1].content

        return answer


agent_service = AgentService()