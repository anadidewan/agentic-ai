from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings


class VerificationResult(BaseModel):
    is_answered: bool
    is_grounded: bool
    needs_retry: bool
    feedback: str


class VerificationService:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def verify(self, question: str, answer: str) -> VerificationResult:
        prompt = f"""
You are validating an answer.

Question:
{question}

Answer:
{answer}

Return JSON with:
- is_answered: whether the answer addresses the question
- is_grounded: whether the answer appears supported rather than speculative
- needs_retry: true if the system should try again with a better search/tool path
- feedback: short explanation
"""
        raw = self.model.invoke(prompt).content

        # In production,I will use structured output/parser instead of naive eval/json loading
        import json
        try:
            data = json.loads(raw)
            return VerificationResult(**data)
        except Exception:
            return VerificationResult(
                is_answered=True,
                is_grounded=False,
                needs_retry=False,
                feedback="Verification parser fallback triggered."
            )


verification_service = VerificationService()