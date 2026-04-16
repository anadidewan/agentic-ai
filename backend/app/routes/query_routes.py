from fastapi import APIRouter
from app.schemas.question_schema import QuestionRequest, QuestionResponse
from app.services.agent_service import agent_service

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    answer = agent_service.run(request.question)
    return QuestionResponse(
        question=request.question,
        answer=answer,
        sources=[]
    )