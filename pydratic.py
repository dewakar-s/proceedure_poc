from pydantic import BaseModel

class PlanRequest(BaseModel):
    user_query: str

class ChatRequest(BaseModel):
    tenant_id: str
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    status: str
    question: str | None = None
    api_output: str | None = None
    final_response: str | None = None
    answers: list[str] | None = None