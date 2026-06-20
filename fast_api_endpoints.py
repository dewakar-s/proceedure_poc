from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uuid

from chat_graph_implementation import create_chat_graph  # your function

app = FastAPI(title="Chat Graph API")

# -------------------------------
# Request / Response models
# -------------------------------

class ChatRequest(BaseModel):
    message: str
    tenant_id: str
    session_id: str | None = None
    retriever_mode: str = "agent_access"
    mode: str = "Ai_agent"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    procedure_triggered: bool


# -------------------------------
# Endpoint
# -------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    try:
        session_id = payload.session_id or str(uuid.uuid4())

        graph = create_chat_graph(
            tenant_id=payload.tenant_id,
            retriever_mode=payload.retriever_mode,
            mode=payload.mode
        )

        initial_state = {
            "messages": [HumanMessage(content=payload.message)],
            "session_id": session_id,
            "tenant_id": payload.tenant_id,
            "persona_data": None,
            "procedure_triggered": False
        }

        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        last_message = final_state["messages"][-1]

        return ChatResponse(
            session_id=session_id,
            response=last_message.content,
            procedure_triggered=final_state["procedure_triggered"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
