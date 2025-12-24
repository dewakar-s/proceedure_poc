from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from router import  workflow
from fastapi import  Header
from json_generation import generate_plan
from pydratic import PlanRequest, ChatRequest, ChatResponse
from mongodb_uttilies_procedure import detect_procedure, collection_procedure, collection_procedure_json, get_procedure_by_id

app = FastAPI()

@app.post("/generate-plan")
def generate_procedure_plan(
    request: PlanRequest,
    tenant_id: str = Header(..., alias="X-Tenant-Id"),
):
    plan = generate_plan(request.user_query, tenant_id)

    return {
        "status": "SUCCESS",
        "execution_plan": plan
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "tenant_id": req.tenant_id
        }
    }

    snapshot = workflow.get_state(config)

    # ---------------- FIRST MESSAGE ----------------
    if snapshot is None or not snapshot.values:
        procedure_id = detect_procedure(req.message, req.tenant_id)

        if not procedure_id:
            return ChatResponse(
                status="NO_PROCEDURE",
                final_response="No matching procedure found."
            )

        procedure = get_procedure_by_id(
            collection_procedure_json,
            procedure_id,
            req.tenant_id
        )
        print("Fetched procedure:", procedure)

        workflow.invoke({
            "step_index": 0,
            "steps": procedure["steps"],
            "user_input": None,
            "api_output": None,
            "final_response": None,
            "answers": []
        }, config)

    # ---------------- RESUME ----------------
    else:
        workflow.invoke(Command(resume=req.message), config)

    snapshot = workflow.get_state(config)
    values = snapshot.values or {}

    # ---------------- WAITING ----------------
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        return ChatResponse(
            status="WAITING_FOR_USER",
            question=snapshot.tasks[0].interrupts[0].value,
            api_output=values.get("api_output"),   # ✅ PREVIOUS API OUTPUT
            answers=values.get("answers")
        )

    # ---------------- COMPLETED ----------------
    return ChatResponse(
        status="COMPLETED",
        final_response=values.get("final_response"),
        api_output=values.get("api_output"),
        answers=values.get("answers")
    )