from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from pydratic import StartRequest, ResumeRequest
from router import workflow

app = FastAPI()

def get_snapshot(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return workflow.get_state(config), config



@app.post("/workflow/start")
def start_workflow(req: StartRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    initial_state = {
        "step_index": 0,
        "steps": req.steps,
        "user_input": None,
        "api_output": None,
        "final_response": None,
        "answers": []
    }

    workflow.invoke(initial_state, config)

    snapshot = workflow.get_state(config)

    # If interrupted → return question
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        question = snapshot.tasks[0].interrupts[0].value
        return {
            "status": "WAITING_FOR_USER",
            "question": question
        }

    return {"status": "RUNNING"}



@app.post("/workflow/resume")
def resume_workflow(req: ResumeRequest):
    snapshot, config = get_snapshot(req.thread_id)

    workflow.invoke(
        Command(resume=req.user_input),
        config
    )

    snapshot = workflow.get_state(config)

    # Completed
    if not snapshot.next:
        return {
            "status": "COMPLETED",
            "final_response": snapshot.values.get("final_response")
        }

    # Next interrupt
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        question = snapshot.tasks[0].interrupts[0].value
        return {
            "status": "WAITING_FOR_USER",
            "question": question
        }

    return {"status": "RUNNING"}
