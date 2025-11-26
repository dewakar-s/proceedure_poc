from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from my_json import proceedure_json
from actions import action
from human_input import human_call
from response_statement import response_statement


# --- STATE ---
class State(TypedDict):
    step_index: int
    steps: List[Dict[str, Any]]
    user_input: str | None
    api_output: str | None
    final_response: str | None


# --- NODES ---
def entry(state: State):
    return {"step_index": state["step_index"]}


def handle_ask_user(state: State):
    # Get the current step to access the question
    current_step = state["steps"][state["step_index"]]
    question = current_step.get("question", "Please provide input:")
    
    # Trigger interrupt with the question
    user_val = interrupt(question)
    
    # After resume, user_val will contain the input
    return {
        "user_input": user_val, 
        "step_index": state["step_index"] + 1
    }


def handle_api_call(state: State):
    out = action()
    return {
        "api_output": out, 
        "step_index": state["step_index"] + 1
    }


def handle_final(state: State):
    result = response_statement()
    return {"final_response": result}


def router(state: State):
    idx = state["step_index"]
    steps = state["steps"]

    if idx >= len(steps):
        return END

    step = steps[idx]

    match step["type"]:
        case "ASK_USER":
            return "ask_user"
        case "API_CALL":
            return "api_call"
        case "RESPOND_FINAL":
            return "final"
        case _:
            return END


# --- BUILD GRAPH ---
graph = StateGraph(State)

graph.add_node("ask_user", handle_ask_user)
graph.add_node("api_call", handle_api_call)
graph.add_node("final", handle_final)
graph.add_node("entry_node", entry)

graph.add_conditional_edges(
    "entry_node",
    router,
    {
        "ask_user": "ask_user",
        "api_call": "api_call",
        "final": "final",
        END: END
    }
)

graph.add_edge("ask_user", "entry_node")
graph.add_edge("api_call", "entry_node")
graph.add_edge("final", END)

graph.set_entry_point("entry_node")

# COMPILE with checkpointer
checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# --- RUN ---
config = {"configurable": {"thread_id": "order_cancellation_session"}}

initial_state = {
    "step_index": 0,
    "steps": proceedure_json["steps"],
    "user_input": None,
    "api_output": None,
    "final_response": None
}

print("--- STARTING PROCEDURE ---\n")

# Start the workflow
workflow.invoke(initial_state, config)

# Interactive loop to handle interrupts
while True:
    snapshot = workflow.get_state(config)

    # Check if workflow is complete
    if not snapshot.next:
        print("\n✅ Process Complete!")
        final = snapshot.values.get("final_response")
        if final:
            print(f"Final Response: {final}")
        break

    # Check if workflow is paused (interrupted)
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        # Get the question from the interrupt
        bot_message = snapshot.tasks[0].interrupts[0].value
        print(f"🤖 AI: {bot_message}")

        # Get user input
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("👋 Session ended.")
            break

        # Resume the workflow with user input
        workflow.invoke(Command(resume=user_input), config)
    else:
        # No interrupts but graph still has next steps - continue
        workflow.invoke(None, config)