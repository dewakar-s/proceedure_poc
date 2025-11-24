from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from my_json import proceedure_json
from actions import action
from human_input import human_input
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
    # Return same step_index to satisfy StateGraph
    return {"step_index": state["step_index"]}



def handle_ask_user(state: State):
    user_val = human_input()
    return {"user_input": user_val, "step_index": state["step_index"] + 1}


def handle_api_call(state: State):
    out = action()
    return {"api_output": out, "step_index": state["step_index"] + 1}


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

# COMPILE
workflow = graph.compile()


# --- RUN ---
# --- RUN ---
initial_state = {
    "step_index": 0,
    "steps": proceedure_json["steps"],
    "user_input": None,
    "api_output": None,
    "final_response": None
}

result_state = workflow.invoke(initial_state)

print("Final state:", result_state)


