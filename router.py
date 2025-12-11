from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from my_json import curd_operations, proceedure_json
from actions import action
from human_input import human_call
from response_statement import response_statement
from action_agents import llm

import os
import dotenv
dotenv.load_dotenv()

client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))

checkpointer = MongoDBSaver(
    client=client,
    db_name="cx_prod"
)

# --- STATE ---
class State(TypedDict):
    step_index: int
    steps: List[Dict[str, Any]]
    user_input: str | None
    api_output: str | None
    final_response: str | None
    answers: List[str]


# --- NODES ---
def entry(state: State):
    """Entry node that passes through current state"""
    return {"step_index": state["step_index"]}


def handle_ask_user(state: State):
    """Handle user input steps"""
    current_step = state["steps"][state["step_index"]]
    question= current_step.get("action")
    question = current_step.get("question", question)
    
    # Trigger interrupt with the question
    user_val = interrupt(question)
    
    # Store the answer in answers list
    answers = state.get("answers", [])
    answers.append(user_val)
    
    return {
        "user_input": user_val,
        "answers": answers,
        "step_index": state["step_index"] + 1
    }


def handle_api_call(state: State):
    """Handle API call steps using tool calling agent"""
    current_step = state["steps"][state["step_index"]]
    action_id = current_step.get("action_id")

    if not action_id:
        return {
            "api_output": "Error: No action_id specified",
            "step_index": state["step_index"] + 1
        }

    try:
        # Fetch the tool from your action() resolver
        selected_tool = action(action_id)
        if selected_tool is None:
            raise ValueError(f"action() returned None for action_id={action_id}")

        # Create a prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided tool to complete the task."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create tool calling agent
        agent = create_tool_calling_agent(llm, [selected_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[selected_tool], verbose=True)

        # Provide full context so LLM can build structured tool input
        context = {
            "action_id": action_id,
            "user_input": state.get("user_input"),
            "previous_api": state.get("api_output"),
            "answers": state.get("answers", []),
        }

        input_text = f"""
Execute the tool `{selected_tool.name}` using the following context:
- Action ID: {action_id}
- User Input: {state.get("user_input")}
- Previous API Output: {state.get("api_output")}
- All Answers: {state.get("answers", [])}

Use the tool with appropriate parameters based on this context.
        """

        # Call the agent executor
        result = agent_executor.invoke({"input": input_text})

        # Extract output
        output = result.get("output", str(result))

        return {
            "api_output": output,
            "step_index": state["step_index"] + 1
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in handle_api_call: {error_details}")
        return {
            "api_output": f"Error calling {action_id}: {str(e)}",
            "step_index": state["step_index"] + 1
        }


def handle_final(state: State):
    """Generate final response"""
    try:
        result = response_statement(
            user_input=state.get("user_input"),
            api_output=state.get("api_output"),
            answers=state.get("answers", [])
        )
        return {"final_response": result}
    except TypeError:
        # If response_statement doesn't accept params, call without them
        result = response_statement()
        return {"final_response": result}


def router(state: State):
    """Route to next node based on current step"""
    idx = state["step_index"]
    steps = state["steps"]

    if idx >= len(steps):
        return END

    step = steps[idx]
    step_type = step.get("type")

    if step_type == "ASK_USER":
        return "ask_user"
    elif step_type == "API_CALL":
        return "api_call"
    elif step_type == "RESPOND_FINAL":
        return "final"
    else:
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
workflow = graph.compile(checkpointer=checkpointer)


# --- RUN ---
def run_workflow():
    """Execute the workflow with interactive user input"""
    config = {"configurable": {"thread_id": "order_cancellation_session"}}

    initial_state = {
        "step_index": 0,
        "steps": proceedure_json["steps"],
        "user_input": None,
        "api_output": None,
        "final_response": None,
        "answers": []
    }

    print("--- STARTING PROCEDURE ---\n")

    # Start the workflow
    try:
        workflow.invoke(initial_state, config)
    except Exception as e:
        print(f"Error starting workflow: {e}")
        import traceback
        traceback.print_exc()
        return

    # Interactive loop to handle interrupts
    while True:
        try:
            snapshot = workflow.get_state(config)

            # Check if workflow is complete
            if not snapshot.next:
                print("\n✅ Process Complete!")
                final = snapshot.values.get("final_response")
                if final:
                    print(f"\nFinal Response:\n{final}")
                break

            # Check if workflow is paused (interrupted)
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                # Get the question from the interrupt
                interrupt_data = snapshot.tasks[0].interrupts[0]
                bot_message = interrupt_data.value
                
                print(f"\n🤖 AI: {bot_message}")

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
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    run_workflow()