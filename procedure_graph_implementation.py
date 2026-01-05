import os
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import interrupt
from pymongo import MongoClient

from actions import action
from mongodb_utilies_actions import MONGODB_ATLAS_URI
from response_statement import response_statement

logger = logging.getLogger(__name__)

# ============================================================================
# PROCEDURE GRAPH STATE (Matching Your Structure)
# ============================================================================

class ProcedureState(TypedDict):
    """State matching your exact structure"""
    step_index: int
    steps: List[Dict[str, Any]]
    user_input: Optional[str]
    api_output: Optional[str]
    final_response: Optional[str]
    answers: List[str]
    validation_failed: bool
    original_user_message: Optional[str]

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2025-03-01-preview"

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION,
    temperature=0.7,
)

mongo_client = MongoClient(MONGODB_ATLAS_URI)

# ============================================================================
# PROCEDURE NODES
# ============================================================================

def route_step(state: ProcedureState) -> str:
    """Route to appropriate node based on current step"""
    steps = state["steps"]
    index = state["step_index"]
    
    # Check if all steps completed
    if index >= len(steps):
        logger.info(f"[ProcedureGraph] All steps completed, generating final response")
        return "final_response"
    
    current_step = steps[index]
    step_type = current_step.get("type")
    
    logger.info(f"[ProcedureGraph] Step {index}/{len(steps)}: {step_type}")
    
    if step_type == "ASK_USER":
        return "ask_user"
    elif step_type == "API_CALL":
        return "api_call"
    elif step_type == "RESPOND_FINAL":
        return "final_response"
    else:
        logger.warning(f"[ProcedureGraph] Unknown step type: {step_type}, ending")
        return "end"

def ask_user_node(state: ProcedureState) -> ProcedureState:
    """
    Handle ASK_USER step - asks question and waits for input
    Uses interrupt() to pause execution
    """
    current_step = state["steps"][state["step_index"]]
    question = current_step.get("message") or current_step.get("question")
    
    logger.info(f"[ProcedureGraph] Asking user: {question}")
    
    # Use interrupt to pause and wait for user input
    # The value returned from interrupt() will be the user's response
    user_response = interrupt(question)
    
    logger.info(f"[ProcedureGraph] User responded: {user_response}")
    
    # Store the answer
    answers = state["answers"].copy()
    answers.append(user_response)
    
    # Move to next step
    return {
        **state,
        "user_input": user_response,
        "answers": answers,
        "step_index": state["step_index"] + 1
    }

def api_call_node(state: ProcedureState) -> ProcedureState:
    """
    Handle API_CALL step - executes the specified action
    """
    current_step = state["steps"][state["step_index"]]
    action_id = current_step.get("action_id")
    
    logger.info(f"[ProcedureGraph] Executing API call: {action_id}")
    
    if not action_id:
        logger.error(f"[ProcedureGraph] No action_id specified in step")
        return {
            **state,
            "api_output": "Error: No action_id specified",
            "step_index": state["step_index"] + 1
        }
    
    try:
        # Get the tool for this action
        selected_tool = action(action_id)
        
        if selected_tool is None:
            raise ValueError(f"action() returned None for action_id={action_id}")
        
        logger.info(f"[ProcedureGraph] Using tool: {selected_tool.name}")
        
        # Create agent with the tool
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided tool to complete the task."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, [selected_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[selected_tool], verbose=True)
        
        # Build context from previous steps
        input_text = f"""
Execute the tool `{selected_tool.name}` using the following context:

Action ID: {action_id}
Current User Input: {state.get('user_input')}
Previous API Output: {state.get('api_output')}
All Previous Answers: {state.get('answers', [])}

Use the tool with appropriate parameters based on this context.
        """
        
        # Execute the tool
        result = agent_executor.invoke({"input": input_text})
        output = result.get("output", str(result))
        
        logger.info(f"[ProcedureGraph] API call successful: {output[:100]}...")
        
        return {
            **state,
            "api_output": output,
            "step_index": state["step_index"] + 1
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[ProcedureGraph] API call error: {error_details}")
        
        return {
            **state,
            "api_output": f"Error calling {action_id}: {str(e)}",
            "step_index": state["step_index"] + 1
        }

def final_response_node(state: ProcedureState) -> ProcedureState:
    """
    Generate final response using response_statement function
    """
    logger.info(f"[ProcedureGraph] Generating final response")
    
    try:
        # Use your response_statement function
        final_text = response_statement(
            user_input=state.get("user_input"),
            api_output=state.get("api_output"),
            answers=state.get("answers", [])
        )
        
        logger.info(f"[ProcedureGraph] Final response generated: {final_text[:100]}...")
        
        return {
            **state,
            "final_response": final_text
        }
        
    except Exception as e:
        logger.error(f"[ProcedureGraph] Error generating final response: {e}")
        
        return {
            **state,
            "final_response": "I apologize, but I encountered an error completing this procedure. Please try again or contact support."
        }

# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_procedure_graph(tenant_id: str, mode: str):
    """
    Create the procedure graph matching your exact flow
    
    Flow:
    1. route_step → determines which node to execute
    2. ask_user → uses interrupt() to pause and wait
    3. api_call → executes the action
    4. final_response → generates final message
    """
    start_time = time.perf_counter()
    
    # Create graph
    workflow = StateGraph(ProcedureState)
    
    # Add nodes
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("api_call", api_call_node)
    workflow.add_node("final_response_node", final_response_node)
    
    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_step,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response_node",
            "end": END
        }
    )
    
    # Add conditional edges from each node back to router
    workflow.add_conditional_edges(
        "ask_user",
        route_step,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response_node",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "api_call",
        route_step,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response_node",
            "end": END
        }
    )
    
    # Final response ends the graph
    workflow.add_edge("final_response_node", END)

    # Choose checkpointer
    if mode == "Ai_agent":
        checkpointer = MongoDBSaver(
            client=mongo_client,
            db_name="rag_bot",
            collection_name="procedure_checkpoints"
        )
    else:
        checkpointer = MemorySaver()
    
    # Compile graph with interrupt support
    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["ask_user"])
    
    logger.info(f"[ProcedureGraph] Graph created in {time.perf_counter() - start_time:.3f}s")
    
    return graph