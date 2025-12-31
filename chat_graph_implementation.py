import os
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from actions import action
from mongodb_utilies_actions import MONGODB_ATLAS_URI

logger = logging.getLogger(__name__)

# ============================================================================
# PROCEDURE GRAPH STATE
# ============================================================================

class ProcedureState(TypedDict):
    """State for procedure graph - focused on step execution"""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    session_id: str
    tenant_id: str
    
    # Procedure execution state
    in_procedure: bool
    procedure_name: Optional[str]
    procedure_steps: List[Dict[str, Any]]
    step_index: int
    
    # Step data
    procedure_answers: List[str]
    current_user_input: Optional[str]
    last_api_output: Optional[str]
    
    # Flow control
    waiting_for_input: bool
    fallback_to_chat: bool
    validation_attempts: int

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
# CONFIGURATION
# ============================================================================

MAX_VALIDATION_ATTEMPTS = 2  # After 2 invalid inputs, fallback to chat

# ============================================================================
# PROCEDURE GRAPH NODES
# ============================================================================

def procedure_entry_node(state: ProcedureState) -> ProcedureState:
    """Entry point - validates procedure state"""
    logger.info(f"[ProcedureGraph] Entry for session: {state['session_id']}")
    
    if not state.get('in_procedure'):
        logger.warning(f"[ProcedureGraph] No active procedure, marking for fallback")
        state['fallback_to_chat'] = True
    
    return state

def ask_user_node(state: ProcedureState) -> ProcedureState:
    """Handle ASK_USER step type"""
    current_step = state['procedure_steps'][state['step_index']]
    question = current_step.get("message") or current_step.get("question")
    
    logger.info(f"[ProcedureGraph] ASK_USER step {state['step_index']}: {question}")
    
    # If we have user input, validate it
    if state.get('current_user_input'):
        # Validation logic here
        valid = validate_user_input(
            user_input=state['current_user_input'],
            step=current_step
        )
        
        if not valid:
            state['validation_attempts'] = state.get('validation_attempts', 0) + 1
            
            if state['validation_attempts'] >= MAX_VALIDATION_ATTEMPTS:
                logger.warning(f"[ProcedureGraph] Max validation attempts reached, falling back to chat")
                state['fallback_to_chat'] = True
                
                # Add message explaining fallback
                fallback_msg = AIMessage(
                    content="I'm having trouble understanding that input. Let me connect you with more flexible assistance."
                )
                state['messages'].append(fallback_msg)
                return state
            
            # Ask again with clarification
            retry_msg = AIMessage(
                content=f"I didn't quite get that. {question}"
            )
            state['messages'].append(retry_msg)
            state['waiting_for_input'] = True
            return state
        
        # Valid input - store and move to next step
        state['procedure_answers'].append(state['current_user_input'])
        state['validation_attempts'] = 0  # Reset
        state['step_index'] += 1
        state['current_user_input'] = None
        state['waiting_for_input'] = False
    
    else:
        # First time asking - just present the question
        question_msg = AIMessage(content=question)
        state['messages'].append(question_msg)
        state['waiting_for_input'] = True
    
    return state

def api_call_node(state: ProcedureState) -> ProcedureState:
    """Handle API_CALL step type"""
    current_step = state['procedure_steps'][state['step_index']]
    action_id = current_step.get("action_id")
    
    logger.info(f"[ProcedureGraph] API_CALL step {state['step_index']}: action_id={action_id}")
    
    if not action_id:
        state['last_api_output'] = "Error: No action_id specified"
        state['step_index'] += 1
        return state
    
    try:
        selected_tool = action(action_id)
        if selected_tool is None:
            raise ValueError(f"action() returned None for action_id={action_id}")
        
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
- Action ID: {action_id}
- Current User Input: {state.get('current_user_input')}
- Previous API Output: {state.get('last_api_output')}
- All Previous Answers: {state.get('procedure_answers', [])}

Use the tool with appropriate parameters based on this context.
        """
        
        result = agent_executor.invoke({"input": input_text})
        output = result.get("output", str(result))
        
        state['last_api_output'] = output
        state['step_index'] += 1
        
        logger.info(f"[ProcedureGraph] API call successful: {output[:100]}...")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[ProcedureGraph] API call error: {error_details}")
        
        state['last_api_output'] = f"Error calling {action_id}: {str(e)}"
        state['fallback_to_chat'] = True
    
    return state

def final_response_node(state: ProcedureState) -> ProcedureState:
    """Generate final response for completed procedure"""
    logger.info(f"[ProcedureGraph] Generating final response")
    
    try:
        from response_statement import response_statement
        
        result = response_statement(
            user_input=state.get('current_user_input'),
            api_output=state.get('last_api_output'),
            answers=state.get('procedure_answers', [])
        )
        
        # Add final response
        final_msg = AIMessage(content=result)
        state['messages'].append(final_msg)
        
        # Mark procedure as complete
        state['in_procedure'] = False
        state['waiting_for_input'] = False
        
        logger.info(f"[ProcedureGraph] Procedure completed successfully")
        
    except Exception as e:
        logger.error(f"[ProcedureGraph] Error in final response: {e}")
        error_msg = AIMessage(
            content="I apologize, but I encountered an error completing this procedure."
        )
        state['messages'].append(error_msg)
        state['fallback_to_chat'] = True
    
    return state

# ============================================================================
# VALIDATION HELPER
# ============================================================================

def validate_user_input(user_input: str, step: Dict[str, Any]) -> bool:
    """
    Validate user input for a step
    Can be extended with specific validation rules per step type
    """
    if not user_input or not user_input.strip():
        return False
    
    # Add more validation logic based on step requirements
    validation_rules = step.get('validation_rules', {})
    
    if validation_rules.get('min_length'):
        if len(user_input) < validation_rules['min_length']:
            return False
    
    if validation_rules.get('expected_type') == 'number':
        try:
            float(user_input)
        except ValueError:
            return False
    
    # Add more validation as needed
    
    return True

# ============================================================================
# ROUTING
# ============================================================================

def procedure_router(state: ProcedureState) -> str:
    """Route to appropriate procedure step or end"""
    
    # Check for fallback first
    if state.get('fallback_to_chat'):
        logger.info(f"[ProcedureGraph] Routing to fallback")
        return "end"
    
    # Check if waiting for user input
    if state.get('waiting_for_input'):
        logger.info(f"[ProcedureGraph] Waiting for user input")
        return "end"  # Pause and wait for next request
    
    # Check if procedure is complete
    if not state.get('in_procedure'):
        logger.info(f"[ProcedureGraph] Procedure complete")
        return "end"
    
    # Determine next step
    idx = state.get('step_index', 0)
    steps = state.get('procedure_steps', [])
    
    if idx >= len(steps):
        logger.info(f"[ProcedureGraph] All steps processed, generating final response")
        return "final_response"
    
    step = steps[idx]
    step_type = step.get('type')
    
    logger.info(f"[ProcedureGraph] Next step type: {step_type}")
    
    if step_type == "ASK_USER":
        return "ask_user"
    elif step_type == "API_CALL":
        return "api_call"
    elif step_type == "RESPOND_FINAL":
        return "final_response"
    else:
        logger.warning(f"[ProcedureGraph] Unknown step type: {step_type}")
        return "end"

# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_procedure_graph(tenant_id: str, mode: str):
    """Create the procedure graph for step-based workflows"""
    start_time = time.perf_counter()
    
    # Create graph
    workflow = StateGraph(ProcedureState)
    
    # Add nodes
    workflow.add_node("entry", procedure_entry_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("api_call", api_call_node)
    workflow.add_node("final_response", final_response_node)
    
    # Set entry point
    workflow.set_entry_point("entry")
    
    # Add edges
    workflow.add_conditional_edges(
        "entry",
        procedure_router,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "ask_user",
        procedure_router,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "api_call",
        procedure_router,
        {
            "ask_user": "ask_user",
            "api_call": "api_call",
            "final_response": "final_response",
            "end": END
        }
    )
    
    workflow.add_edge("final_response", END)
    
    # Choose checkpointer
    if mode == "Ai_agent":
        checkpointer = MongoDBSaver(
            client=mongo_client,
            db_name="rag_bot",
            collection_name="procedure_checkpoints"
        )
    else:
        checkpointer = MemorySaver()
    
    # Compile graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info(f"[ProcedureGraph] Graph created in {time.perf_counter() - start_time:.3f}s")
    
    return graph