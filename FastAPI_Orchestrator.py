import os
import logging
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langgraph.types import Command

from chat_graph_implementation import create_chat_graph
from procedure_graph_implementation import create_procedure_graph
from mongodb_utilies_actions import MONGODB_ATLAS_URI
from mongodb_uttilies_procedure import detect_procedure, collection_procedure_json, get_procedure_by_id

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production Customer Support Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Thread/Session identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    message: str = Field(..., description="User message")
    mode: str = Field(default="Ai_agent", description="Mode: 'Ai_agent' or 'memory'")
    retriever_mode: str = Field(default="hybrid", description="Retriever mode")

class ChatResponse(BaseModel):
    status: str  # "WAITING_FOR_USER", "COMPLETED", "NO_PROCEDURE", "CHAT_RESPONSE"
    question: Optional[str] = None
    final_response: Optional[str] = None
    api_output: Optional[str] = None
    answers: Optional[list] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# VALIDATION HELPER
# ============================================================================

def validate_user_input_for_step(user_input: str, step: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate user input against step requirements
    
    Returns:
        (is_valid, error_message)
    """
    if not user_input or not user_input.strip():
        return False, "Input cannot be empty"
    
    # Get the question/message that was asked
    step_message = step.get('message') or step.get('question', '')
    
    # Add your validation rules here
    validation_rules = step.get('validation_rules', {})
    
    # Example: minimum length
    if validation_rules.get('min_length'):
        if len(user_input.strip()) < validation_rules['min_length']:
            return False, f"Input must be at least {validation_rules['min_length']} characters"
    
    # Example: expected type (number, email, etc.)
    expected_type = validation_rules.get('type')
    if expected_type == 'number':
        try:
            float(user_input)
        except ValueError:
            return False, "Please enter a valid number"
    
    if expected_type == 'email':
        if '@' not in user_input:
            return False, "Please enter a valid email address"
    
    # Example: order number format
    if 'order' in step_message.lower() and 'number' in step_message.lower():
        # Check if input looks like an order number (e.g., ORD-12345)
        if len(user_input) < 5:
            return False, "Order number seems too short. Please check and try again"
    
    # Add more validation logic based on your requirements
    
    return True, None

# ============================================================================
# GRAPH INSTANCES (CACHED)
# ============================================================================

_chat_graphs = {}
_procedure_graphs = {}

def get_chat_graph(tenant_id: str, retriever_mode: str, mode: str):
    """Get or create chat graph (with caching)"""
    key = f"{tenant_id}_{retriever_mode}_{mode}"
    if key not in _chat_graphs:
        _chat_graphs[key] = create_chat_graph(tenant_id, retriever_mode, mode)
    return _chat_graphs[key]

def get_procedure_graph(tenant_id: str, mode: str):
    """Get or create procedure graph (with caching)"""
    key = f"{tenant_id}_{mode}"
    if key not in _procedure_graphs:
        _procedure_graphs[key] = create_procedure_graph(tenant_id, mode)
    return _procedure_graphs[key]

# ============================================================================
# MAIN CHAT ENDPOINT
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint with automatic routing:
    1. First message → detect_procedure → get_procedure_by_id → start procedure
    2. Resume message → continue procedure OR fallback to chat if invalid
    3. No procedure → use chat graph
    """
    start_time = time.perf_counter()
    
    try:
        # Config for LangGraph
        config = {
            "configurable": {
                "thread_id": req.thread_id,
                "tenant_id": req.tenant_id
            }
        }
        
        # Get procedure graph
        procedure_graph = get_procedure_graph(req.tenant_id, req.mode)
        snapshot = procedure_graph.get_state(config)
        
        # ================================================================
        # CASE 1: FIRST MESSAGE - Detect and Initialize Procedure
        # ================================================================
        if snapshot is None or not snapshot.values:
            logger.info(f"[Orchestrator] First message for thread {req.thread_id}")
            
            # Detect procedure using your function
            procedure_id = detect_procedure(req.message, req.tenant_id)
            
            if not procedure_id:
                logger.info(f"[Orchestrator] No procedure detected, using chat graph")
                
                # Use chat graph for non-procedure queries
                return handle_chat_query(req, start_time)
            
            # Get procedure JSON
            procedure = get_procedure_by_id(
                collection_procedure_json,
                procedure_id,
                req.tenant_id
            )
            
            logger.info(f"[Orchestrator] Procedure detected: {procedure.get('procedure_name')}")
            logger.info(f"[Orchestrator] Steps: {len(procedure.get('steps', []))}")
            
            # Initialize procedure graph
            procedure_graph.invoke({
                "step_index": 0,
                "steps": procedure["steps"],
                "user_input": None,
                "api_output": None,
                "final_response": None,
                "answers": [],
                "validation_failed": False,
                "original_user_message": req.message  # Store for potential fallback
            }, config)
            
            # Get updated state
            snapshot = procedure_graph.get_state(config)
            values = snapshot.values or {}
            
            # Check if waiting for user input
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                execution_time = time.perf_counter() - start_time
                
                return ChatResponse(
                    status="WAITING_FOR_USER",
                    question=snapshot.tasks[0].interrupts[0].value,
                    api_output=values.get("api_output"),
                    answers=values.get("answers"),
                    execution_time=execution_time,
                    metadata={
                        "procedure_started": True,
                        "procedure_name": procedure.get("procedure_name"),
                        "current_step": values.get("step_index", 0),
                        "total_steps": len(procedure.get("steps", []))
                    }
                )
            
            # If completed immediately (no user input needed)
            execution_time = time.perf_counter() - start_time
            return ChatResponse(
                status="COMPLETED",
                final_response=values.get("final_response"),
                api_output=values.get("api_output"),
                answers=values.get("answers"),
                execution_time=execution_time
            )
        
        # ================================================================
        # CASE 2: RESUME - Continue Procedure with Validation
        # ================================================================
        else:
            logger.info(f"[Orchestrator] Resuming procedure for thread {req.thread_id}")
            
            values = snapshot.values or {}
            current_step_index = values.get("step_index", 0)
            steps = values.get("steps", [])
            
            # Get current step
            if current_step_index < len(steps):
                current_step = steps[current_step_index]
                
                # =========================================================
                # VALIDATION: Check if user input is valid for this step
                # =========================================================
                if current_step.get("type") == "ASK_USER":
                    is_valid, error_message = validate_user_input_for_step(
                        req.message, 
                        current_step
                    )
                    
                    if not is_valid:
                        logger.warning(f"[Orchestrator] Invalid input: {error_message}")
                        logger.info(f"[Orchestrator] Falling back to chat graph")
                        
                        # FALLBACK TO CHAT GRAPH
                        return handle_chat_query_with_context(
                            req, 
                            start_time,
                            context={
                                "procedure_failed": True,
                                "validation_error": error_message,
                                "original_question": current_step.get("message") or current_step.get("question"),
                                "procedure_name": values.get("procedure_name")
                            }
                        )
            
            # Valid input - continue procedure
            procedure_graph.invoke(Command(resume=req.message), config)
            
            # Get updated state
            snapshot = procedure_graph.get_state(config)
            values = snapshot.values or {}
            
            # Check if still waiting for input
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                execution_time = time.perf_counter() - start_time
                
                return ChatResponse(
                    status="WAITING_FOR_USER",
                    question=snapshot.tasks[0].interrupts[0].value,
                    api_output=values.get("api_output"),
                    answers=values.get("answers"),
                    execution_time=execution_time,
                    metadata={
                        "current_step": values.get("step_index", 0),
                        "total_steps": len(values.get("steps", []))
                    }
                )
            
            # Procedure completed
            execution_time = time.perf_counter() - start_time
            return ChatResponse(
                status="COMPLETED",
                final_response=values.get("final_response"),
                api_output=values.get("api_output"),
                answers=values.get("answers"),
                execution_time=execution_time,
                metadata={
                    "procedure_completed": True,
                    "total_steps": values.get("step_index", 0)
                }
            )
        
    except Exception as e:
        logger.error(f"[Orchestrator] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# ============================================================================
# CHAT GRAPH HANDLERS
# ============================================================================

def handle_chat_query(req: ChatRequest, start_time: float) -> ChatResponse:
    """Handle query using chat graph (no procedure)"""
    logger.info(f"[ChatHandler] Processing chat query for thread {req.thread_id}")
    
    # Initialize chat history
    chat_history = MongoDBChatMessageHistory(
        session_id=req.thread_id,
        connection_string=MONGODB_ATLAS_URI,
        database_name="test_rag_bot",
        collection_name="message_history"
    )
    
    # Get chat graph
    chat_graph = get_chat_graph(req.tenant_id, req.retriever_mode, req.mode)
    
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "tenant_id": req.tenant_id
        }
    }
    
    historical_messages = chat_history.messages
    
    initial_state = {
        "messages": historical_messages + [HumanMessage(content=req.message)],
        "session_id": req.thread_id,
        "tenant_id": req.tenant_id,
        "persona_data": None
    }
    
    result = chat_graph.invoke(initial_state, config=config)
    
    # Extract response
    last_message = result['messages'][-1]
    response_text = last_message.content
    
    # Save to history
    chat_history.add_user_message(req.message)
    chat_history.add_ai_message(response_text)
    
    execution_time = time.perf_counter() - start_time
    
    return ChatResponse(
        status="CHAT_RESPONSE",
        final_response=response_text,
        execution_time=execution_time,
        metadata={
            "graph_used": "chat",
            "message_count": len(result['messages'])
        }
    )

def handle_chat_query_with_context(
    req: ChatRequest, 
    start_time: float,
    context: Dict[str, Any]
) -> ChatResponse:
    """
    Handle query using chat graph with context from failed procedure
    This provides better UX by informing the chat graph about what went wrong
    """
    logger.info(f"[ChatHandler] Processing chat query with procedure context")
    
    # Initialize chat history
    chat_history = MongoDBChatMessageHistory(
        session_id=req.thread_id,
        connection_string=MONGODB_ATLAS_URI,
        database_name="test_rag_bot",
        collection_name="message_history"
    )
    
    # Get chat graph
    chat_graph = get_chat_graph(req.tenant_id, req.retriever_mode, req.mode)
    
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "tenant_id": req.tenant_id
        }
    }
    
    historical_messages = chat_history.messages
    
    # Add context message to help chat understand the situation
    context_message = f"""[System Context: User was in a {context.get('procedure_name', 'procedure')} 
but validation failed. Original question was: "{context.get('original_question')}". 
Validation error: {context.get('validation_error')}. 
User's input was: "{req.message}". 
Please help the user in a conversational way.]"""
    
    initial_state = {
        "messages": historical_messages + [HumanMessage(content=context_message + "\n\n" + req.message)],
        "session_id": req.thread_id,
        "tenant_id": req.tenant_id,
        "persona_data": None
    }
    
    result = chat_graph.invoke(initial_state, config=config)
    
    # Extract response
    last_message = result['messages'][-1]
    response_text = last_message.content
    
    # Save to history (without the context message, just user input and response)
    chat_history.add_user_message(req.message)
    chat_history.add_ai_message(response_text)
    
    execution_time = time.perf_counter() - start_time
    
    return ChatResponse(
        status="CHAT_RESPONSE",
        final_response=response_text,
        execution_time=execution_time,
        metadata={
            "graph_used": "chat",
            "fallback_from_procedure": True,
            "validation_error": context.get('validation_error')
        }
    )

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Production Customer Support Agent",
        "version": "2.0.0",
        "graphs": ["chat", "procedure"]
    }

@app.post("/reset-procedure/{thread_id}")
async def reset_procedure(thread_id: str, tenant_id: str = Header(..., alias="X-Tenant-Id")):
    """Force reset a procedure for a thread"""
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": tenant_id
            }
        }
        
        procedure_graph = get_procedure_graph(tenant_id, "Ai_agent")
        
        # Clear the state by updating to empty
        procedure_graph.update_state(config, {
            "step_index": 0,
            "steps": [],
            "user_input": None,
            "api_output": None,
            "final_response": None,
            "answers": []
        })
        
        logger.info(f"[Orchestrator] Reset procedure for thread {thread_id}")
        return {"status": "success", "message": "Procedure reset"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thread/{thread_id}/status")
async def get_thread_status(thread_id: str, tenant_id: str = Header(..., alias="X-Tenant-Id")):
    """Get current status of a thread"""
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": tenant_id
            }
        }
        
        procedure_graph = get_procedure_graph(tenant_id, "Ai_agent")
        snapshot = procedure_graph.get_state(config)
        
        if not snapshot or not snapshot.values:
            return {
                "thread_id": thread_id,
                "has_active_procedure": False,
                "status": "no_active_procedure"
            }
        
        values = snapshot.values
        is_waiting = bool(snapshot.tasks and snapshot.tasks[0].interrupts)
        
        return {
            "thread_id": thread_id,
            "has_active_procedure": True,
            "current_step": values.get("step_index", 0),
            "total_steps": len(values.get("steps", [])),
            "waiting_for_input": is_waiting,
            "answers_collected": len(values.get("answers", []))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)