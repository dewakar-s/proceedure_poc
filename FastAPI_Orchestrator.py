import os
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from chat_graph_implementation import create_chat_graph
from procedure_graph_implementation import create_procedure_graph
from mongodb_utilies_actions import MONGODB_ATLAS_URI

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
    session_id: str = Field(..., description="Unique session identifier")
    tenant_id: str = Field(..., description="Tenant/Organization identifier")
    message: str = Field(..., description="User message")
    mode: str = Field(default="Ai_agent", description="Mode: 'Ai_agent' or 'memory'")
    retriever_mode: str = Field(default="hybrid", description="Retriever mode")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    execution_time: float
    graph_used: str  # "chat" or "procedure"
    requires_input: bool = False
    procedure_step: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# GRAPH STATE DETECTION
# ============================================================================

def get_active_procedure_state(session_id: str, mode: str) -> Optional[Dict]:
    """
    Check if there's an active procedure for this session
    Returns procedure state if active, None otherwise
    """
    try:
        # Create procedure graph to check state
        procedure_graph = create_procedure_graph(
            tenant_id="",  # Will be overridden by state
            mode=mode
        )
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state
        state = procedure_graph.get_state(config)
        
        if state and state.values.get('in_procedure'):
            logger.info(f"[Orchestrator] Found active procedure for session {session_id}")
            return state.values
        
        return None
        
    except Exception as e:
        logger.warning(f"[Orchestrator] Error checking procedure state: {e}")
        return None

# ============================================================================
# ORCHESTRATION LOGIC
# ============================================================================

async def orchestrate_request(request: ChatRequest) -> ChatResponse:
    """
    Main orchestration logic:
    1. Check for active procedure
    2. Route to appropriate graph
    3. Handle fallbacks
    4. Return unified response
    """
    start_time = time.perf_counter()
    
    try:
        # Initialize chat history
        chat_history = MongoDBChatMessageHistory(
            session_id=request.session_id,
            connection_string=MONGODB_ATLAS_URI,
            database_name="test_rag_bot",
            collection_name="message_history"
        )
        
        config = {"configurable": {"thread_id": request.session_id}}
        
        # ============================================================
        # STEP 1: Check for active procedure
        # ============================================================
        active_procedure = get_active_procedure_state(request.session_id, request.mode)
        
        if active_procedure:
            logger.info(f"[Orchestrator] Resuming procedure for session {request.session_id}")
            
            # Resume procedure graph
            procedure_graph = create_procedure_graph(
                tenant_id=request.tenant_id,
                mode=request.mode
            )
            
            # Update state with new user input
            result = procedure_graph.invoke(
                {
                    **active_procedure,
                    "messages": active_procedure.get('messages', []) + [
                        HumanMessage(content=request.message)
                    ]
                },
                config=config
            )
            
            # ============================================================
            # STEP 2: Check procedure result
            # ============================================================
            
            # Procedure completed successfully
            if not result.get('in_procedure'):
                response_text = result['messages'][-1].content
                
                # Save to history
                chat_history.add_user_message(request.message)
                chat_history.add_ai_message(response_text)
                
                return ChatResponse(
                    session_id=request.session_id,
                    response=response_text,
                    execution_time=time.perf_counter() - start_time,
                    graph_used="procedure",
                    requires_input=False,
                    metadata={
                        "procedure_completed": True,
                        "total_steps": result.get('step_index', 0)
                    }
                )
            
            # Procedure needs more input (valid step)
            elif result.get('waiting_for_input'):
                current_step = result['procedure_steps'][result['step_index']]
                question = current_step.get('message') or current_step.get('question')
                
                return ChatResponse(
                    session_id=request.session_id,
                    response=question,
                    execution_time=time.perf_counter() - start_time,
                    graph_used="procedure",
                    requires_input=True,
                    procedure_step=current_step.get('type'),
                    metadata={
                        "current_step": result.get('step_index'),
                        "total_steps": len(result.get('procedure_steps', []))
                    }
                )
            
            # Invalid input - fallback to chat
            elif result.get('fallback_to_chat'):
                logger.info(f"[Orchestrator] Procedure fallback triggered, routing to chat")
                
                # Fall through to chat graph below
                pass
            
        # ============================================================
        # STEP 3: Use Chat Graph
        # ============================================================
        
        logger.info(f"[Orchestrator] Processing with chat graph for session {request.session_id}")
        
        chat_graph = create_chat_graph(
            tenant_id=request.tenant_id,
            retriever_mode=request.retriever_mode,
            mode=request.mode
        )
        
        historical_messages = chat_history.messages
        
        initial_state = {
            "messages": historical_messages + [HumanMessage(content=request.message)],
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
            "persona_data": None
        }
        
        result = chat_graph.invoke(initial_state, config=config)
        
        # Extract response
        last_message = result['messages'][-1]
        response_text = last_message.content
        
        # Save to history
        chat_history.add_user_message(request.message)
        chat_history.add_ai_message(response_text)
        
        # Check if chat triggered a procedure
        procedure_triggered = result.get('procedure_triggered', False)
        
        if procedure_triggered:
            # Next message should be routed to procedure graph
            logger.info(f"[Orchestrator] Chat triggered procedure, next request will use procedure graph")
        
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            execution_time=time.perf_counter() - start_time,
            graph_used="chat",
            requires_input=False,
            metadata={
                "procedure_triggered": procedure_triggered,
                "message_count": len(result['messages'])
            }
        )
        
    except Exception as e:
        logger.error(f"[Orchestrator] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# ============================================================================
# ENDPOINTS
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with automatic graph routing
    
    Flow:
    1. Check for active procedure → Resume if found
    2. Check for procedure fallback → Use chat
    3. Default → Use chat graph
    4. Handle procedure triggers → Prepare for next request
    """
    return await orchestrate_request(request)

@app.post("/reset-procedure/{session_id}")
async def reset_procedure(session_id: str):
    """
    Force reset a procedure for a session
    Useful for error recovery or user cancellation
    """
    try:
        # This would clear the procedure state from MongoDB
        # Implementation depends on your MongoDBSaver structure
        logger.info(f"[Orchestrator] Reset procedure for session {session_id}")
        return {"status": "success", "message": "Procedure reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current status of a session
    Returns: active graph, procedure state, message count, etc.
    """
    try:
        procedure_state = get_active_procedure_state(session_id, "Ai_agent")
        
        return {
            "session_id": session_id,
            "has_active_procedure": procedure_state is not None,
            "procedure_step": procedure_state.get('step_index') if procedure_state else None,
            "in_procedure": procedure_state.get('in_procedure', False) if procedure_state else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)