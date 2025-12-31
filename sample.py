import os
import uuid
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional
from datetime import datetime
import dotenv

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver

# Import your existing tools
from actions import create_tool_retriever
from mongodb_utilies_actions import MONGODB_ATLAS_URI
from get_the_persona_tool import create_persona_tool
from knowledge_base_retriever_tool import create_retriever

dotenv.load_dotenv()

# Configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2025-03-01-preview"

if not all([AZURE_API_KEY, AZURE_ENDPOINT, DEPLOYMENT_NAME]):
    raise ValueError("❌ Missing required environment variables")

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION,
    temperature=0.7,
)
print("✅ AzureChatOpenAI initialized")

from pymongo import MongoClient

mongo_client = MongoClient(MONGODB_ATLAS_URI)

# FastAPI App
app = FastAPI(title="LangGraph Customer Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    session_id: str
    tenant_id: str
    persona_data: Optional[dict]

# ============================================================================
# PYDANTIC MODELS FOR API
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
    message_count: int

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a customer support assistant.
At the start of every new session, you must always call the persona tool (`get_persona_tool`) to fetch persona details.

Persona usage rules:
1. Use **greetingMessage** only once, at the beginning of the session.
2. Use **name** and **description** only when explicitly needed in responses (e.g., when the customer asks who you are).
3. For the **first session message**, follow the values of:
   - tone
   - messageLength
   - allowEmoji
   as returned by the persona tool.
4. For **subsequent conversation messages**, override persona values as follows:
   - tone → "friendly"
   - allowEmoji → true
   - messageLength → keep following the persona tool value.
5. Never invent persona details; always rely on the persona tool output.

6. If the messageLength is **Concise**, give the answer in the smallest possible, short and sweet way maximum one or two sentence only.
If it is **Medium**, give necessary important details required for an understandable explanation.
If it is **Normal**, give all important details required for a complete explanation and answer only in 3-5 sentences.

Conversation Guidelines:
1. Always search the provided tools for an answer to the customer's question.
2. Only use information explicitly present in the supplied tools.
   - If the answer is not in the tools, respond with:
     "I'm sorry, but I couldn't find any information related to your question. Could you please rephrase it or let me know if there's something else I can assist you with?"
3. Do not generate information, assumptions, or opinions beyond what is in the tools.
4. Maintain professionalism and politeness in all responses, aligned with customer support standards.
5. Adapt your communication style to the customer's tone and mood, while respecting persona tone and emoji rules:
   - Be professional when the customer is formal.
   - Be friendly when the customer is casual.
   - Be empathetic when the customer expresses frustration or concern.
6. Ensure responses feel natural, consistent with the company's brand voice, and optimized for a smooth customer experience.
7. If a document has a valid URL, append a 'Sources' section at the end of your answer, with each link formatted in Markdown: [Link Text](URL).
   - If the URL is 'N/A' or missing, do not print it.
"""

# ============================================================================
# GRAPH NODES
# ============================================================================

def fetch_persona_node(state: AgentState, persona_tool) -> AgentState:
    """Fetch persona details using the persona tool at the start of session"""
    logging.info(f"[fetch_persona_node] Fetching persona for tenant: {state['tenant_id']}")
    start_time = time.perf_counter()
    
    try:
        # Call persona tool (it has its own fallback logic)
        persona_result = persona_tool.invoke({})
        state['persona_data'] = persona_result
        
        # Add system message with persona context
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        state['messages'].insert(0, system_msg)
        
        logging.info(f"[fetch_persona_node] Persona fetched in {time.perf_counter() - start_time:.3f}s")
        logging.info(f"[fetch_persona_node] Persona data: {persona_result}")
    except Exception as e:
        logging.error(f"[fetch_persona_node] Error fetching persona: {e}")
        state['persona_data'] = {}
    
    return state

def agent_node(state: AgentState, llm_with_tools) -> AgentState:
    """Main agent node that processes messages and calls tools"""
    logging.info(f"[agent_node] Processing message for session: {state['session_id']}")
    start_time = time.perf_counter()
    
    # Invoke the LLM with tools
    response = llm_with_tools.invoke(state['messages'])
    
    # Add AI response to messages
    state['messages'].append(response)
    
    logging.info(f"[agent_node] Response generated in {time.perf_counter() - start_time:.3f}s")
    
    return state

def tool_node(state: AgentState, tools) -> AgentState:
    """Execute tools and return results"""
    logging.info(f"[tool_node] Executing tools")
    
    # Get the last message which should contain tool calls
    last_message = state['messages'][-1]
    
    # Execute each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        # Find and execute the tool
        tool_result = None
        for tool in tools:
            if tool.name == tool_name:
                tool_result = tool.invoke(tool_args)
                break
        
        # Add tool result as a message
        from langchain_core.messages import ToolMessage
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call['id']
        )
        state['messages'].append(tool_message)
    
    return state

def should_continue(state: AgentState) -> str:
    """Router function to determine next step based on tool calls"""
    last_message = state['messages'][-1]
    
    # If the last message has tool calls, continue to tools node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"

# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_agent_graph(tenant_id: str, retriever_mode: str, mode: str):
    """Create the LangGraph agent graph"""
    start_time = time.perf_counter()
    
    # Create tools
    persona_tool = create_persona_tool(tenant_id)
    retriever_tool = create_retriever(tenant_id, retriever_mode)
    actions = create_tool_retriever(tenant_id)
    tools = actions + [persona_tool, retriever_tool]
    
    logging.info(f"[create_agent_graph] Tools created in {time.perf_counter() - start_time:.3f}s")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fetch_persona", lambda state: fetch_persona_node(state, persona_tool))
    workflow.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    workflow.add_node("tools", lambda state: tool_node(state, tools))
    
    # Set entry point - start with fetching persona
    workflow.set_entry_point("fetch_persona")
    
    # Add edges
    workflow.add_edge("fetch_persona", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    # Choose checkpointer based on mode
    if mode == "Ai_agent":
        checkpointer = MongoDBSaver(
            client=mongo_client,
            db_name="rag_bot",
            collection_name="checkpoints"
        )
    else:
        checkpointer = MemorySaver()
    
    # Compile graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    logging.info(f"[create_agent_graph] Graph created in {time.perf_counter() - start_time:.3f}s")
    
    return graph

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LangGraph Customer Support Agent"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    start_time = time.perf_counter()
    
    try:
        logging.info(f"[chat] Received request for session: {request.session_id}")
        
        # Initialize MongoDB chat message history
        chat_history = MongoDBChatMessageHistory(
            session_id=request.session_id,
            connection_string=MONGODB_ATLAS_URI,
            database_name="test_rag_bot",
            collection_name="message_history"
        )
        
        # Get historical messages
        historical_messages = chat_history.messages
        logging.info(f"[chat] Loaded {len(historical_messages)} historical messages")
        
        # Create agent graph
        graph = create_agent_graph(
            tenant_id=request.tenant_id,
            retriever_mode=request.retriever_mode,
            mode=request.mode
        )
        
        # Prepare initial state with historical messages
        initial_state = {
            "messages": historical_messages + [HumanMessage(content=request.message)],
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
            "persona_data": None
        }
        
        # Configure thread
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }
        
        # Invoke graph
        result = graph.invoke(initial_state, config=config)
        
        # Extract response
        last_message = result['messages'][-1]
        response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Save new messages to MongoDB (only human and AI messages)
        # Add the human message
        chat_history.add_user_message(request.message)
        
        # Add the AI response
        chat_history.add_ai_message(response_text)
        
        logging.info(f"[chat] Saved messages to MongoDB history")
        
        execution_time = time.perf_counter() - start_time
        logging.info(f"[chat] Request completed in {execution_time:.3f}s")
        
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            execution_time=execution_time,
            message_count=len(result['messages'])
        )
        
    except Exception as e:
        logging.error(f"[chat] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")



# ============================================================================
# MAIN
# ============================================================================