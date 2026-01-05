import os
import logging
import time
from typing import TypedDict, Annotated, Sequence, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from actions import create_tool_retriever
from mongodb_utilies_actions import MONGODB_ATLAS_URI
from get_the_persona_tool import create_persona_tool
from knowledge_base_retriever_tool import create_retriever

logger = logging.getLogger(__name__)

# ============================================================================
# CHAT GRAPH STATE
# ============================================================================

class ChatState(TypedDict):
    """State for chat graph - clean and focused"""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    session_id: str
    tenant_id: str
    persona_data: Optional[dict]
    procedure_triggered: bool  # Flag if a procedure tool was called

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
# SYSTEM PROMPT
# ============================================================================

CHAT_SYSTEM_PROMPT = """You are a customer support assistant.
At the start of every new session, you must always call the persona tool (`get_persona_tool`) to fetch persona details.

Persona usage rules:
1. Use **greetingMessage** only once, at the beginning of the session.
2. Use **name** and **description** only when explicitly needed in responses.
3. For the **first session message**, follow persona values: tone, messageLength, allowEmoji.
4. For **subsequent messages**, override as: tone → "friendly", allowEmoji → true.
5. Never invent persona details; always rely on the persona tool output.

Message Length Guidelines:
- **Concise**: 1-2 sentences maximum
- **Medium**: Necessary important details
- **Normal**: Complete explanation in 3-5 sentences

Conversation Guidelines:
1. Always search the provided tools for answers.
2. Only use information explicitly present in the supplied tools.
3. If answer not found, respond: "I'm sorry, but I couldn't find any information related to your question. Could you please rephrase it or let me know if there's something else I can assist you with?"
4. Do not generate information beyond what is in the tools.
5. Maintain professionalism and politeness.
6. If a document has a valid URL, append a 'Sources' section at the end.

Multi-step Procedures:
- If you need to execute a multi-step procedure (order cancellation, booking modification, etc.),
  call the appropriate procedure tool (e.g., cancel_order_tool).
- The tool will return procedure data that will be handled by a specialized procedure graph.
- Do NOT try to handle the procedure steps yourself - just call the tool.
"""

# ============================================================================
# CHAT GRAPH NODES
# ============================================================================

def fetch_persona_node(state: ChatState, persona_tool) -> ChatState:
    """Fetch persona details at the start of session"""
    logger.info(f"[ChatGraph] Fetching persona for tenant: {state['tenant_id']}")
    
    try:
        persona_result = persona_tool.invoke({})
        state['persona_data'] = persona_result
        
        system_msg = SystemMessage(content=CHAT_SYSTEM_PROMPT)
        state['messages'].insert(0, system_msg)
        
        logger.info(f"[ChatGraph] Persona fetched successfully")
    except Exception as e:
        logger.error(f"[ChatGraph] Error fetching persona: {e}")
        state['persona_data'] = {}
    
    return state

def agent_node(state: ChatState, llm_with_tools) -> ChatState:
    """Main agent node that processes messages and calls tools"""
    logger.info(f"[ChatGraph] Processing message for session: {state['session_id']}")
    
    response = llm_with_tools.invoke(state['messages'])
    state['messages'].append(response)
    
    return state

def tool_node(state: ChatState, tools) -> ChatState:
    """Execute tools and return results"""
    logger.info(f"[ChatGraph] Executing tools")
    
    last_message = state['messages'][-1]
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        tool_result = None
        for tool in tools:
            if tool.name == tool_name:
                tool_result = tool.invoke(tool_args)
                
                # Check if this is a procedure tool
                if isinstance(tool_result, dict) and tool_result.get('is_procedure'):
                    logger.info(f"[ChatGraph] Procedure tool called: {tool_name}")
                    state['procedure_triggered'] = True
                    
                    # Store procedure initiation message
                    # The actual procedure will be handled by procedure graph on next request
                    tool_result = {
                        "message": "I'll help you with that. Let me start the process.",
                        "procedure_name": tool_result.get('procedure_name'),
                        "is_procedure": True
                    }
                
                break
        
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call['id']
        )
        state['messages'].append(tool_message)
    
    return state

# ============================================================================
# ROUTING
# ============================================================================

def should_continue(state: ChatState) -> str:
    """Router function to determine next step"""
    last_message = state['messages'][-1]
    
    # Check if tools were called
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"

# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_chat_graph(tenant_id: str, retriever_mode: str, mode: str):
    """Create the chat graph for conversational RAG"""
    start_time = time.perf_counter()
    
    # Create tools
    persona_tool = create_persona_tool(tenant_id)
    retriever_tool = create_retriever(tenant_id, retriever_mode)
    actions = create_tool_retriever(tenant_id)
    tools = actions + [persona_tool, retriever_tool]
    
    logger.info(f"[ChatGraph] Tools created: {len(tools)} tools")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("fetch_persona", lambda state: fetch_persona_node(state, persona_tool))
    workflow.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    workflow.add_node("tools", lambda state: tool_node(state, tools))
    
    # Set entry point
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
    
    # Choose checkpointer
    if mode == "Ai_agent":
        checkpointer = MongoDBSaver(
            client=mongo_client,
            db_name="rag_bot",
            collection_name="chat_checkpoints"
        )
    else:
        checkpointer = MemorySaver()
    
    # Compile graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info(f"[ChatGraph] Graph created in {time.perf_counter() - start_time:.3f}s")
    
    return graph