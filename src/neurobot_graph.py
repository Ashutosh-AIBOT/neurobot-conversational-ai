import os, logging, time
from typing import Annotated, TypedDict, List, Optional
from collections import deque
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from src.neurobot_tools import get_neuro_tools
from src.neurobot_db import get_checkpointer
from src.neurobot_eval import setup_tracing

# Initialize tracing early
setup_tracing()

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuroBotGraph")

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---
class NeuroResponse(BaseModel):
    """Structured response from NeuroBot."""
    answer: str = Field(description="The main text of the response.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0.")
    sources: List[str] = Field(default=[], description="List of document or web sources used.")
    suggested_questions: List[str] = Field(default=[], description="Follow-up questions for the user.")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    is_relevant: Optional[bool]
    structured_response: Optional[NeuroResponse]

# Rate Limiter
class RateLimiter:
    def __init__(self, max_calls=15, period=60):
        self.calls = deque()
        self.max_calls = max_calls
        self.period = period
    def is_allowed(self):
        now = time.time()
        while self.calls and self.calls[0] < now - self.period: self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True, None
        return False, int(self.period - (now - self.calls[0]))

rate_limiter = RateLimiter()

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return None
    return ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.1)

def agent_node(state: AgentState, config: dict = None):
    try:
        allowed, wait = rate_limiter.is_allowed()
        if not allowed:
            return {"messages": [AIMessage(content=f"⚠️ Rate Limit: {wait}s.")]}
        
        llm = get_llm()
        if not llm: return {"messages": [AIMessage(content="⚠️ Config Error.")]}

        tools = get_neuro_tools()
        llm_with_tools = llm.bind_tools(tools)
        
        tid = config["configurable"].get("thread_id", "default") if config else "default"
        
        # System prompt for the agent
        sys_msg = (
            "You are NeuroBot Pro v2.1, a high-performance cognitive research engine.\n"
            "Your objective is to provide absolute factual accuracy using provided sources.\n\n"
            "OPERATIONAL PROTOCOLS:\n"
            "1. INGESTION: Use `pdf_qa_tool` for primary knowledge retrieval from uploaded files.\n"
            "2. EXPLORATION: Use `arxiv_search` for peer-reviewed scientific context.\n"
            "3. VERIFICATION: Use `duckduckgo_search` for real-time validation of facts.\n"
            "4. AUDITING: Always call `evaluate_response` to show the accuracy dashboard to the user.\n"
            "5. TONE: Professional, analytical, and data-centric.\n\n"
            "Constraint: If confidence is < 70%, explicitly mention the uncertainty.\n"
            "SESSION_ID: " + tid
        )
        
        response = llm_with_tools.invoke([SystemMessage(content=sys_msg)] + state["messages"])
        
        # If no tool calls, it's a final response - apply Pydantic structuring
        if not response.tool_calls:
            try:
                structured_llm = llm.with_structured_output(NeuroResponse)
                struct_resp = structured_llm.invoke(
                    [SystemMessage(content="Format the following information into the required schema.")] + 
                    state["messages"] + [response]
                )
                response.content = (
                    f"{struct_resp.answer}\n\n"
                    f"**Confidence:** {int(struct_resp.confidence * 100)}%  \n"
                    f"**Sources:** {', '.join(struct_resp.sources) if struct_resp.sources else 'General Knowledge'}"
                )
                return {"messages": [response], "structured_response": struct_resp}
            except Exception as e:
                logger.warning(f"Structured output failed, falling back to raw: {e}")
                return {"messages": [response]}

        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent Node Error: {e}")
        return {"messages": [AIMessage(content=f"⚠️ Brain Error: {str(e)}")]}

def grader_node(state: AgentState):
    """Corrective RAG: Uses the LLM to grade document relevance."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or last_msg.name != "pdf_qa_tool":
        return state

    llm = get_llm()
    if not llm: return state

    # Define a simple grader prompt
    grader_sys = "You are a grader assessing relevance of a retrieved document to a user question. Respond with 'yes' if relevant, 'no' if not."
    user_query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    grade_resp = llm.invoke([
        SystemMessage(content=grader_sys),
        HumanMessage(content=f"Question: {user_query}\n\nRetrieved Context: {last_msg.content}")
    ])
    
    is_relevant = "yes" in grade_resp.content.lower()
    
    if not is_relevant:
        logger.info("CRAG: LLM graded context as IRRELEVANT. Triggering web search.")
        return {
            "messages": [AIMessage(content="The document context wasn't sufficient. Expanding search to the web...")],
            "is_relevant": False
        }
    
    logger.info("CRAG: LLM graded context as RELEVANT.")
    return {"is_relevant": True}

def reflector_node(state: AgentState):
    """Self-reflection node: checks if the generated response needs improvement."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        return state
        
    # If confidence is too low, we might want to re-reason
    if state.get("structured_response") and state["structured_response"].confidence < 0.6:
        logger.info("Self-Reflection: Confidence low. Re-triggering search.")
        return {"messages": [AIMessage(content="My internal confidence is low. I will attempt to find more precise data...")]}
        
    return state

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(get_neuro_tools()))
workflow.add_node("grader", grader_node)
workflow.add_node("reflector", reflector_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", 
    tools_condition
)
workflow.add_edge("tools", "grader")
workflow.add_edge("grader", "agent")
workflow.add_edge("agent", "reflector")
workflow.add_edge("reflector", END)

# Compile with persistence
checkpointer = get_checkpointer()
neurobot_brain = workflow.compile(checkpointer=checkpointer)
