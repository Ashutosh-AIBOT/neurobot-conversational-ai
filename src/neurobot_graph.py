import os, logging, time
from typing import Annotated, TypedDict, List, Optional
from collections import deque
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.neurobot_tools import duckduckgo_search, get_neuro_tools
from src.neurobot_db import get_checkpointer
from src.neurobot_eval import run_evaluation, setup_tracing
from src.neurobot_settings import get_settings

# Initialize tracing early
setup_tracing()
settings = get_settings()

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
    needs_web_recovery: Optional[bool]
    recovery_attempted: Optional[bool]

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

rate_limiter = RateLimiter(
    max_calls=settings.rate_limit_requests,
    period=settings.rate_limit_period_seconds,
)

def get_chat_model():
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key: return None
    return ChatGroq(
        model_name=settings.model_name,
        groq_api_key=api_key,
        temperature=settings.model_temperature,
    )

def _latest_user_query(state: AgentState) -> str:
    return next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

def _latest_context(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name in {
            "pdf_qa_tool",
            "duckduckgo_search",
            "arxiv_search",
            "download_and_talk_to_paper",
            "evaluate_response",
        }:
            return message.content
    return ""

def agent_node(state: AgentState, config: dict = None):
    try:
        allowed, wait = rate_limiter.is_allowed()
        if not allowed:
            return {"messages": [AIMessage(content=f"Request limit reached. Try again in {wait} seconds.")]}
        
        llm = get_chat_model()
        if not llm: return {"messages": [AIMessage(content="Model configuration is incomplete.")]}

        tools = get_neuro_tools()
        llm_with_tools = llm.bind_tools(tools)
        
        tid = config["configurable"].get("thread_id", "default") if config else "default"
        
        # System prompt for the agent
        sys_msg = (
            "You are NeuroBot, a document-aware research assistant.\n"
            "Use retrieved evidence when available and be explicit about uncertainty.\n\n"
            "OPERATIONAL PROTOCOLS:\n"
            "1. INGESTION: Use `pdf_qa_tool` for primary knowledge retrieval from uploaded files.\n"
            "2. EXPLORATION: Use `arxiv_search` for peer-reviewed scientific context.\n"
            "3. VERIFICATION: Use `duckduckgo_search` for real-time validation of facts.\n"
            "4. MCP: These tools are MCP-backed; call them normally and trust their returned context.\n"
            "5. CITATIONS: Prefer source-backed answers over broad unsupported claims.\n"
            "6. TONE: Professional, analytical, and concise.\n\n"
            "Constraint: If confidence is low, explicitly mention the uncertainty.\n"
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
                state_update = {
                    "messages": [response],
                    "structured_response": struct_resp,
                    "needs_web_recovery": struct_resp.confidence < settings.low_confidence_threshold,
                }

                if settings.auto_eval_responses:
                    audit = run_evaluation(
                        query=_latest_user_query(state),
                        answer=struct_resp.answer,
                        context=_latest_context(state),
                    )
                    state_update["messages"].append(
                        ToolMessage(
                            content=audit,
                            name="response_audit",
                            tool_call_id="response_audit",
                        )
                    )

                return state_update
            except Exception as e:
                logger.warning(f"Structured output failed, falling back to raw: {e}")
                return {"messages": [response], "needs_web_recovery": False}

        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent Node Error: {e}")
        return {"messages": [AIMessage(content=f"Assistant error: {str(e)}")]}

def grader_node(state: AgentState):
    """Corrective RAG: Uses the LLM to grade document relevance."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, ToolMessage) or last_msg.name != "pdf_qa_tool":
        return state

    llm = get_chat_model()
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
    """Self-reflection node: checks if recovery is needed but doesn't mark it attempted yet."""
    # This node is a pass-through that ensures state flags are consistent
    if state.get("needs_web_recovery") and not state.get("recovery_attempted"):
        logger.info("Self-Reflection: low confidence detected, routing to recovery search.")
    return state

def web_recovery_node(state: AgentState):
    """Fallback search path when retrieved document context is weak or answer confidence is low."""
    query = _latest_user_query(state)
    if not query:
        return {"needs_web_recovery": False}

    logger.info("Web recovery triggered for query: %s", query)
    content = duckduckgo_search.invoke({"query": query})
    return {
        "messages": [
            ToolMessage(
                content=content,
                name="duckduckgo_search",
                tool_call_id="web_recovery_search",
            )
        ],
        "needs_web_recovery": False,
        "recovery_attempted": True,
    }

def route_after_agent(state: AgentState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"
    return "reflector"

def route_after_grader(state: AgentState):
    return "web_recovery" if state.get("is_relevant") is False else "agent"

def route_after_reflector(state: AgentState):
    if state.get("needs_web_recovery") and not state.get("recovery_attempted"):
        return "web_recovery"
    return END

# Build Graph
def compile_brain(tenant_id: str | None = None):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(get_neuro_tools()))
    workflow.add_node("grader", grader_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("web_recovery", web_recovery_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "reflector": "reflector"},
    )
    workflow.add_edge("tools", "grader")
    workflow.add_conditional_edges(
        "grader",
        route_after_grader,
        {"agent": "agent", "web_recovery": "web_recovery"},
    )
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {"web_recovery": "web_recovery", END: END},
    )
    workflow.add_edge("web_recovery", "agent")

    checkpointer = get_checkpointer(tenant_id=tenant_id)
    return workflow.compile(
        checkpointer=checkpointer,
        recursion_limit=15,
    )


neurobot_brain = compile_brain(settings.default_tenant_id)
