import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.neurobot_logging import setup_logging
from src.neurobot_quality import build_quality_report, format_quality_report
from src.neurobot_rag import clear_runtime_state, get_doc_metadata, ingest_pdf
from src.neurobot_service import get_brain
from src.neurobot_settings import get_settings
from src.neurobot_validation import validate_pdf_upload, validate_user_prompt

load_dotenv()
setup_logging()
settings = get_settings()

# --- SESSION INITIALIZATION ---
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True

# --- CONFIG ---
st.set_page_config(page_title="NeuroBot", page_icon="🤖", layout="wide")

# Custom CSS for Premium Dashboard Look (Pure Black / OLED)
st.markdown("""
    <style>
    /* Global Pure Black Background */
    .stApp, .main, .block-container, header, .stHeader, [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Pure Black */
    [data-testid="stSidebar"], .stSidebar {
        background-color: #050505 !important;
        border-right: 1px solid #1f1f1f !important;
    }
    
    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        background-color: #000000 !important;
        border-radius: 10px !important;
    }
    
    /* Stats Dashboard Styling (Bronze/Gold accents) */
    .stats-card {
        background: #111111;
        border: 1px solid #c5a059;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }
    .metric-value { color: #c5a059; font-size: 1.6rem; font-weight: 900; }
    .metric-label { color: #888888; font-size: 0.8rem; text-transform: uppercase; }
    
    /* Message bubbles - Clearly Visible */
    .stChatMessage { 
        background-color: #0a0a0a !important; 
        border: 1px solid #1a1a1a !important; 
        border-radius: 12px !important; 
        padding: 15px !important;
        margin-bottom: 20px !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* User message specific highlight */
    div[data-testid="stChatMessage"]:has(div[data-testid="user-avatar"]) {
        border-left: 4px solid #c5a059 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_thread_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_id
    st.session_state.conversations[new_id] = {"name": "New Chat", "history": [], "quality_report": None}
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = settings.default_tenant_id

def create_chat_session():
    new_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_id
    st.session_state.conversations[new_id] = {
        "name": f"Chat {len(st.session_state.conversations) + 1}",
        "history": [],
        "quality_report": None,
    }
    st.rerun()

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("🤖 NeuroBot Pro")
    st.caption("Document Research Assistant")
    st.session_state.tenant_id = st.text_input("Workspace ID", value=st.session_state.tenant_id)
    
    # HLD Graph Expander
    with st.expander("🏗️ Architecture Overview", expanded=False):
        st.markdown("""
        ```mermaid
        graph TD
            A[User] --> B[Agentic Brain]
            B --> C{Reflection}
            C -->|Verified| D[Response]
            C -->|Uncertain| E[Tools]
            E --> B
            subgraph Tools
                F[FAISS PDF]
                G[ArXiv Search]
                H[Web Verification]
            end
        ```
        """, unsafe_allow_html=True)
        st.caption("LangGraph, retrieval, and search workflow")
    
    st.divider()
    if st.button("➕ New Session", use_container_width=True, type="primary"):
        create_chat_session()

    st.divider()
    for tid, data in st.session_state.conversations.items():
        label = f"💬 {data['name']}"
        if tid == st.session_state.current_thread_id: label = f"🚀 **{data['name']}**"
        if st.button(label, key=tid, use_container_width=True):
            st.session_state.current_thread_id = tid
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear All Sessions", use_container_width=True, type="secondary"):
        clear_runtime_state()
        st.session_state.conversations = {}
        new_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_id
        st.session_state.conversations[new_id] = {"name": "New Chat", "history": [], "quality_report": None}
        st.rerun()

    st.divider()
    st.subheader("📁 Knowledge Source")
    pdf_file = st.file_uploader("Upload PDF Paper", type="pdf")
    if pdf_file:
        upload_validation_error = validate_pdf_upload(
            filename=pdf_file.name,
            file_size=pdf_file.size,
            max_size_mb=settings.max_pdf_size_mb,
        )
        if upload_validation_error:
            st.error(upload_validation_error)
        else:
            with st.status("Indexing document...", expanded=False) as s:
                session_key = settings.session_namespace(st.session_state.tenant_id, st.session_state.current_thread_id)
                res = ingest_pdf(pdf_file.getvalue(), session_key, pdf_file.name)
                if "error" in res:
                    st.error(res["error"])
                    s.update(label="Indexing failed", state="error")
                else:
                    st.success(f"Indexed document: {res['filename']} ({res['chunks']} chunks)")
                    s.update(label="Indexing complete", state="complete")
    
    st.divider()
    st.subheader("📊 Session Overview")
    session_key = settings.session_namespace(st.session_state.tenant_id, st.session_state.current_thread_id)
    current_meta = get_doc_metadata(session_key)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Open Sessions", str(len(st.session_state.conversations)))
    with col2:
        st.metric("Indexed Chunks", str(current_meta.get("chunks", 0)))

    current_quality = st.session_state.conversations[st.session_state.current_thread_id].get("quality_report")
    if current_quality:
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Latency (ms)", f"{current_quality['latency_ms']:.0f}")
        with col4:
            st.metric("Source Count", str(current_quality["source_count"]))
    
    if current_meta:
        st.caption(
            f"Current document: {current_meta.get('filename', 'N/A')} | "
            f"Pages: {current_meta.get('pages', 0)}"
        )

    tracing_status = "enabled" if settings.langchain_api_key else "disabled"
    st.info(
        f"Model: {settings.model_name}\n\n"
        f"Tracing: {tracing_status}\n\n"
        f"Auto-evaluation: {'on' if settings.auto_eval_responses else 'off'}"
    )
    st.caption("Metrics appear only when they are computed from the active session.")

# --- MAIN CHAT ---
active_chat = st.session_state.conversations[st.session_state.current_thread_id]
st.title(f"{active_chat['name']}")
st.info("Assistant mode: retrieval, search recovery, and response audit when grounded context is available")

# Display history
for msg in active_chat["history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if "### Response Audit" in msg.content or "### Quality Report" in msg.content:
            st.markdown(f'<div class="stats-card">{msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg.content)

# User input
if prompt := st.chat_input("Ask about a document, arXiv topic, or web topic..."):
    prompt_validation_error = validate_user_prompt(prompt, max_chars=settings.max_prompt_chars)
    if prompt_validation_error:
        st.error(prompt_validation_error)
        st.stop()

    active_chat["history"].append(HumanMessage(content=prompt))
    if active_chat["name"] == "New Chat": active_chat["name"] = prompt[:25] + "..."
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = ""
        placeholder = st.empty()
        tool_events = []
        started = time.perf_counter()
        
        try:
            brain = get_brain(st.session_state.tenant_id)
            config = {
                "configurable": {
                    "thread_id": settings.session_namespace(
                        st.session_state.tenant_id,
                        st.session_state.current_thread_id,
                    )
                }
            }
            for chunk in brain.stream({"messages": active_chat["history"]}, config=config, stream_mode="messages"):
                msg = chunk[0] if isinstance(chunk, tuple) else chunk
                
                if isinstance(msg, AIMessage) and msg.content:
                    response_text += str(msg.content)
                    placeholder.markdown(response_text + "▌")
                
                if isinstance(msg, ToolMessage):
                    tool_events.append(msg)
                    with st.expander(f"Tool activity: {msg.name}"):
                        if "Response Audit" in msg.content:
                            st.info("Running grounded response audit...")
                        st.write(msg.content)
                        
        except Exception as e:
            st.error(f"Assistant error: {e}")
            response_text = "I hit a snag. Please check your connection."

        placeholder.markdown(response_text)
        active_chat["history"].append(AIMessage(content=response_text))
        quality_report = build_quality_report(
            answer_text=response_text,
            tool_events=tool_events,
            latency_ms=(time.perf_counter() - started) * 1000,
            indexed_chunks=int(current_meta.get("chunks", 0)),
        )
        active_chat["quality_report"] = quality_report
        quality_markdown = format_quality_report(quality_report)
        active_chat["history"].append(AIMessage(content=quality_markdown))
        st.markdown(f'<div class="stats-card">{quality_markdown}</div>', unsafe_allow_html=True)
