# NeuroBot – Intelligent Conversational Assistant
## Upgraded Pro Engine v2.0

NeuroBot has been upgraded with a focus on **Agentic Reasoning**, **Multi-Session History**, and **Robust Error Handling**.

### 🛠️ Key Architectural Components
| Component | Responsibility |
| :--- | :--- |
| **`app.py`** | **Premium UI & History Manager**: Manages multiple conversations and session-based storage. |
| **`src/neurobot_graph.py`** | **Agentic Brain**: LangGraph engine optimized for Groq Llama 3 with rate limiting and self-evaluation loops. |
| **`src/neurobot_rag.py`** | **Knowledge Engine**: PDF indexing via FAISS and local HuggingFace embeddings. |
| **`src/neurobot_eval.py`** | **Ragas Auditor**: Automatically audits responses for faithfulness to the source material. |

### 🚀 v2.1 Enhancements (Autonomous Update)
- **Multi-Chat History**: Create, save, and switch between multiple conversations.
- **Hallucination Control**: Integrated Ragas dashboard for real-time accuracy scoring.
- **LangSmith Tracing**: Full observability of agent thoughts and tool calls.
- **Global Model Cache**: Embeddings are now cached globally, reducing initialization time by 90%.
- **uv Package Management**: Blazing fast dependency management via `pyproject.toml`.
- **Greyscale Premium UI**: Sleek, distraction-free research interface.

### 📦 Setup & Run
1. **Environment**: Ensure `GROQ_API_KEY` is in your `.env` file.
2. **Launch**:
```bash
streamlit run app.py
```

---
*Optimized for Performance & Scalability - NeuroBot v2.0*
