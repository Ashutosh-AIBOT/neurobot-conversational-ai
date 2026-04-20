---
title: NeuroBot Intelligent Conversational Assistant
emoji: 🤖
colorFrom: brown
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# NeuroBot – Intelligent Conversational Assistant
## Upgraded Pro Engine v2.1 (Neural Excellence)

NeuroBot is a state-of-the-art Agentic RAG assistant built with **LangGraph**, **RAGAS**, and **LangSmith**. It features a specialized "Self-Reflection" loop to minimize hallucinations and provide peer-reviewed scientific accuracy.

### 🌟 Key Enhancements
- **Brown/Bronze Premium Dashboard**: A sophisticated, research-grade interface.
- **Hallucination Control**: Integrated RAGAS auditing to show real-time accuracy metrics.
- **Cognitive Reasoning**: Multi-stage graph thoughts with self-correction nodes.
- **ArXiv & Web Intelligence**: Instant discovery and indexing of technical papers.
- **uv Integration**: High-performance package management for lightning-fast deployments.

### 🚀 Local Quick Start
```bash
# Install with uv
uv pip install -e .

# Launch the terminal
streamlit run app.py
```

---

# 🤖 NeuroBot — Intelligent Conversational AI Assistant (Legacy Archive)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/Built_with-LangGraph-purple?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=flat-square)
![OpenRouter](https://img.shields.io/badge/API-OpenRouter-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Stage-Live-brightgreen?style=flat-square)

---

🌐 **Live Demo:** [neurobot-intelligent-conversational-si37.onrender.com](https://neurobot-intelligent-conversational-si37.onrender.com/)
👤 **Author:** [Ashutosh — GitHub](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/)
💼 **Portfolio:** [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/)

---

## 📋 Table of Contents

- [What This Does](#-what-this-does)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [What I Built](#-what-i-built)
- [LangGraph State Machine](#-langgraph-state-machine)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [Tech Stack](#-tech-stack)
- [Project Status](#-project-status)
- [Links](#-links)
- [Author](#-author)

---

## 🧠 What This Does

NeuroBot is a production-grade conversational AI assistant
built with LangGraph state machines, OpenRouter API
multi-model routing, and clean OOP Python architecture.

1. **Problem** — Most chatbot demos are single-turn,
   stateless, and break on multi-step conversations.
   They have no memory and no tool-calling ability.
2. **Solution** — LangGraph state machine manages full
   conversation memory across turns. OpenRouter gives
   access to multiple LLMs. Tool calling extends
   the assistant beyond just text responses.
3. **For** — GenAI / LLM Engineer hiring managers
   looking for real LangGraph + agentic AI proof

---

## 🌐 Live Demo

👉 **[neurobot-intelligent-conversational-si37.onrender.com](https://neurobot-intelligent-conversational-si37.onrender.com/)**

> Chat with NeuroBot directly in the browser.
> Multi-turn memory active — it remembers the full conversation.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🧠 Multi-Turn Memory | Full conversation history via LangGraph state |
| 🔀 Multi-Model Routing | OpenRouter API — switch between LLMs |
| 🛠️ Tool Calling | Extend responses beyond pure text |
| 🏗️ OOP Architecture | Clean, modular, production-ready Python code |
| 💬 Streamlit UI | Clean chat interface with message history |
| 🚀 Live Deployment | Hosted on Render — always available |

---

## 🏗️ Architecture
```
User Message (Streamlit UI)
        ↓
LangGraph State Machine
  → Maintains full conversation state
  → Manages message history across turns
  → Routes to appropriate node
        ↓
Nodes
  → chat_node: main LLM response generation
  → tool_node: external tool execution
  → memory_node: state update and persistence
        ↓
OpenRouter API
  → Multi-model access (GPT, Claude, Mistral, etc.)
  → Model selection based on query type
  → Streaming response support
        ↓
Tool Calling Layer
  → Web search tool
  → Calculator tool
  → Custom function tools
        ↓
Response Streamed Back
  → Streamlit chat message display
  → Conversation history updated in state
        ↓
Deployed on Render
  → Always-on live URL
```

---

## 🔨 What I Built

### 1. LangGraph State Machine
- Defined full conversation graph with typed state
- Nodes: chat, tool execution, memory update
- Edges: conditional routing based on response type
- State persists full message history across all turns
- Handles tool call detection and routing automatically

### 2. OpenRouter API Integration
- Connected to OpenRouter for multi-model LLM access
- Supports GPT-4, Claude, Mistral, Llama via single API
- Model can be swapped without changing core logic
- Handles API key management and error retries

### 3. Tool Calling Pipeline
- Defined custom tools as Python functions
- LLM decides when to call tools vs respond directly
- Tool results injected back into conversation state
- Supports chained tool calls in single turn

### 4. OOP Architecture
- `NeuroBot` class encapsulates full bot logic
- `StateManager` handles LangGraph state operations
- `ToolRegistry` manages all available tools
- `ModelRouter` handles OpenRouter API calls
- Clean separation of concerns — easy to extend

### 5. Streamlit Chat Interface
- Clean chat bubble UI with user and assistant styling
- Full message history displayed in session
- Model selector in sidebar
- Typing indicator during response generation
- Error handling with user-friendly messages

### 6. Deployment on Render
- Configured `render.yaml` for one-click deploy
- Environment variables managed via Render dashboard
- Auto-deploy on push to main branch
- Always-on service — no cold start on free tier

---

## 🔀 LangGraph State Machine
```
START
  ↓
[chat_node] ← Main LLM response
  ↓
Tool call detected?
  ├── YES → [tool_node] → Execute tool → Back to chat_node
  └── NO  → [memory_node] → Update state
              ↓
           [END] → Stream response to UI
```

---

## ⚡ Quick Start

**Prerequisites:** Python 3.11+, Git, OpenRouter API key
```bash
# 1. Clone the repo
git clone https://github.com/Ashutosh-AIBOT/neurobot-conversational-ai.git
cd neurobot-conversational-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your OpenRouter API key to .env

# 5. Run the app
streamlit run app.py

# 6. Open browser
# http://localhost:8501
```

---

## 🔑 Environment Variables

| Variable | What It Is | Where To Get |
|----------|-----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key | [openrouter.ai](https://openrouter.ai/) |
| `MODEL_NAME` | Default LLM model | e.g. `openai/gpt-4o` |
| `APP_ENV` | Environment flag | `development` or `production` |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| LangGraph | Conversation state machine |
| LangChain | LLM framework and tool calling |
| OpenRouter API | Multi-model LLM access |
| Streamlit | Chat UI and deployment interface |
| Render | Live hosting and deployment |
| Git | Version control |

---

## 📁 Repository Structure
```
neurobot-conversational-ai/
│
├── app.py                          # Main Streamlit app entry point
│
├── neurobot/
│   ├── __init__.py
│   ├── bot.py                      # Main NeuroBot class
│   ├── state.py                    # LangGraph state definition
│   ├── nodes.py                    # Graph nodes (chat, tool, memory)
│   ├── tools.py                    # Tool definitions and registry
│   └── router.py                   # OpenRouter API handler
│
├── .env.example                    # Environment variable template
├── requirements.txt
├── render.yaml                     # Render deployment config
└── README.md
```

---

## 📊 Project Status

| Deliverable | Status |
|-------------|--------|
| LangGraph State Machine | ✅ Complete |
| OpenRouter API Integration | ✅ Complete |
| Tool Calling Pipeline | ✅ Complete |
| OOP Architecture | ✅ Complete |
| Streamlit Chat UI | ✅ Complete |
| Live Deployment (Render) | ✅ Live |
| Multi-model Switching | ✅ Complete |
| Streaming Responses | 🔄 In Progress |

---

## 🌐 Links

| Resource | URL |
|----------|-----|
| 🚀 Live Demo | [neurobot-intelligent-conversational-si37.onrender.com](https://neurobot-intelligent-conversational-si37.onrender.com/) |
| 💼 Portfolio | [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/) |
| 🐙 GitHub | [github.com/Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT) |
| 🔗 LinkedIn | [linkedin.com/in/ashutosh1975271](https://www.linkedin.com/in/ashutosh1975271/) |

---

## 👤 Author

**Ashutosh**
B.Tech Electronics Engineering · CGPA 7.5 · Batch 2026
[GitHub](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/) · [Portfolio](https://ashutosh-portfolio-kappa.vercel.app/)

---

> *"Not just a chatbot.*
> *A state machine that thinks, remembers, and acts."*
>
> — Ashutosh, building this from zero.
