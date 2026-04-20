
import os
import pandas as pd
import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_groq import ChatGroq
from src.neurobot_rag import get_embeddings
from langsmith import Client

logger = logging.getLogger("NeuroBotEval")

# --- LANGSMITH SETUP ---
def setup_tracing():
    """Configures LangSmith tracing if API key is present."""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "NeuroBot-AI-v2"
        logger.info("LangSmith Tracing Enabled")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith Tracing Disabled (No API Key)")

# --- GLOBAL EVAL LLM ---
_eval_llm = None

def get_eval_llm():
    global _eval_llm
    if _eval_llm is None:
        _eval_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    return _eval_llm

def run_evaluation(query: str, answer: str, context: str) -> str:
    """Evaluates RAG quality using Groq, HuggingFace, and Ragas metrics."""
    try:
        if not context or "error" in context.lower() or len(context) < 50:
            return "### 📊 Groq Accuracy Dashboard\n- **Status:** Evaluation skipped (Insufficient context)\n- **Recommendation:** Upload a document for more accurate analysis."

        data = {
            "question": [query], 
            "answer": [answer], 
            "contexts": [[context]], 
            "ground_truth": [answer]
        }
        dataset = Dataset.from_dict(data)
        
        # Execute Evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=get_eval_llm(),
            embeddings=get_embeddings()
        )
        
        df = result.to_pandas()
        faith = df['faithfulness'].iloc[0]
        relevancy = df['answer_relevancy'].iloc[0]
        precision = df['context_precision'].iloc[0]
        
        # Determine Status & Hallucination Reduction
        hallucination_reduction = int(faith * 100)
        status = "🟢 Optimized" if faith > 0.8 else "🟡 Hallucination Risk"
        if faith < 0.5: status = "🔴 Inaccurate"

        return (
            f"### 📊 Groq Accuracy Dashboard\n"
            f"- **System Status:** {status}\n"
            f"- **Hallucination Reduction:** {hallucination_reduction}%\n"
            f"- **Faithfulness Score:** {faith:.2f}\n"
            f"- **Answer Relevancy:** {int(relevancy * 100)}%\n"
            f"- **Context Precision:** {int(precision * 100)}%\n"
            f"--- \n"
            f"*Metrics audited via Llama-3.3-70B and Ragas Engine*"
        )
    except Exception as e:
        logger.error(f"Evaluation Failed: {e}")
        return f"⚠️ **Auditor Error:** {str(e)}"

# Initialize Tracing on Load
setup_tracing()
