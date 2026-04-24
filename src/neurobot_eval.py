import os
import logging
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from datasets import Dataset
from langchain_groq import ChatGroq
from src.neurobot_rag import get_embeddings
from src.neurobot_settings import get_settings

logger = logging.getLogger("NeuroBotEval")

# --- LANGSMITH SETUP ---
def setup_tracing():
    """Configures LangSmith tracing if API key is present."""
    settings = get_settings()
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith Tracing Enabled")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith Tracing Disabled (No API Key)")

# --- GLOBAL EVAL LLM ---
_eval_llm = None

def get_evaluation_model():
    global _eval_llm
    if _eval_llm is None:
        settings = get_settings()
        if not settings.groq_api_key:
            return None
        _eval_llm = ChatGroq(
            model_name=settings.model_name,
            groq_api_key=settings.groq_api_key,
            temperature=0
        )
    return _eval_llm

def run_evaluation(query: str, answer: str, context: str) -> str:
    """Runs a lightweight response audit when enough context exists."""
    try:
        if not context or "error" in context.lower() or len(context) < 50:
            return (
                "### Response Audit\n"
                "- Status: skipped\n"
                "- Reason: not enough grounded context was available"
            )

        eval_llm = get_evaluation_model()
        if not eval_llm:
            return (
                "### Response Audit\n"
                "- Status: unavailable\n"
                "- Reason: evaluation model is not configured"
            )

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [[context]],
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=get_embeddings()
        )
        df = result.to_pandas()
        faith = float(df["faithfulness"].iloc[0])
        relevancy = float(df["answer_relevancy"].iloc[0])

        if faith >= 0.8:
            status = "strongly grounded"
        elif faith >= 0.6:
            status = "partially grounded"
        else:
            status = "needs manual review"

        return (
            "### Response Audit\n"
            f"- Status: {status}\n"
            f"- Faithfulness: {faith:.2f}\n"
            f"- Answer relevancy: {relevancy:.2f}\n"
            "- Notes: this is a groundedness proxy, not a substitute for human verification"
        )
    except Exception as e:
        logger.error(f"Evaluation Failed: {e}")
        return f"### Response Audit\n- Status: failed\n- Reason: {str(e)}"

# Initialize Tracing on Load
setup_tracing()
