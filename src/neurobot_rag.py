import json
import os
import tempfile
import logging
import shutil
from pathlib import Path
from typing import Any, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.neurobot_settings import get_settings

logger = logging.getLogger("NeuroBotRAG")
settings = get_settings()

# Global cache to store model and retrievers
_EMBEDDINGS: Any = None
_RETRIEVERS: Dict[str, Any] = {}
_METADATA: Dict[str, dict] = {}

def _thread_dir(thread_id: str) -> Path:
    return settings.tenant_vector_dir(thread_id)

def _metadata_path(thread_id: str) -> Path:
    return _thread_dir(thread_id) / "metadata.json"

def get_embeddings():
    """Uses a free local HuggingFace model for embeddings (cached)."""
    global _EMBEDDINGS
    try:
        if _EMBEDDINGS is None:
            logger.info("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
            _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDINGS
    except Exception as e:
        logger.error(f"Embeddings Load Failed: {e}")
        raise

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """Processes a PDF, chunks it, and stores it in an in-memory FAISS vector store."""
    path = None
    try:
        settings.ensure_runtime_dirs()
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_bytes)
            path = f.name
        
        # Load and Split
        loader = PyPDFLoader(path)
        docs = loader.load()
        
        if not docs:
            return {"error": "The PDF file appears to be empty or unreadable."}
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create Vector Store
        vector_store = FAISS.from_documents(chunks, get_embeddings())
        thread_dir = _thread_dir(thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(thread_dir))
        
        # Cache Retriever
        _RETRIEVERS[str(thread_id)] = vector_store.as_retriever(search_kwargs={"k": 4})
        _METADATA[str(thread_id)] = {
            "filename": filename,
            "pages": len(docs),
            "chunks": len(chunks),
            "status": "success",
        }
        _metadata_path(thread_id).write_text(
            json.dumps(_METADATA[str(thread_id)], indent=2),
            encoding="utf-8",
        )
        
        logger.info(f"Successfully indexed {filename} for thread {thread_id}")
        return _METADATA[str(thread_id)]
        
    except Exception as e:
        logger.error(f"Ingestion Error for {filename}: {e}")
        return {"error": f"Failed to process PDF: {str(e)}"}
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

def get_retriever(thread_id: str):
    """Retrieves the retriever instance for a specific thread."""
    cached = _RETRIEVERS.get(str(thread_id))
    if cached:
        return cached

    thread_dir = _thread_dir(thread_id)
    if not thread_dir.exists():
        return None

    try:
        vector_store = FAISS.load_local(
            str(thread_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        _RETRIEVERS[str(thread_id)] = retriever
        return retriever
    except Exception as e:
        logger.error("Failed to reload vector store for thread %s: %s", thread_id, e)
        return None

def load_metadata(thread_id: str) -> dict:
    path = _metadata_path(thread_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to read metadata for thread %s: %s", thread_id, e)
    return {}

def clear_runtime_state():
    """Removes cached retrievers and local runtime artifacts created by the app."""
    _RETRIEVERS.clear()
    _METADATA.clear()

    if settings.runtime_dir.exists():
        shutil.rmtree(settings.runtime_dir, ignore_errors=True)
    settings.ensure_runtime_dirs()
    logger.info("Runtime state cleared.")

def get_doc_metadata(thread_id: str):
    """Retrieves metadata for the document associated with a thread."""
    if str(thread_id) in _METADATA:
        return _METADATA[str(thread_id)]

    persisted = load_metadata(thread_id)
    if persisted:
        _METADATA[str(thread_id)] = persisted
        return persisted

    return {}
