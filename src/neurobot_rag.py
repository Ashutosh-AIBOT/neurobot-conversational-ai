import os
import tempfile
import logging
from typing import Any, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("NeuroBotRAG")

# Global cache to store model and retrievers
_EMBEDDINGS: Any = None
_RETRIEVERS: Dict[str, Any] = {}
_METADATA: Dict[str, dict] = {}

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
        
        # Cache Retriever
        _RETRIEVERS[str(thread_id)] = vector_store.as_retriever(search_kwargs={"k": 4})
        _METADATA[str(thread_id)] = {
            "filename": filename, 
            "pages": len(docs), 
            "chunks": len(chunks),
            "status": "success"
        }
        
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
    return _RETRIEVERS.get(str(thread_id))

def get_doc_metadata(thread_id: str):
    """Retrieves metadata for the document associated with a thread."""
    return _METADATA.get(str(thread_id), {})
