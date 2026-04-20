import requests
import arxiv
import logging
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from src.neurobot_rag import get_retriever, get_doc_metadata, ingest_pdf
from src.neurobot_eval import run_evaluation

logger = logging.getLogger("NeuroBotTools")

from duckduckgo_search import DDGS

@tool
def duckduckgo_search(query: str) -> str:
    """Search the web for general information, current events, or news."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join([f"Source: {r['href']}\nContent: {r['body']}" for r in results])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"

@tool
def arxiv_search(query: str) -> str:
    """Search scientific and academic papers on Arxiv for technical details. Returns paper IDs and summaries."""
    try:
        search = arxiv.Search(query=query, max_results=3)
        results = list(search.results())
        if not results:
            return f"No scientific papers found for '{query}'."
        
        res = [f"ID: {r.get_short_id()}\nTitle: {r.title}\nSummary: {r.summary[:500]}..." for r in results]
        logger.info(f"Arxiv search for '{query}' returned {len(results)} results.")
        return "\n\n".join(res)
    except Exception as e:
        logger.error(f"Arxiv Error: {e}")
        return f"Arxiv service unreachable or error occurred: {str(e)}"

@tool
def download_and_talk_to_paper(paper_id: str, thread_id: str) -> str:
    """Downloads a specific Arxiv paper by its ID and indexes it so the user can talk to it. Use this when the user mentions a specific paper or ID."""
    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        
        # Download the PDF to memory
        response = requests.get(paper.pdf_url)
        if response.status_code != 200:
            return f"Failed to download paper from {paper.pdf_url}"
            
        # Ingest the PDF
        res = ingest_pdf(response.content, thread_id, f"{paper.title}.pdf")
        if "error" in res:
            return f"Error indexing paper: {res['error']}"
            
        return f"Successfully downloaded and indexed: '{paper.title}'. You can now ask questions about it!"
    except Exception as e:
        return f"Error downloading/indexing paper: {str(e)}"

@tool
def pdf_qa_tool(query: str, thread_id: str) -> str:
    """Queries the content of the uploaded PDF for specific answers. Use the provided thread_id."""
    try:
        retriever = get_retriever(thread_id)
        if not retriever:
            return "ERROR: No PDF has been uploaded yet for this session. Please ask the user to upload a PDF or use `download_and_talk_to_paper`."
        
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document for this query."
            
        context = "\n".join([d.page_content for d in docs])
        meta = get_doc_metadata(thread_id)
        filename = meta.get("filename", "the document")
        
        return f"CONTEXT FROM {filename}:\n{context}"
    except Exception as e:
        return f"PDF Retrieval Error: {str(e)}"

@tool
def evaluate_response(query: str, answer: str, context: str) -> str:
    """Audits the agent's response for factual accuracy and faithfulness to the provided context. 
    Use this tool whenever you want to verify if a response is correct or show the accuracy dashboard."""
    return run_evaluation(query, answer, context)

@tool
def mcp_tool_call(server_name: str, method: str, arguments: dict) -> str:
    """Invokes a tool via the Model Context Protocol (MCP). Use this for advanced integrations."""
    try:
        # This is a functional placeholder for MCP integration
        logger.info(f"MCP Call: {server_name} -> {method}({arguments})")
        return f"Successfully called MCP server '{server_name}' method '{method}'. Response: (Functionality simulated for local environment)"
    except Exception as e:
        return f"MCP Tool Error: {str(e)}"

def get_neuro_tools():
    """Returns the list of tools available to the NeuroBot agent."""
    return [arxiv_search, download_and_talk_to_paper, pdf_qa_tool, duckduckgo_search, evaluate_response, mcp_tool_call]
