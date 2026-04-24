import logging

from langchain_core.tools import tool

from src.neurobot_mcp import call_mcp_tool
from src.neurobot_settings import get_settings


logger = logging.getLogger("NeuroBotTools")
settings = get_settings()


def _default_mcp_server() -> str:
    return settings.default_mcp_server_name


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web for general information, current events, or news."""
    try:
        return call_mcp_tool(_default_mcp_server(), "duckduckgo_search", {"query": query})
    except Exception as e:
        logger.error("Search error: %s", e)
        return f"Search error: {str(e)}"


@tool
def arxiv_search(query: str) -> str:
    """Search scientific and academic papers on Arxiv for technical details."""
    try:
        return call_mcp_tool(_default_mcp_server(), "arxiv_search", {"query": query})
    except Exception as e:
        logger.error("Arxiv Error: %s", e)
        return f"Arxiv service unreachable or error occurred: {str(e)}"


@tool
def download_and_talk_to_paper(paper_id: str, thread_id: str) -> str:
    """Downloads a specific Arxiv paper by its ID and indexes it for question answering."""
    try:
        return call_mcp_tool(
            _default_mcp_server(),
            "download_and_talk_to_paper",
            {"paper_id": paper_id, "thread_id": thread_id},
        )
    except Exception as e:
        return f"Error downloading/indexing paper: {str(e)}"


@tool
def pdf_qa_tool(query: str, thread_id: str) -> str:
    """Queries the content of the uploaded PDF for specific answers."""
    try:
        return call_mcp_tool(
            _default_mcp_server(),
            "pdf_qa_tool",
            {"query": query, "thread_id": thread_id},
        )
    except Exception as e:
        return f"PDF Retrieval Error: {str(e)}"


@tool
def evaluate_response(query: str, answer: str, context: str) -> str:
    """Runs a groundedness audit on the current answer when enough context is available."""
    return call_mcp_tool(
        _default_mcp_server(),
        "evaluate_response",
        {"query": query, "answer": answer, "context": context},
    )


@tool
def mcp_tool_call(server_name: str, method: str, arguments: dict) -> str:
    """Invokes a configured MCP server tool."""
    try:
        logger.info("MCP Call: %s -> %s(%s)", server_name, method, arguments)
        return call_mcp_tool(server_name=server_name, tool_name=method, arguments=arguments or {})
    except Exception as e:
        return f"MCP Tool Error: {str(e)}"


def get_neuro_tools():
    """Returns the list of tools available to the NeuroBot agent."""
    return [arxiv_search, download_and_talk_to_paper, pdf_qa_tool, duckduckgo_search, evaluate_response, mcp_tool_call]
