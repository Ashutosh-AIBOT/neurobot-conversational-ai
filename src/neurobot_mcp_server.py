from mcp.server.fastmcp import FastMCP

from src.neurobot_tool_impl import (
    arxiv_search as local_arxiv_search,
    download_and_talk_to_paper as local_download_and_talk_to_paper,
    evaluate_response as local_evaluate_response,
    pdf_qa_tool as local_pdf_qa_tool,
    duckduckgo_search as local_duckduckgo_search,
)


mcp = FastMCP("NeuroBot MCP", json_response=True)


@mcp.tool()
def duckduckgo_search(query: str) -> str:
    """Search the web for general information, current events, or news."""
    return local_duckduckgo_search(query)


@mcp.tool()
def arxiv_search(query: str) -> str:
    """Search scientific and academic papers on Arxiv for technical details."""
    return local_arxiv_search(query)


@mcp.tool()
def download_and_talk_to_paper(paper_id: str, thread_id: str) -> str:
    """Download an arXiv paper and index it for question answering."""
    return local_download_and_talk_to_paper(paper_id, thread_id)


@mcp.tool()
def pdf_qa_tool(query: str, thread_id: str) -> str:
    """Query the content of the uploaded PDF for specific answers."""
    return local_pdf_qa_tool(query, thread_id)


@mcp.tool()
def evaluate_response(query: str, answer: str, context: str) -> str:
    """Run a groundedness audit on the current answer."""
    return local_evaluate_response(query, answer, context)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
