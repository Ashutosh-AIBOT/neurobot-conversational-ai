import logging

import arxiv
import requests
from duckduckgo_search import DDGS

from src.neurobot_eval import run_evaluation
from src.neurobot_rag import get_doc_metadata, get_retriever, ingest_pdf


logger = logging.getLogger("NeuroBotToolImpl")


def duckduckgo_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join([f"Source: {r['href']}\nContent: {r['body']}" for r in results])
    except Exception as e:
        logger.error("Search error: %s", e)
        return f"Search error: {str(e)}"


def arxiv_search(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results=3)
        results = list(search.results())
        if not results:
            return f"No scientific papers found for '{query}'."

        res = [
            f"ID: {r.get_short_id()}\nTitle: {r.title}\nSummary: {r.summary[:500]}..."
            for r in results
        ]
        logger.info("Arxiv search for '%s' returned %s results.", query, len(results))
        return "\n\n".join(res)
    except Exception as e:
        logger.error("Arxiv Error: %s", e)
        return f"Arxiv service unreachable or error occurred: {str(e)}"


def download_and_talk_to_paper(paper_id: str, thread_id: str) -> str:
    try:
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())

        response = requests.get(paper.pdf_url)
        if response.status_code != 200:
            return f"Failed to download paper from {paper.pdf_url}"

        res = ingest_pdf(response.content, thread_id, f"{paper.title}.pdf")
        if "error" in res:
            return f"Error indexing paper: {res['error']}"

        return f"Successfully downloaded and indexed: '{paper.title}'. You can now ask questions about it!"
    except Exception as e:
        return f"Error downloading/indexing paper: {str(e)}"


def pdf_qa_tool(query: str, thread_id: str) -> str:
    try:
        retriever = get_retriever(thread_id)
        if not retriever:
            return (
                "ERROR: No PDF has been uploaded yet for this session. "
                "Please ask the user to upload a PDF or use `download_and_talk_to_paper`."
            )

        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document for this query."

        context = "\n".join([d.page_content for d in docs])
        meta = get_doc_metadata(thread_id)
        filename = meta.get("filename", "the document")
        return f"CONTEXT FROM {filename}:\n{context}"
    except Exception as e:
        return f"PDF Retrieval Error: {str(e)}"


def evaluate_response(query: str, answer: str, context: str) -> str:
    return run_evaluation(query, answer, context)
