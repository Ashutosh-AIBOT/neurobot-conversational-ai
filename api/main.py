from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.neurobot_quality import format_quality_report
from src.neurobot_service import chat_turn, ingest_document
from src.neurobot_settings import get_settings
from src.neurobot_validation import validate_pdf_upload, validate_user_prompt

settings = get_settings()
app = FastAPI(title="NeuroBot Service API", version="1.0.0")


class ChatRequest(BaseModel):
    tenant_id: str = Field(default="default")
    session_id: str = Field(default="default")
    message: str


class ChatResponse(BaseModel):
    answer: str
    tenant_id: str
    session_id: str
    status: str
    chunks_indexed: int = 0
    latency_ms: float = 0.0
    audit_status: str = "not_run"
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    source_count: int = 0
    tool_count: int = 0
    quality_report: str = ""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tenant_mode": True,
        "default_tenant_id": settings.default_tenant_id,
        "model": settings.model_name,
        "benchmark_cases": len((settings.benchmark_dir / "mini_benchmark.jsonl").read_text(encoding="utf-8").splitlines())
        if (settings.benchmark_dir / "mini_benchmark.jsonl").exists()
        else 0,
    }


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    error = validate_user_prompt(request.message, settings.max_prompt_chars)
    if error:
        raise HTTPException(status_code=400, detail=error)

    result = chat_turn(
        request.tenant_id,
        request.session_id,
        [HumanMessage(content=request.message)],
    )
    return ChatResponse(
        answer=result["answer"],
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        status="ok",
        chunks_indexed=int(result["metadata"].get("chunks", 0)),
        latency_ms=float(result["quality_report"]["latency_ms"]),
        audit_status=str(result["quality_report"]["audit_status"]),
        faithfulness=result["quality_report"]["faithfulness"],
        answer_relevancy=result["quality_report"]["answer_relevancy"],
        source_count=int(result["quality_report"]["source_count"]),
        tool_count=int(result["quality_report"]["tool_count"]),
        quality_report=format_quality_report(result["quality_report"]),
    )


@app.post("/v1/documents")
async def documents(
    tenant_id: str = Form(default="default"),
    session_id: str = Form(default="default"),
    file: UploadFile = File(...),
):
    error = validate_pdf_upload(file.filename or "", None, settings.max_pdf_size_mb)
    if error:
        raise HTTPException(status_code=400, detail=error)

    contents = await file.read()
    if len(contents) > settings.max_pdf_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF is too large. Maximum supported size is {settings.max_pdf_size_mb} MB.")

    result = ingest_document(contents, tenant_id, session_id, file.filename or "document.pdf")
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"status": "indexed", "tenant_id": tenant_id, "session_id": session_id, **result}
