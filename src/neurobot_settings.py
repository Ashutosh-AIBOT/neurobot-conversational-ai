import json
import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    default_tenant_id: str
    default_mcp_server_name: str
    groq_api_key: str | None
    langchain_api_key: str | None
    langchain_project: str
    model_name: str
    model_temperature: float
    max_pdf_size_mb: int
    max_prompt_chars: int
    rate_limit_requests: int
    rate_limit_period_seconds: int
    low_confidence_threshold: float
    auto_eval_responses: bool
    mcp_servers: dict[str, dict]
    runtime_dir: Path
    vector_store_dir: Path
    checkpoint_path: Path
    benchmark_dir: Path

    def ensure_runtime_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
        cleaned = "-".join(filter(None, cleaned.split("-")))
        return cleaned or "default"

    def tenant_dir(self, tenant_id: str | None = None) -> Path:
        return self.runtime_dir / "tenants" / self._slug(tenant_id or self.default_tenant_id)

    def tenant_vector_dir(self, tenant_id: str | None = None) -> Path:
        return self.tenant_dir(tenant_id) / "vector_store"

    def tenant_checkpoint_path(self, tenant_id: str | None = None) -> Path:
        return self.tenant_dir(tenant_id) / "neurobot.db"

    def session_namespace(self, tenant_id: str | None, session_id: str) -> str:
        tenant = self._slug(tenant_id or self.default_tenant_id)
        session = self._slug(session_id)
        return f"{tenant}:{session}"


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    base_dir = Path(__file__).resolve().parents[1]
    runtime_dir = base_dir / "runtime"
    vector_store_dir = runtime_dir / "vector_store"
    checkpoint_path = runtime_dir / "neurobot.db"
    benchmark_dir = base_dir / "data" / "benchmark"
    raw_mcp_servers = os.getenv("MCP_SERVERS_JSON", "{}").strip() or "{}"
    try:
        mcp_servers = json.loads(raw_mcp_servers)
    except json.JSONDecodeError:
        mcp_servers = {}

    _SETTINGS = Settings(
        app_env=os.getenv("APP_ENV", "development"),
        default_tenant_id=os.getenv("DEFAULT_TENANT_ID", "default"),
        default_mcp_server_name=os.getenv("DEFAULT_MCP_SERVER_NAME", "neurobot"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "NeuroBot"),
        model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        max_pdf_size_mb=int(os.getenv("MAX_PDF_SIZE_MB", "15")),
        max_prompt_chars=int(os.getenv("MAX_PROMPT_CHARS", "4000")),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "15")),
        rate_limit_period_seconds=int(os.getenv("RATE_LIMIT_PERIOD_SECONDS", "60")),
        low_confidence_threshold=float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6")),
        auto_eval_responses=_env_bool("AUTO_EVAL_RESPONSES", True),
        mcp_servers=mcp_servers if isinstance(mcp_servers, dict) else {},
        runtime_dir=runtime_dir,
        vector_store_dir=vector_store_dir,
        checkpoint_path=checkpoint_path,
        benchmark_dir=benchmark_dir,
    )
    _SETTINGS.ensure_runtime_dirs()
    return _SETTINGS
