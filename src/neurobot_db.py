import logging
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from src.neurobot_settings import get_settings

logger = logging.getLogger("NeuroBotDB")

def get_checkpointer(db_path=None, tenant_id=None):
    """Returns a SQLite checkpointer for conversation persistence."""
    settings = get_settings()
    target_path = str(db_path or settings.tenant_checkpoint_path(tenant_id))
    try:
        settings.ensure_runtime_dirs()
        conn = sqlite3.connect(target_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:
        logger.error(f"FATAL: Database Initialization Error: {e}")
        # Fallback to in-memory if disk fails
        try:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            return SqliteSaver(conn)
        except:
            return None
