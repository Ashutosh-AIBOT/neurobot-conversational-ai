import sqlite3
import logging
import os
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger("NeuroBotDB")

def get_checkpointer(db_path="neurobot.db"):
    """Returns a SQLite checkpointer for conversation persistence."""
    try:
        # Ensure the directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:
        logger.error(f"FATAL: Database Initialization Error: {e}")
        # Fallback to in-memory if disk fails
        try:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            return SqliteSaver(conn)
        except:
            return None
