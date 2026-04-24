import re
from pathlib import Path


PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+message\b", re.IGNORECASE),
    re.compile(r"\breveal\b.*\binstructions\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+to\s+be\b", re.IGNORECASE),
)


def validate_pdf_upload(filename: str, file_size: int | None, max_size_mb: int) -> str | None:
    if not filename:
        return "Uploaded file must have a name."

    if Path(filename).suffix.lower() != ".pdf":
        return "Only PDF uploads are supported."

    if file_size is not None and file_size > max_size_mb * 1024 * 1024:
        return f"PDF is too large. Maximum supported size is {max_size_mb} MB."

    return None


def validate_user_prompt(prompt: str, max_chars: int) -> str | None:
    cleaned = prompt.strip()
    if not cleaned:
        return "Prompt cannot be empty."

    if len(cleaned) > max_chars:
        return f"Prompt is too long. Maximum supported length is {max_chars} characters."

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return (
                "Prompt looks unsafe because it appears to override system behavior. "
                "Please rephrase the request as a normal user question."
            )

    return None
