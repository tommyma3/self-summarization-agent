from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_chat_template(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.is_file() and path.startswith("src/self_summarization_agent/"):
        candidate = Path(__file__).resolve().parent / path.removeprefix("src/self_summarization_agent/")
    if not candidate.is_file():
        raise FileNotFoundError(f"Chat template not found: {candidate}")
    return candidate.read_text(encoding="utf-8")


def configure_tokenizer_chat_template(tokenizer: Any, path: str | None) -> str | None:
    template = load_chat_template(path)
    if template is not None:
        tokenizer.chat_template = template
    return template
