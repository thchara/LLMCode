from __future__ import annotations
import json
import os
import datetime
import pathlib
import threading

_lock = threading.Lock()

LOG_DIR_DEFAULT = pathlib.Path("logs")


def log_prompt(
    prompt: str,
    response: str,
    *,
    model: str,
    temperature: float,
    system_prompt: str | None,
) -> None:
    if os.getenv("LLMCODE_NO_LOG"):
        return
    log_dir = pathlib.Path(os.getenv("LLMCODE_LOG_DIR", LOG_DIR_DEFAULT))
    log_dir.mkdir(parents=True, exist_ok=True)

    fname = f"llmcode_{datetime.date.today():%Y-%m-%d}.jsonl"
    record = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "response": response,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(log_dir / fname, "a", encoding="utf-8") as f:
            f.write(line + "\n")
