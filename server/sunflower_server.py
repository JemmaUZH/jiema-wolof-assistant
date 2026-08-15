import json
import os
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.environ.get("SUNFLOWER_MODEL_PATH", str(ROOT / "models" / "Sunflower-Gemma4-E2B"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

app = FastAPI(title="Jiema Sunflower Local Server")
processor = None
model = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool | None = False
    format: str | None = None
    options: dict[str, Any] | None = None


def load_model():
    global processor, model
    if model is not None:
        return

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(DEVICE)


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_json_text(text: str) -> str:
    stripped = (
        text.replace("<end_of_turn>", "")
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return json.dumps(
        {
            "wolof": stripped or "Jàppandi na. Jéemaatal.",
            "english": "The local model did not return strict JSON.",
            "category": "general",
            "safety": "Manual review required."
        },
        ensure_ascii=False,
    )


@app.on_event("startup")
def startup():
    load_model()


@app.get("/api/tags")
def tags():
    return {"models": [{"name": "Sunflower-Gemma4-E2B", "model": "Sunflower-Gemma4-E2B"}]}


@app.post("/api/chat")
def chat(request: ChatRequest):
    load_model()

    messages = [
        {"role": message.role, "content": normalize_content(message.content)}
        for message in request.messages
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        text_kwargs={"padding": False, "truncation": True, "add_special_tokens": False},
    ).to(DEVICE)

    max_new_tokens = int((request.options or {}).get("num_predict", 180))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_response = processor.decode(new_tokens, skip_special_tokens=False)
    try:
        content = processor.parse_response(raw_response)["content"]
    except Exception:
        content = raw_response

    return {
        "model": request.model or "Sunflower-Gemma4-E2B",
        "message": {
            "role": "assistant",
            "content": extract_json_text(content),
        },
        "done": True,
    }


if __name__ == "__main__":
    port = int(os.environ.get("SUNFLOWER_PORT", "11434"))
    uvicorn.run(app, host="127.0.0.1", port=port)
