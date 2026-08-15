import base64
import gc
import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)

try:
    from transformers import VitsModel
except Exception:  # pragma: no cover - depends on installed transformers build
    VitsModel = None


ASR_MODEL_ID = os.environ.get("JIEMA_ASR_MODEL_ID", "sokho2/mms-300m-wolof")
LLM_MODEL_ID = os.environ.get(
    "JIEMA_LLM_MODEL_ID", "google/gemma-4-E2B-it-qat-mobile-transformers"
)
TTS_MODEL_ID = os.environ.get("JIEMA_TTS_MODEL_ID", "facebook/mms-tts-wlx")
DEVICE = os.environ.get("JIEMA_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
TTS_DEVICE = os.environ.get("JIEMA_TTS_DEVICE", "cpu")
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

app = FastAPI(title="Jiema Local MMS/Gemma/MMS-TTS Pipeline")
process = psutil.Process(os.getpid())

asr_processor = None
asr_model = None
llm_tokenizer = None
llm_processor = None
llm_model = None
tts_tokenizer = None
tts_model = None


class AnswerRequest(BaseModel):
    transcript: str
    synthesize: bool = False
    max_new_tokens: int = 180


class SynthesizeRequest(BaseModel):
    text: str


def rss_mb() -> float:
    return round(process.memory_info().rss / 1024**2, 1)


def memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"rss_mb": rss_mb()}
    if torch.backends.mps.is_available():
        snapshot["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024**2, 1)
        snapshot["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 1024**2, 1)
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    return snapshot


def now_metrics(start: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = {
        "elapsed_s": round(time.time() - start, 3),
        **memory_snapshot(),
        "device": DEVICE,
    }
    if extra:
        metrics.update(extra)
    return metrics


def unload_torch_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_asr() -> dict[str, Any]:
    global asr_processor, asr_model
    start = time.time()
    before = rss_mb()
    if asr_model is None:
        asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
        asr_model = AutoModelForCTC.from_pretrained(ASR_MODEL_ID).eval().to(DEVICE)
    return now_metrics(start, {"model": ASR_MODEL_ID, "rss_delta_mb": round(rss_mb() - before, 1)})


def load_llm() -> dict[str, Any]:
    global llm_tokenizer, llm_processor, llm_model
    start = time.time()
    before = rss_mb()
    if llm_model is None:
        llm_processor = AutoProcessor.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        llm_tokenizer = getattr(llm_processor, "tokenizer", None)
        if llm_tokenizer is None:
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        llm_model = AutoModelForMultimodalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        llm_model.to(DEVICE)
    return now_metrics(start, {"model": LLM_MODEL_ID, "rss_delta_mb": round(rss_mb() - before, 1)})


def load_tts() -> dict[str, Any]:
    global tts_tokenizer, tts_model
    start = time.time()
    before = rss_mb()
    if VitsModel is None:
        raise HTTPException(status_code=501, detail="This transformers build does not expose VitsModel.")
    if tts_model is None:
        tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        tts_model = VitsModel.from_pretrained(TTS_MODEL_ID).eval().to(TTS_DEVICE)
    return now_metrics(start, {"model": TTS_MODEL_ID, "tts_device": TTS_DEVICE, "rss_delta_mb": round(rss_mb() - before, 1)})


def read_audio(path: str) -> tuple[torch.Tensor, int]:
    audio, sampling_rate = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return torch.tensor(audio, dtype=torch.float32), int(sampling_rate)


def transcribe_file(path: str) -> tuple[str, dict[str, Any]]:
    load = load_asr()
    start = time.time()
    audio, sampling_rate = read_audio(path)
    inputs = asr_processor(audio.numpy(), sampling_rate=sampling_rate, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = asr_processor.batch_decode(predicted_ids)[0]
    metrics = now_metrics(start, {"load": load})
    return transcript, metrics


def jiema_messages(transcript: str) -> list[dict[str, str]]:
    system = (
        "You are Jiema, a practical assistant for Senegalese users. "
        "The user may write Wolof, French, English, or mixed Wolof-French. "
        "Answer in simple Wolof when possible. If you are unsure, answer in short simple English. "
        "Do not invent bus lines, prices, clinic names, medicine names, medicine doses, phone numbers, or emails. "
        "For health, do not diagnose; give triage and danger signs. "
        "For money or secret codes, tell the user not to share secret codes."
    )
    user = f"User message: {transcript}\nGive one short practical answer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(transcript: str, max_new_tokens: int = 180) -> tuple[str, dict[str, Any]]:
    load = load_llm()
    start = time.time()
    messages = jiema_messages(transcript)
    try:
        prompt = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_processor(
        text=[prompt],
        return_tensors="pt",
        text_kwargs={"padding": False, "truncation": True, "add_special_tokens": False},
    ).to(DEVICE)
    eos_token_id = getattr(llm_model.generation_config, "eos_token_id", None)
    pad_token_id = getattr(llm_model.generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = llm_tokenizer.pad_token_id or llm_tokenizer.eos_token_id
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.08,
        "pad_token_id": pad_token_id,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    first_token_s = None
    chunks: list[str] = []
    thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    for chunk in streamer:
        if chunk and first_token_s is None:
            first_token_s = time.time() - start
        chunks.append(chunk)
    thread.join()
    total_s = time.time() - start
    answer = "".join(chunks).strip()
    retry_used = False
    raw_debug = None
    if not answer:
        retry_used = True
        plain_prompt = (
            "You are Jiema, a practical assistant for Senegalese users.\n"
            "Answer shortly and safely. If possible, answer in simple Wolof.\n\n"
            f"User: {transcript}\nJiema:"
        )
        inputs = llm_processor(
            text=[plain_prompt],
            return_tensors="pt",
            text_kwargs={"padding": False, "truncation": True, "add_special_tokens": False},
        ).to(DEVICE)
        retry_start = time.time()
        with torch.inference_mode():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        total_s = time.time() - start
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        raw_response = llm_tokenizer.decode(new_tokens, skip_special_tokens=False)
        try:
            answer = llm_processor.parse_response(raw_response)["content"].strip()
        except Exception:
            answer = llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        raw_debug = {
            "token_ids": new_tokens[:20].detach().cpu().tolist(),
            "raw_decode": raw_response[:500],
        }
        if answer and first_token_s is None:
            first_token_s = retry_start - start
    metrics = now_metrics(
        start,
        {
            "load": load,
            "ttft_s": round(first_token_s, 3) if first_token_s is not None else None,
            "generation_s": round(total_s, 3),
            "retry_used": retry_used,
            "raw_debug": raw_debug,
            "tokens_generated_estimate": len(llm_tokenizer(answer).input_ids),
        },
    )
    return answer, metrics


def synthesize(text: str) -> tuple[str, dict[str, Any]]:
    load = load_tts()
    start = time.time()
    inputs = tts_tokenizer(text, return_tensors="pt").to(TTS_DEVICE)
    with torch.inference_mode():
        output = tts_model(**inputs).waveform
    waveform = output[0].detach().cpu().float().numpy()
    sampling_rate = int(getattr(tts_model.config, "sampling_rate", 16000))
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sampling_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return audio_b64, now_metrics(start, {"load": load, "sampling_rate": sampling_rate})


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "models": {"asr": ASR_MODEL_ID, "llm": LLM_MODEL_ID, "tts": TTS_MODEL_ID},
        "device": DEVICE,
        "memory": memory_snapshot(),
        "loaded": {
            "asr": asr_model is not None,
            "llm": llm_model is not None,
            "tts": tts_model is not None,
        },
    }


@app.post("/api/warmup/{component}")
def warmup(component: str):
    if component == "asr":
        return {"component": component, "metrics": load_asr(), "memory": memory_snapshot()}
    if component == "llm":
        return {"component": component, "metrics": load_llm(), "memory": memory_snapshot()}
    if component == "tts":
        return {"component": component, "metrics": load_tts(), "memory": memory_snapshot()}
    if component == "all":
        return {
            "component": component,
            "metrics": {"asr": load_asr(), "llm": load_llm(), "tts": load_tts()},
            "memory": memory_snapshot(),
        }
    raise HTTPException(status_code=404, detail="Use asr, llm, tts, or all.")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        transcript, metrics = transcribe_file(tmp_path)
        return {"transcript": transcript, "metrics": metrics}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/answer")
def answer(request: AnswerRequest):
    answer_text, answer_metrics = generate_answer(request.transcript, request.max_new_tokens)
    result: dict[str, Any] = {
        "transcript": request.transcript,
        "answer": answer_text,
        "metrics": {"answer": answer_metrics, "memory": memory_snapshot()},
    }
    if request.synthesize:
        audio_b64, tts_metrics = synthesize(answer_text)
        result["audio_wav_base64"] = audio_b64
        result["metrics"]["tts"] = tts_metrics
    return result


@app.post("/api/synthesize")
def synthesize_endpoint(request: SynthesizeRequest):
    audio_b64, metrics = synthesize(request.text)
    return {"audio_wav_base64": audio_b64, "metrics": metrics, "memory": memory_snapshot()}


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), synthesize_audio: bool = True):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        transcript, asr_metrics = transcribe_file(tmp_path)
        answer_text, answer_metrics = generate_answer(transcript)
        result: dict[str, Any] = {
            "transcript": transcript,
            "answer": answer_text,
            "metrics": {"asr": asr_metrics, "answer": answer_metrics, "memory": memory_snapshot()},
        }
        if synthesize_audio:
            audio_b64, tts_metrics = synthesize(answer_text)
            result["audio_wav_base64"] = audio_b64
            result["metrics"]["tts"] = tts_metrics
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("JIEMA_PIPELINE_PORT", "8008"))
    uvicorn.run(app, host="127.0.0.1", port=port)
