# Jiema — Wolof Assistant

Jiema is a Wolof-first assistant for Senegalese users who may prefer Wolof over French. It is being built toward a local, low-memory voice stack for affordable Android devices.

```text
Wolof voice or transcript -> Wolof ASR -> small local LLM -> simple Wolof answer -> optional Wolof TTS
```

The app does not require a domain selector. It accepts Wolof, French, and mixed Wolof-French input, then asks the model to infer whether the request is about agriculture, transport, health, administration, education, finance, or general daily help.

See [PROPOSAL.md](./PROPOSAL.md) for the product/demo proposal.

## Run

```bash
npm install
OPENAI_API_KEY=your_key npm run dev
```

Then open:

```text
http://localhost:5177
```

Without `OPENAI_API_KEY`, the original Whisper path is disabled. Text answers can still run through the local backend paths described below.

## Local Pipeline

The current local pipeline server is experimental and runs on your machine:

```bash
python3 server/jiema_pipeline_server.py
```

Default local models:

```text
ASR: sokho2/mms-300m-wolof
LLM: google/gemma-4-E2B-it-qat-mobile-transformers
TTS: facebook/mms-tts-wlx
```

Then point the Node app at it:

```bash
JIEMA_PIPELINE_URL=http://127.0.0.1:8008 npm run dev
```

Useful local endpoints:

```text
GET  /api/health
POST /api/warmup/all
POST /api/answer
POST /api/transcribe
POST /api/synthesize
```

Mac prototype note: `google/gemma-4-E2B-it-qat-mobile-transformers` loads through Transformers/MPS on the tested Mac, but text generation currently returns `<pad><eos>`. The model remains the mobile/QAT target, while the Mac path is mainly useful for memory and integration experiments until the runtime issue is fixed.

## Legacy Local Chat Endpoint

Jiema also supports an Ollama-style local chat endpoint for earlier Sunflower/Gemma experiments.

The backend calls an Ollama-style `/api/chat` endpoint:

```bash
LOCAL_CHAT_MODEL=Sunflower-Gemma4-E2B \
LOCAL_CHAT_URL=http://localhost:11434/api/chat \
OPENAI_API_KEY=your_key \
npm run dev
```

This path is useful for comparing local LLM checkpoints without changing the browser UI.

## Quantization Spike

Use [notebooks/JIEMA_SUNFLOWER_QUANTIZATION_SPIKE.ipynb](./notebooks/JIEMA_SUNFLOWER_QUANTIZATION_SPIKE.ipynb) in Colab to test whether local Gemma-family checkpoints can run under compressed/mobile-oriented settings and still produce useful Jiema responses.

The notebook is a feasibility check. It does not yet produce an Android-ready artifact.

## Chat History

Chat history is stored locally in the browser with `localStorage`. It does not require a cloud database and does not sync across devices. The UI includes a local keyword search over saved chats.

## Wolof Test Prompts

- `Sama doom dafa am tàngoor bu metti, lan laa wara def?`  
  My child has a bad fever. What should I do?
- `Fan laa wara jël bus bu dem Sandaga?`  
  Where should I take a bus to Sandaga?
- `Xob yi ci sama gerte dañuy weex, ndax dundkat la walla ndox la?`  
  My peanut leaves are turning pale. Is it pests or water?
- `Bataaxal bii ci français la, manuma ko xam. Lan la ma laaj?`  
  This message is in French and I do not understand it. What is it asking me?

## Notes

- Local ASR target is `sokho2/mms-300m-wolof`.
- Local TTS target is `facebook/mms-tts-wlx`.
- Local LLM target is a Gemma E2B QAT/mobile checkpoint, with Qwen-family text-only models as comparison baselines.
- Health answers are constrained to triage and possible causes, not definitive diagnosis.
