# Jiema Model Plan

## Decision

Jiema will use a Gemma E2B QAT/mobile checkpoint as the first target local LLM family, while keeping Qwen-style dense text models as comparison baselines.

This is not because Gemma is guaranteed to be the best Wolof model. It is because Jiema's long-term product constraint is local, low-cost Android deployment, and the Gemma E2B/QAT mobile line gives the project a concrete small-model target that matches that constraint.

## Current State

The current demo has two backend paths: a legacy local chat endpoint and an experimental local voice pipeline.

```text
ASR: sokho2/mms-300m-wolof
LLM: google/gemma-4-E2B-it-qat-mobile-transformers
TTS: facebook/mms-tts-wlx
```

The experimental pipeline runs with:

```text
python3 server/jiema_pipeline_server.py
JIEMA_PIPELINE_URL=http://127.0.0.1:8008 npm run dev
```

On the tested MacBook/MPS runtime, the Gemma QAT checkpoint loads but currently returns `<pad><eos>` during text generation. This is treated as a runtime compatibility blocker, not a final judgment of the mobile checkpoint.

## Why Gemma E2B QAT / Mobile

- The project is ultimately meant for affordable phones, not only servers.
- A small target model forces evaluation around latency, memory, and answer usefulness.
- The E2B/QAT/mobile line gives Jiema a credible on-device direction without starting from an expensive custom QAT pipeline.
- It keeps the project honest: first prove a small model can produce useful Wolof explanations, then optimize.
- It preserves a path toward future multimodal support.

## What Gemma Must Prove

The LLM should not be treated as final until it passes a Wolof/French evaluation set and a mobile-runtime latency/memory profile.

Minimum evaluation:

```text
20 Wolof questions
20 French questions
20 mixed Wolof-French questions
10 agriculture questions
10 transport questions
10 health triage questions
```

Scoring:

```text
understands intent
keeps answer in simple Wolof
does not hallucinate local facts
handles health conservatively
latency is acceptable
memory use is realistic for Android
```

## Baselines

Qwen small models remain strong comparison baselines. They should be tested against the Gemma E2B target for multilingual understanding, instruction following, latency, and memory.

The project should use evaluation results, not model branding, to decide whether Gemma remains the final model.

## Roadmap

1. Keep the browser demo connected to the local Python pipeline.
2. Resolve Gemma E2B QAT generation on a supported runtime such as Colab/CUDA or Android LiteRT.
3. Build a small native-speaker Wolof/French eval set.
4. Compare Gemma E2B QAT against Qwen small models.
5. Profile TTFT, tokens/s, peak memory, and post-generation retained memory.
6. Distill or optimize the Wolof ASR path if `sokho2/mms-300m-wolof` remains too heavy.
7. Profile Android feasibility under the sub-4GB target.
