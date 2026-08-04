# Jiema Demo Proposal

## One-Line Summary

Jiema is a Wolof-first voice assistant that helps Senegalese users access large language model capabilities without needing to speak or read French.

## Problem

Many Senegalese people use Wolof as their primary everyday language, while formal digital services, healthcare information, education material, and administrative processes often rely on French. This creates a practical access gap: people may be able to speak their needs clearly, but not comfortably type, search, or read service information in French.

Jiema explores a voice-first interface for that gap. The user can speak in Wolof, French, Senegalese French, or a natural mix. The system may use French or English internally, but the user-facing answer should be simple spoken Wolof.

## Target User

The first target users are Senegalese users who:

- Prefer Wolof for everyday communication.
- May not read or write French comfortably.
- Are more comfortable speaking than typing.
- Need practical help with daily questions, transport, agriculture, health triage, forms, messages, or local services.

## Product Principle

Jiema is not a "Wolof typing app." It is a voice gateway to AI.

The product should accept messy real-world input:

```text
Wolof speech
French speech
Wolof-French mixed speech
Messy Latin Wolof typing
French text copied from SMS/forms
```

The response should be short, practical, and understandable when read aloud in Wolof.

## Current Demo

The current demo is a browser prototype with:

- A mobile-first visual interface.
- Jiema branding.
- A voice input button.
- Text input as fallback.
- A chat-style response area.
- A Wolof/French ASR backend route.
- A Wolof answer-generation backend route.

Current technical flow:

```text
Audio input
-> OpenAI whisper-1 transcription
-> LLM reasoning with Wolof/French/English support
-> JSON response
-> Simple Wolof answer displayed in chat
```

The backend currently uses:

- ASR: `whisper-1`
- LLM: `gpt-4o-mini` by default, configurable with `OPENAI_CHAT_MODEL`

## Demo Scope

The demo should prove three things:

1. A user can ask a question by voice without choosing a domain.
2. The system can handle Wolof, French, and mixed input.
3. The assistant can answer in simple Wolof while using stronger French/English resources internally.

The demo should avoid:

- Domain selector UI.
- Visible model/API/debug text.
- Requiring users to type standard Wolof.
- Over-explaining the mission inside the product UI.

## Initial Use Cases

Jiema should start with practical, high-value questions:

- Agriculture: crops, pests, irrigation, fertilizer, weather-related farming advice.
- Transport: buses, landmarks, destinations, routes, transfer questions.
- Health: basic triage, danger signs, possible causes, and when to seek care.
- Messages/forms: explaining French SMS, notices, or simple forms in Wolof.

Health support must be framed as triage and guidance, not definitive diagnosis. Jiema should be conservative for children, pregnancy, older adults, severe symptoms, and chronic conditions.

## Data Plan

The next data work should prioritize:

- Kallaama / OpenSLR 151 for Wolof agriculture speech and text.
- Urban Bus Wolof for Dakar transport speech.
- Common Voice / ALFFA / aggregated Wolof ASR data for robustness.
- French-Wolof parallel data such as SENCORPUS for translation and explanation.
- Small native-speaker evaluation sets for real Wolof quality checks.

Minimum evaluation set:

```text
50 agriculture questions
50 transport questions
30 health triage questions
20 French SMS/form explanation tasks
```

Each item should include:

```text
audio or text input
human transcript
expected answer points
input language mix
domain
risk level
native-speaker quality rating
```

## Evaluation

Jiema should be evaluated on:

- ASR transcript quality for Wolof and mixed Wolof-French.
- Key entity recall: places, symptoms, crops, pests, medications, numbers.
- Answer usefulness.
- Wolof naturalness.
- Safety behavior for health questions.
- Ability to recover when the user is unclear.

Success for the first demo is not perfect Wolof. Success is showing that a non-French-speaking user can ask a practical question and receive a useful Wolof explanation.

## Risks

- Whisper may struggle with Wolof, local names, and code-switching.
- Wolof spelling and transcription are inconsistent.
- LLM Wolof output may sound unnatural without native review.
- Health guidance can be harmful if the safety boundary is weak.
- The project needs local Senegalese validation before claiming user readiness.

## Next Steps

1. Connect the current frontend to the real backend functions, replacing any remaining placeholder demo logic.
2. Test with typed Wolof/French examples using an `OPENAI_API_KEY`.
3. Record or source 20-30 short Wolof audio samples for first ASR testing.
4. Ask native Wolof speakers to rate answer clarity and naturalness.
5. Build a small evaluation spreadsheet from agriculture, transport, health, and French-message examples.
6. Decide whether to fine-tune ASR after the first transcript-quality review.

## Pitch

Jiema is a voice-first AI assistant for Wolof-speaking Senegalese users. It lets people ask questions naturally in Wolof, French, or a mix, then returns practical Wolof explanations. The first demo focuses on proving the access layer: speech in, AI reasoning, Wolof explanation out.

Longer term, Jiema can become a local-language interface for agriculture, transport, health triage, administrative services, education, and financial literacy.
