# Jiema

A small browser demo for a Wolof-first voice assistant with Wolof/French ASR input:

```text
Wolof/French/mixed voice or transcript -> Whisper transcription -> LLM answer in simple Wolof
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

Without `OPENAI_API_KEY`, the UI can be reviewed but live transcription and answers are disabled.

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

- Uses OpenAI `whisper-1` for transcription.
- Whisper is prompted to expect Wolof, French, Senegalese French, and Wolof-French code-switching.
- Uses `gpt-4o-mini` by default for answers. Set `OPENAI_CHAT_MODEL` to override it.
- Health answers are constrained to triage and possible causes, not definitive diagnosis.
