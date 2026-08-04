import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import OpenAI from 'openai';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');
const upload = multer({ dest: path.join(rootDir, 'tmp') });

const app = express();
const port = process.env.PORT || 5177;
const openai = process.env.OPENAI_API_KEY ? new OpenAI() : null;

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(rootDir, 'public')));

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, openaiConfigured: Boolean(openai) });
});

app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  if (!openai) {
    return res.status(503).json({ error: 'Voice is not ready yet.' });
  }

  if (!req.file) {
    return res.status(400).json({ error: 'Missing audio file.' });
  }

  try {
    const audio = await fs.open(req.file.path);
    const transcription = await openai.audio.transcriptions.create({
      file: audio.createReadStream(),
      model: 'whisper-1',
      prompt:
        'The audio may contain Wolof, French, Senegalese French, or code-switching between Wolof and French. Preserve names, places, and local terms as accurately as possible.',
      response_format: 'verbose_json'
    });

    res.json({
      transcript: transcription.text || '',
      language: transcription.language,
      duration: transcription.duration
    });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Transcription failed.' });
  } finally {
    if (req.file?.path) {
      await fs.unlink(req.file.path).catch(() => {});
    }
  }
});

app.post('/api/answer', async (req, res) => {
  const { transcript } = req.body || {};
  if (!transcript || typeof transcript !== 'string') {
    return res.status(400).json({ error: 'Missing transcript.' });
  }

  if (!openai) {
    return res.status(503).json({ error: 'Jiema is not ready yet.' });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: [
            'You are a Wolof voice assistant for Senegalese users who may not speak French.',
            'The user transcript may be Wolof, French, Senegalese French, English, or mixed Wolof-French.',
            'Answer primarily in simple spoken Wolof. Keep it short and practical.',
            'Internally you may use French or English reasoning, but do not force the user to know French.',
            'Automatically infer the broad category: agriculture, transport, health, admin, education, finance, or general.',
            'For health questions: provide triage and possible causes, not a definitive diagnosis. Mention urgent danger signs and encourage local medical care for serious symptoms. Do not prescribe antibiotics or exact prescription medicine doses.',
            'Return strict JSON with keys: wolof, english, category, safety.'
          ].join(' ')
        },
        {
          role: 'user',
          content: transcript
        }
      ]
    });

    const raw = completion.choices[0]?.message?.content || '{}';
    res.json(JSON.parse(raw));
  } catch (error) {
    res.status(500).json({ error: error.message || 'Answer generation failed.' });
  }
});

app.listen(port, () => {
  console.log(`Wolof voice demo running at http://localhost:${port}`);
});
