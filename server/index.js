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
const localModel = process.env.LOCAL_CHAT_MODEL || 'Sunflower-Gemma4-E2B';
const localChatUrl = process.env.LOCAL_CHAT_URL || 'http://localhost:11434/api/chat';
const jiemaPipelineUrl = process.env.JIEMA_PIPELINE_URL || '';

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(rootDir, 'public')));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    asr: {
      provider: 'openai',
      model: 'whisper-1',
      configured: Boolean(openai)
    },
    llm: {
      provider: 'local',
      model: localModel,
      url: localChatUrl
    },
    pipeline: {
      provider: jiemaPipelineUrl ? 'local-python' : 'disabled',
      url: jiemaPipelineUrl || null
    }
  });
});

function buildJiemaSystemPrompt() {
  return [
    'You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages.',
    'You are powering Jiema, a Wolof-first assistant for Senegalese users.'
  ].join(' ');
}

function buildJiemaUserPrompt(transcript) {
  return [
    'Task: Read the user message, then produce a practical answer for the user in Wolof.',
    'The user message may be Wolof, French, English, or mixed Wolof-French.',
    'Return only valid JSON. Do not use markdown.',
    'JSON schema:',
    '{"wolof":"short answer in simple Wolof","english":"short English explanation","category":"agriculture|transport|health|admin|education|finance|general","safety":"short safety note"}',
    'Rules:',
    '- Do not invent exact bus lines, prices, clinic names, medicine names, or medicine doses.',
    '- For health, give triage guidance and danger signs. Do not give a definitive diagnosis.',
    '- Keep the Wolof answer short and practical.',
    `User message: ${transcript}`
  ].join('\n');
}

function normalizeAnswer(answer) {
  return {
    wolof: String(answer.wolof || 'Jàppandi na. Jéemaatal.'),
    english: String(answer.english || ''),
    category: String(answer.category || 'general'),
    safety: String(answer.safety || '')
  };
}

function fallbackAnswer(transcript) {
  const text = transcript.toLowerCase();
  if (text.includes('tàng') || text.includes('fièvre') || text.includes('poitrine') || text.includes('respire') || text.includes('ëmb') || text.includes('diarrh')) {
    return {
      wolof: 'Mën na am solo. Waxal ma at mi, naka la yaram wi mel, ak ba ñaata fan la tàmbali. Su dee noyyi metti, xel mi fatt, deret, walla doom bu ndaw la, demal ci fajkat léegi.',
      english: 'Fallback health triage response with urgent danger signs.',
      category: 'health',
      safety: 'Triage only; seek medical care for danger signs.'
    };
  }
  if (text.includes('bus') || text.includes('taxi') || text.includes('sandaga') || text.includes('plateau') || text.includes('colobane')) {
    return {
      wolof: 'Waxal ma fan nga nekk léegi ak fan nga bëgg dem. Su ma amee barab bi, dinaa la dimbali nga xam lan nga wara laaj walla jël.',
      english: 'Fallback transport response asking for origin and destination.',
      category: 'transport',
      safety: 'Does not invent route numbers, fares, or schedules.'
    };
  }
  if (text.includes('gerte') || text.includes('tool') || text.includes('arachide') || text.includes('tomate') || text.includes('xob') || text.includes('ndox')) {
    return {
      wolof: 'Mën na jóge ci ndox, suuf, walla ay dundkat. Xoolal suuf si, xob yi ci suuf ak ci kaw, te wax ak ku xam mbay mi bu dee jafe-jafe bi yokku.',
      english: 'Fallback agriculture response covering water, soil, and pests.',
      category: 'agriculture',
      safety: 'Does not recommend pesticide names or doses.'
    };
  }
  if (text.includes('adresse permanente') || text.includes('permanent address') || text.includes('formulaire')) {
    return {
      wolof: 'Adresse permanente mooy dëkkuwaay bi nga mëna joxe ngir ñu jot la, te du barab bi nga nekkandi rekk. Bindal sa kër walla sa dëkkuwaay bu gën a sax, te laajal ku la dimbali bu leerul.',
      english: 'Fallback form-help response explaining permanent address.',
      category: 'admin',
      safety: 'Explains the form field without asking for private details.'
    };
  }
  if (text.includes('sms') || text.includes('facture') || text.includes('code secret') || text.includes('rendez-vous')) {
    return {
      wolof: 'Bataaxal bi dafay laaj nga seet li ñu bind. Bul joxe sa code secret mukk. Bu dee xaalis walla lien la laaj, seetal bu baax ndax dëgg la.',
      english: 'Fallback SMS/forms response with fraud warning.',
      category: 'admin',
      safety: 'Warns against sharing secret codes.'
    };
  }
  return {
    wolof: 'Maa ngi la dégg. Waxal ma lenn ci lu gàtt ngir ma dimbali la bu baax.',
    english: 'Fallback general clarification response.',
    category: 'general',
    safety: 'Asks for clarification.'
  };
}

function shouldUseFallback(answer, transcript) {
  const wolof = String(answer.wolof || '').trim().toLowerCase();
  const source = transcript.trim().toLowerCase();
  return (
    wolof.length < 20 ||
    wolof === source ||
    wolof.includes('wolof') ||
    wolof.includes('jàppandi na') ||
    wolof.includes('jeemaatal') ||
    wolof.includes('jéemaatal')
  );
}

async function generateWithGemma(transcript) {
  if (jiemaPipelineUrl) {
    const response = await fetch(`${jiemaPipelineUrl}/api/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript, synthesize: false })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Jiema pipeline answer generation failed.');
    }

    return normalizeAnswer({
      wolof: data.answer,
      english: data.answer,
      category: 'general',
      safety: `TTFT ${data.metrics?.answer?.ttft_s ?? 'n/a'}s, MPS ${data.metrics?.memory?.mps_driver_allocated_mb ?? 'n/a'} MB`
    });
  }

  const response = await fetch(localChatUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: localModel,
      stream: false,
      format: 'json',
      messages: [
        {
          role: 'system',
          content: buildJiemaSystemPrompt()
        },
        {
          role: 'user',
          content: buildJiemaUserPrompt(transcript)
        }
      ],
      options: {
        num_predict: 180
      }
    })
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Gemma answer generation failed.');
  }

  return normalizeAnswer(JSON.parse(data.message?.content || data.response || '{}'));
}

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

  try {
    const answer = await generateWithGemma(transcript);
    res.json(shouldUseFallback(answer, transcript) ? fallbackAnswer(transcript) : answer);
  } catch (error) {
    res.status(500).json({ error: error.message || 'Answer generation failed.' });
  }
});

app.listen(port, () => {
  console.log(`Wolof voice demo running at http://localhost:${port}`);
});
