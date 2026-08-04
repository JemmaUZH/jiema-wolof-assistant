const shell = document.querySelector('#shell');
const log = document.querySelector('#log');
const input = document.querySelector('#text');
const send = document.querySelector('#send');
const mic = document.querySelector('#mic');

let recorder;
let chunks = [];
let recording = false;

function setState(state) {
  shell.classList.remove('listening', 'thinking');
  if (state !== 'idle') shell.classList.add(state);
}

function addTurn(who, text) {
  const turn = document.createElement('div');
  turn.className = `turn ${who}`;

  const label = document.createElement('p');
  label.className = 'who';
  label.textContent = who === 'you' ? 'YOW' : 'JIEMA';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  turn.append(label, bubble);
  log.appendChild(turn);
  log.scrollTop = log.scrollHeight;
}

function showThinking() {
  const turn = document.createElement('div');
  turn.className = 'turn jiema';
  turn.id = 'thinking';
  turn.innerHTML = '<div class="dots" aria-label="Jiema mungi xalaat"><span></span><span></span><span></span></div>';
  log.appendChild(turn);
  log.scrollTop = log.scrollHeight;
}

function clearThinking() {
  document.querySelector('#thinking')?.remove();
}

async function getReply(userText) {
  const response = await fetch('/api/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript: userText })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Answer failed.');
  return data.wolof || 'Jàppandi na. Jéemaatal.';
}

async function transcribe(audioBlob) {
  const body = new FormData();
  body.append('audio', audioBlob, 'speech.webm');

  const response = await fetch('/api/transcribe', {
    method: 'POST',
    body
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Transcription failed.');
  return data.transcript || '';
}

async function ask(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  addTurn('you', trimmed);
  setState('thinking');
  showThinking();

  try {
    const reply = await getReply(trimmed);
    clearThinking();
    addTurn('jiema', reply);
  } catch (error) {
    clearThinking();
    addTurn('jiema', 'Jiema jekkagul léegi.');
  } finally {
    setState('idle');
  }
}

async function toggleMic() {
  if (recording) {
    recorder?.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });

    recorder.addEventListener('stop', async () => {
      stream.getTracks().forEach((track) => track.stop());
      recording = false;
      setState('thinking');
      showThinking();

      try {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const transcript = await transcribe(blob);
        clearThinking();
        await ask(transcript);
      } catch (error) {
        clearThinking();
        addTurn('jiema', 'Jiema jekkagul léegi.');
        setState('idle');
      }
    });

    recorder.start();
    recording = true;
    setState('listening');
  } catch (error) {
    addTurn('jiema', 'Duma gis micro bi.');
  }
}

send.addEventListener('click', () => {
  const value = input.value;
  input.value = '';
  ask(value);
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    const value = input.value;
    input.value = '';
    ask(value);
  }
});

mic.addEventListener('click', toggleMic);
