const state = {
  sessionId: crypto.randomUUID(),
  ws: null,
  stream: null,
  audioContext: null,
  sourceNode: null,
  processorNode: null,
  silentGain: null,
  connected: false,
  muted: false,
  speechActive: false,
  voiceMs: 0,
  silenceMs: 0,
  preRoll: [],
  preRollMs: 0,
  assistantSpeaking: false,
  assistantDraft: '',
  assistantEl: null,
  speechBuffer: '',
  queuedSpeech: 0,
  generation: 0,
};

const el = {
  badge: document.querySelector('#connectionBadge'),
  status: document.querySelector('#callStatus'),
  hint: document.querySelector('#callHint'),
  orb: document.querySelector('#voiceOrb'),
  start: document.querySelector('#startButton'),
  mute: document.querySelector('#muteButton'),
  end: document.querySelector('#endButton'),
  reset: document.querySelector('#resetButton'),
  clear: document.querySelector('#clearTranscriptButton'),
  messages: document.querySelector('#messages'),
  textForm: document.querySelector('#textForm'),
  textInput: document.querySelector('#textInput'),
  sendText: document.querySelector('#sendTextButton'),
  runtime: document.querySelector('#runtimeDetails'),
  voiceSelect: document.querySelector('#voiceSelect'),
  vadThreshold: document.querySelector('#vadThreshold'),
};

function setBadge(mode, text) {
  el.badge.className = `badge ${mode}`;
  el.badge.querySelector('b').textContent = text;
}

function setActivity(mode, status, hint) {
  el.orb.className = `voice-orb ${mode || ''}`.trim();
  el.status.textContent = status;
  el.hint.textContent = hint;
}

function updateControls() {
  el.start.disabled = state.connected;
  el.end.disabled = !state.connected;
  el.mute.disabled = !state.connected;
  el.textInput.disabled = !state.connected;
  el.sendText.disabled = !state.connected;
  el.mute.textContent = state.muted ? 'Unmute' : 'Mute';
}

function clearEmptyState() {
  el.messages.querySelector('.empty-state')?.remove();
}

function addMessage(role, text, extraClass = '') {
  clearEmptyState();
  const node = document.createElement('div');
  node.className = `message ${role} ${extraClass}`.trim();
  node.textContent = text;
  el.messages.appendChild(node);
  el.messages.scrollTop = el.messages.scrollHeight;
  return node;
}

function startAssistantDraft() {
  state.assistantDraft = '';
  state.speechBuffer = '';
  state.assistantEl = addMessage('assistant', '');
}

function appendAssistantDelta(delta) {
  if (!state.assistantEl) startAssistantDraft();
  state.assistantDraft += delta;
  state.speechBuffer += delta;
  state.assistantEl.textContent = state.assistantDraft;
  el.messages.scrollTop = el.messages.scrollHeight;
  flushCompleteSentences(false);
}

function markAssistantInterrupted() {
  if (state.assistantEl && !state.assistantEl.classList.contains('interrupted')) {
    state.assistantEl.classList.add('interrupted');
    if (state.assistantEl.textContent.trim()) {
      state.assistantEl.textContent = `${state.assistantEl.textContent.trim()} [interrupted]`;
    }
  }
  state.assistantEl = null;
  state.assistantDraft = '';
  state.speechBuffer = '';
}

function sendJson(payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify(payload));
}

function selectedVoice() {
  const name = el.voiceSelect.value;
  return speechSynthesis.getVoices().find(voice => voice.name === name) || null;
}

function speakChunk(text) {
  const value = String(text || '').trim();
  if (!value || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(value);
  const voice = selectedVoice();
  if (voice) utterance.voice = voice;
  utterance.rate = 1.03;
  utterance.pitch = 1.0;
  state.queuedSpeech += 1;
  utterance.onstart = () => {
    state.assistantSpeaking = true;
    setActivity('speaking', 'Speaking', 'Speak normally to interrupt.');
  };
  utterance.onend = utterance.onerror = () => {
    state.queuedSpeech = Math.max(0, state.queuedSpeech - 1);
    if (state.queuedSpeech === 0) {
      state.assistantSpeaking = false;
      if (state.connected && !state.speechActive) {
        setActivity('', 'Connected', 'Speak naturally; local VAD detects when you finish.');
      }
    }
  };
  speechSynthesis.speak(utterance);
}

function flushCompleteSentences(force) {
  let buffer = state.speechBuffer;
  if (!buffer.trim()) return;

  if (force) {
    speakChunk(buffer);
    state.speechBuffer = '';
    return;
  }

  // Queue complete sentence-sized chunks so playback can begin before the whole
  // local LLM response is finished.
  const boundary = /[.!?](?:\s|$)/g;
  let lastEnd = -1;
  let match;
  while ((match = boundary.exec(buffer)) !== null) {
    if (match.index + 1 >= 28) {
      lastEnd = match.index + 1;
      break;
    }
  }
  if (lastEnd > 0) {
    const chunk = buffer.slice(0, lastEnd);
    state.speechBuffer = buffer.slice(lastEnd);
    speakChunk(chunk);
  }
}

function interruptPlayback(notifyServer = true) {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  state.queuedSpeech = 0;
  state.assistantSpeaking = false;
  markAssistantInterrupted();
  if (notifyServer) sendJson({type: 'interrupt'});
}

function downsampleTo16k(input, inputRate) {
  const outputRate = 16000;
  if (inputRate === outputRate) return new Float32Array(input);
  const ratio = inputRate / outputRate;
  const length = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(length);
  let offset = 0;
  for (let i = 0; i < length; i += 1) {
    const nextOffset = Math.min(input.length, Math.round((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    while (offset < nextOffset) {
      sum += input[offset++];
      count += 1;
    }
    output[i] = count ? sum / count : 0;
  }
  return output;
}

function floatToPcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

function frameRms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(1, samples.length));
}

function beginSpeech() {
  if (state.speechActive) return;
  state.speechActive = true;
  state.silenceMs = 0;
  interruptPlayback(false);
  sendJson({type: 'speech_start'});
  for (const frame of state.preRoll) {
    if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(frame);
  }
  state.preRoll = [];
  state.preRollMs = 0;
  setActivity('listening', 'Listening', 'Keep speaking; pause naturally when finished.');
}

function finishSpeech() {
  if (!state.speechActive) return;
  state.speechActive = false;
  state.voiceMs = 0;
  state.silenceMs = 0;
  sendJson({type: 'speech_end'});
  setActivity('thinking', 'Transcribing locally', 'The first turn can take longer while the model loads.');
}

function processAudioFrame(input) {
  if (!state.connected || state.muted || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

  const samples16k = downsampleTo16k(input, state.audioContext.sampleRate);
  const pcm = floatToPcm16(samples16k);
  const rms = frameRms(input);
  const frameMs = (input.length / state.audioContext.sampleRate) * 1000;
  const sensitivity = Number(el.vadThreshold.value);
  const threshold = (63 - sensitivity) / 1000;
  const isVoice = rms >= threshold;

  if (isVoice) {
    state.voiceMs += frameMs;
    state.silenceMs = 0;
    if (!state.speechActive && state.voiceMs >= 110) beginSpeech();
  } else {
    state.voiceMs = Math.max(0, state.voiceMs - frameMs * 0.6);
    if (state.speechActive) state.silenceMs += frameMs;
  }

  if (state.speechActive) {
    state.ws.send(pcm);
    if (state.silenceMs >= 720) finishSpeech();
  } else {
    state.preRoll.push(pcm);
    state.preRollMs += frameMs;
    while (state.preRollMs > 280 && state.preRoll.length > 1) {
      state.preRoll.shift();
      state.preRollMs -= frameMs;
    }
  }
}

function populateVoices() {
  if (!('speechSynthesis' in window)) return;
  const previous = el.voiceSelect.value;
  const voices = speechSynthesis.getVoices();
  el.voiceSelect.innerHTML = '<option value="">Browser default</option>';
  for (const voice of voices) {
    const option = document.createElement('option');
    option.value = voice.name;
    option.textContent = `${voice.name} (${voice.lang})${voice.localService ? ' · local' : ''}`;
    el.voiceSelect.appendChild(option);
  }
  if ([...el.voiceSelect.options].some(option => option.value === previous)) {
    el.voiceSelect.value = previous;
  } else {
    const preferred = voices.find(voice => voice.localService && /^en/i.test(voice.lang))
      || voices.find(voice => /^en/i.test(voice.lang));
    if (preferred) el.voiceSelect.value = preferred.name;
  }
}

function handleServerEvent(event) {
  switch (event.type) {
    case 'ready':
      state.connected = true;
      setBadge('online', 'Local');
      setActivity('', 'Connected', 'Speak naturally; local VAD detects when you finish.');
      updateControls();
      break;
    case 'listening':
      setActivity('listening', 'Listening', 'Keep speaking; pause naturally when finished.');
      break;
    case 'transcribing':
      setActivity('thinking', 'Transcribing locally', 'No cloud API is being used.');
      break;
    case 'thinking':
      setActivity('thinking', 'Thinking locally', 'MOSAIC-CRS and Ollama are preparing the response.');
      break;
    case 'user_transcript':
      if (event.text?.trim()) addMessage('user', event.text.trim());
      break;
    case 'assistant_start':
      state.generation = Number(event.generation || state.generation);
      startAssistantDraft();
      setActivity('thinking', 'Preparing response', 'The answer will begin speaking sentence by sentence.');
      break;
    case 'assistant_delta':
      if (Number(event.generation) !== state.generation) return;
      appendAssistantDelta(event.delta || '');
      break;
    case 'assistant_done':
      if (Number(event.generation) !== state.generation) return;
      if (state.assistantEl && !state.assistantEl.textContent.trim()) {
        state.assistantEl.textContent = event.text || '';
      }
      flushCompleteSentences(true);
      state.assistantEl = null;
      state.assistantDraft = '';
      break;
    case 'assistant_interrupted':
      state.generation = Number(event.generation || state.generation + 1);
      interruptPlayback(false);
      if (state.connected && !state.speechActive) {
        setActivity('', 'Interrupted', 'Continue with your newest request.');
      }
      break;
    case 'empty_transcript':
      addMessage('tool', 'I did not hear enough speech. Please try again or raise the microphone sensitivity.');
      setActivity('', 'Connected', 'Speak naturally; local VAD detects when you finish.');
      break;
    case 'reset_done':
      break;
    case 'error':
      addMessage('tool', `Local runtime error: ${event.message || 'Unknown error'}`);
      setActivity('', 'Error', 'Open Local runtime details and check the PowerShell window.');
      break;
    default:
      break;
  }
}

async function startCall() {
  if (state.connected) return;
  setBadge('connecting', 'Connecting');
  setActivity('thinking', 'Connecting locally', 'Requesting microphone permission…');
  el.start.disabled = true;

  try {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    state.ws = new WebSocket(`${scheme}://${location.host}/ws/call/${state.sessionId}`);
    state.ws.binaryType = 'arraybuffer';
    state.ws.onmessage = message => {
      if (typeof message.data !== 'string') return;
      try { handleServerEvent(JSON.parse(message.data)); }
      catch (error) { console.error('Invalid local event', error, message.data); }
    };
    state.ws.onerror = () => {
      addMessage('tool', 'Could not connect to the local WebSocket server.');
    };
    state.ws.onclose = () => {
      if (state.connected) stopCall(false);
    };

    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true},
    });
    state.audioContext = new AudioContext();
    state.sourceNode = state.audioContext.createMediaStreamSource(state.stream);
    state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.silentGain = state.audioContext.createGain();
    state.silentGain.gain.value = 0;
    state.processorNode.onaudioprocess = event => {
      processAudioFrame(event.inputBuffer.getChannelData(0));
    };
    state.sourceNode.connect(state.processorNode);
    state.processorNode.connect(state.silentGain);
    state.silentGain.connect(state.audioContext.destination);
  } catch (error) {
    console.error(error);
    addMessage('tool', `Could not start local call: ${error.message || error}`);
    stopCall(false);
  }
}

function stopCall(userInitiated = true) {
  interruptPlayback(false);
  try { state.processorNode?.disconnect(); } catch (_) {}
  try { state.sourceNode?.disconnect(); } catch (_) {}
  try { state.silentGain?.disconnect(); } catch (_) {}
  try { state.audioContext?.close(); } catch (_) {}
  state.stream?.getTracks().forEach(track => track.stop());
  try { state.ws?.close(); } catch (_) {}

  state.ws = null;
  state.stream = null;
  state.audioContext = null;
  state.sourceNode = null;
  state.processorNode = null;
  state.silentGain = null;
  state.connected = false;
  state.muted = false;
  state.speechActive = false;
  state.preRoll = [];
  state.preRollMs = 0;
  setBadge('offline', 'Offline');
  setActivity('', userInitiated ? 'Call ended' : 'Disconnected', 'Start a new local call when ready.');
  updateControls();
}

function toggleMute() {
  state.muted = !state.muted;
  const track = state.stream?.getAudioTracks?.()[0];
  if (track) track.enabled = !state.muted;
  setActivity('', state.muted ? 'Microphone muted' : 'Connected', state.muted ? 'Unmute to continue.' : 'Speak naturally; local VAD detects when you finish.');
  updateControls();
}

function sendTypedMessage(text) {
  const value = String(text || '').trim();
  if (!value || !state.connected) return;
  interruptPlayback(true);
  addMessage('user', value);
  sendJson({type: 'typed_message', text: value});
  setActivity('thinking', 'Thinking locally', 'MOSAIC-CRS and Ollama are preparing the response.');
}

async function resetSession() {
  stopCall(true);
  try {
    await fetch('/api/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: state.sessionId}),
    });
  } catch (_) {}
  state.sessionId = crypto.randomUUID();
  el.messages.innerHTML = '<div class="empty-state">Start a call, then ask for a movie recommendation.</div>';
  el.textInput.value = '';
}

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    el.runtime.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    el.runtime.textContent = String(error.message || error);
  }
}

el.start.addEventListener('click', startCall);
el.end.addEventListener('click', () => stopCall(true));
el.mute.addEventListener('click', toggleMute);
el.reset.addEventListener('click', resetSession);
el.clear.addEventListener('click', () => {
  el.messages.innerHTML = '<div class="empty-state">Transcript cleared. The recommendation session memory is unchanged.</div>';
});
el.textForm.addEventListener('submit', event => {
  event.preventDefault();
  const value = el.textInput.value;
  el.textInput.value = '';
  sendTypedMessage(value);
});

if ('speechSynthesis' in window) {
  populateVoices();
  speechSynthesis.onvoiceschanged = populateVoices;
}

updateControls();
loadStatus();
