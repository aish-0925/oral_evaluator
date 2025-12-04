// static/js/recorder.js
// Robust recorder: stops system mic, 15s auto-stop, uses server audio_url with cache-bust,
// shows feedback, avoids memory leaks. Shows metrics GRAPH ONLY if server returns plot_path.

const MAX_TIME_MS = 15000;

let currentQuestion = null;
let mediaRecorder = null;
let chunks = [];
let progressTimer = null;
let autoStopTimer = null;
let clientStream = null; // store MediaStream to stop tracks
let lastObjectURL = null;

// DOM elements (must exist in template)
const questionEl = document.getElementById('question');
const qidEl = document.getElementById('qid');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const progressFill = document.getElementById('progressFill');
const timeLeft = document.getElementById('timeLeft');
const playerWrap = document.getElementById('playerWrap');
const player = document.getElementById('player');
const feedback = document.getElementById('feedback');
const scoreEl = document.getElementById('score');
const scoreLabel = document.getElementById('scoreLabel');
const transcriptEl = document.getElementById('transcript');
const subscoresEl = document.getElementById('subscores');
const bestRefEl = document.getElementById('bestRef');
const graphsEl = document.getElementById('graphs');
const metricGraph = document.getElementById('metricGraph');
const nextQuestionBtn = document.getElementById('nextQuestion');
const retryBtn = document.getElementById('retry');
const nextBtn = document.getElementById('next');

// small safe-access helpers (in case any element is missing)
function elSetText(el, txt){ if(el) el.innerText = txt; }
function elShow(el, show = true){ if(el) el.style.display = show ? 'block' : 'none'; }
function elSetSrc(el, src){ if(el) el.src = src; }
function elClear(el){ if(el) el.innerHTML = ''; }

function setProgress(pct){
  if(progressFill) progressFill.style.width = `${Math.max(0, Math.min(100, pct*100))}%`;
}
function formatTime(ms){
  let s = Math.ceil(ms/1000);
  if(s < 0) s = 0;
  return '00:' + String(s).padStart(2,'0');
}
function revokeLastObjectURL(){
  if(lastObjectURL){
    try { URL.revokeObjectURL(lastObjectURL); } catch(e){/*ignore*/ }
    lastObjectURL = null;
  }
}

function resetUIForRecord(){
  // stop/cleanup any running timers/streams
  if(autoStopTimer){ clearTimeout(autoStopTimer); autoStopTimer = null; }
  stopProgress();
  revokeLastObjectURL();

  setProgress(0);
  elSetText(timeLeft, formatTime(MAX_TIME_MS));
  elShow(feedback, false);
  elShow(playerWrap, false);
  elSetSrc(player, '');
  elClear(subscoresEl);
  elSetText(bestRefEl, '');
  elSetText(transcriptEl, '');
  elSetText(scoreEl, '-');
  elSetText(scoreLabel, '');
  elShow(graphsEl, false);
  if(metricGraph) metricGraph.src = '';
  if(nextQuestionBtn) nextQuestionBtn.disabled = true;
  if(nextBtn) nextBtn.disabled = true;
}

// Load a random question from server
async function loadQuestion(){
  try{
    const r = await fetch('/api/questions');
    const qs = await r.json();
    if(!qs || qs.length === 0){
      elSetText(questionEl, 'No questions available');
      elSetText(qidEl, '');
      if(startBtn) startBtn.disabled = true;
      return;
    }
    currentQuestion = qs[Math.floor(Math.random()*qs.length)];
    elSetText(questionEl, currentQuestion.question_text || '(No text)');
    elSetText(qidEl, currentQuestion.question_id || '');
    resetUIForRecord();
    if(startBtn) startBtn.disabled = false;
    if(stopBtn) stopBtn.disabled = true;
  }catch(err){
    console.error('Failed to load question', err);
    elSetText(questionEl, 'Failed to load question');
    if(startBtn) startBtn.disabled = true;
  }
}

// Start recording (asks for mic permission)
if(startBtn) startBtn.addEventListener('click', async ()=>{
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    alert('Microphone not available. Use Chrome/Edge and open via http://localhost:8501');
    return;
  }
  if(!currentQuestion){
    alert('No question loaded');
    return;
  }

  try{
    // acquire stream and keep reference so we can stop hardware completely later
    clientStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(clientStream);
    chunks = [];
    mediaRecorder.ondataavailable = e => { if(e.data && e.data.size>0) chunks.push(e.data); };
    mediaRecorder.onstop = onStopRecording;
    mediaRecorder.start();
    if(startBtn) startBtn.disabled = true;
    if(stopBtn) stopBtn.disabled = false;
    startProgress();

    // auto-stop after MAX_TIME_MS
    if(autoStopTimer) clearTimeout(autoStopTimer);
    autoStopTimer = setTimeout(()=>{
      if(mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
    }, MAX_TIME_MS);
  }catch(err){
    alert('Microphone permission required: ' + (err.message || err));
    clientStream = null;
  }
});

// Stop button stops the recording and then onStopRecording will handle hardware stop
if(stopBtn) stopBtn.addEventListener('click', ()=>{
  if(mediaRecorder && mediaRecorder.state === 'recording'){
    mediaRecorder.stop();
    // clear auto-stop timer if manual stop
    if(autoStopTimer){ clearTimeout(autoStopTimer); autoStopTimer = null; }
  }
});

// Progress UI
function startProgress(){
  const t0 = Date.now();
  stopProgress();
  progressTimer = setInterval(()=>{
    const elapsed = Date.now() - t0;
    const pct = Math.min(1, elapsed / MAX_TIME_MS);
    setProgress(pct);
    elSetText(timeLeft, formatTime(Math.max(0, MAX_TIME_MS - elapsed)));
    if(pct >= 1) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }, 80);
}
function stopProgress(){
  if(progressTimer){ clearInterval(progressTimer); progressTimer = null; }
  // keep current progress at 0 when resetting
  setProgress(0);
  elSetText(timeLeft, formatTime(MAX_TIME_MS));
}

// onStopRecording: stop mic hardware, build blob, upload, show results
async function onStopRecording(){
  // clear autos
  if(autoStopTimer){ clearTimeout(autoStopTimer); autoStopTimer = null; }
  stopProgress();
  if(startBtn) startBtn.disabled = false;
  if(stopBtn) stopBtn.disabled = true;

  // --- FULLY STOP the microphone hardware (so system mic indicator disappears) ---
  try{
    if(clientStream){
      clientStream.getTracks().forEach(t => {
        try{ t.stop(); } catch(e){ /* ignore */ }
      });
      clientStream = null;
    }
  }catch(e){
    console.warn('Error stopping client stream', e);
  }

  // Build audio blob
  const blob = new Blob(chunks, { type: 'audio/webm' });

  // Show immediate feedback while uploading
  elSetText(transcriptEl, 'Uploading & evaluating...');
  elShow(feedback, true);

  try{
    const fd = new FormData();
    fd.append('file', blob, 'answer.webm');
    fd.append('question_id', currentQuestion.question_id);

    const resp = await fetch('/api/evaluate', { method: 'POST', body: fd });
    if(!resp.ok){
      const txt = await resp.text().catch(()=>'(no response body)');
      // show a friendly message; log details to console
      console.error('Server returned error for /api/evaluate:', resp.status, txt);
      elSetText(transcriptEl, 'Server error during evaluation — check server logs.');
      return;
    }
    const json = await resp.json();
    handleEvalResponse(json, blob);
  }catch(err){
    console.error('Upload/eval failed', err);
    // show a user-friendly message (do NOT write raw error like "Failed to fetch" in transcript)
    elSetText(transcriptEl, 'Upload or evaluation failed — check the server and browser console for details.');
  } finally {
    if(nextQuestionBtn) nextQuestionBtn.disabled = false;
    if(nextBtn) nextBtn.disabled = false;
  }
}

// Handle JSON response from server and update UI
function handleEvalResponse(json, blob){
  // Transcript
  const transcript = json.transcript || json.transcribed_text || '';
  elSetText(transcriptEl, transcript);

  // Score
  const score = json.final_score_0_10 ?? json.final_score ?? json.scaled_score ?? null;
  const sc = (score !== null) ? Number(score) : null;
  elSetText(scoreEl, sc !== null ? sc.toFixed(1) : '-');
  if(sc !== null){
    if(sc >= 7.0) elSetText(scoreLabel, 'Good');
    else if(sc >= 4.0) elSetText(scoreLabel, 'Needs improvement');
    else elSetText(scoreLabel, 'Weak');
  } else {
    elSetText(scoreLabel, '');
  }

  // Subscores
  elClear(subscoresEl);
  const subs = [
    {k:'semantic_similarity', label:'Semantic'},
    {k:'concept_coverage', label:'Concept coverage'},
    {k:'nli_score', label:'NLI'},
    {k:'asr_confidence', label:'ASR confidence'}
  ];
  subs.forEach(s=>{
    if(json[s.k] !== undefined && json[s.k] !== null){
      const val = (typeof json[s.k] === 'number') ? Number(json[s.k]).toFixed(3) : json[s.k];
      const div = document.createElement('div'); div.className='chip';
      div.innerHTML = `<strong>${s.label}:</strong> ${val}`;
      if(subscoresEl) subscoresEl.appendChild(div);
    }
  });

  // matched/missing concepts
  if(json.matched_concepts && Array.isArray(json.matched_concepts) && json.matched_concepts.length){
    const d = document.createElement('div'); d.className='chip';
    d.innerHTML = `<strong>Matched:</strong> ${json.matched_concepts.join(', ')}`;
    if(subscoresEl) subscoresEl.appendChild(d);
  }
  if(json.missing_concepts && Array.isArray(json.missing_concepts) && json.missing_concepts.length){
    const d2 = document.createElement('div'); d2.className='chip';
    d2.innerHTML = `<strong>Missing:</strong> ${json.missing_concepts.join(', ')}`;
    if(subscoresEl) subscoresEl.appendChild(d2);
  }

  // best reference
  elSetText(bestRefEl, json.best_reference || '');

  // Show recorded audio player using server-provided audio_url (preferred), otherwise audio_path, otherwise client blob.
  revokeLastObjectURL(); // clean up previous
  let audioSrc = null;
  if(json.audio_url){
    audioSrc = json.audio_url;
  } else if(json.audio_path){
    const parts = String(json.audio_path).split('/');
    const fname = parts[parts.length-1];
    if(fname) audioSrc = '/audio/' + encodeURIComponent(fname);
  }

  if(audioSrc){
    // cache-bust so latest file is fetched
    audioSrc = audioSrc + (audioSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
    elSetSrc(player, audioSrc);
    try { if(player) player.load(); } catch(e){ /* ignore */ }
    elShow(playerWrap, true);
  } else {
    // fallback to client blob URL
    const url = URL.createObjectURL(blob);
    lastObjectURL = url;
    elSetSrc(player, url);
    elShow(playerWrap, true);
  }

  // IMPORTANT: show metrics graph ONLY if server explicitly returned a plot_path (strict)
  if(json.plot_path){
    const gp = json.plot_path.startsWith('/') ? json.plot_path : ('/' + json.plot_path);
    metricGraph.src = gp + (gp.includes('?') ? '&' : '?') + 't=' + Date.now();
    elShow(graphsEl, true);
  } else {
    // do NOT set fallback image here — hide graphs section entirely
    elShow(graphsEl, false);
    if(metricGraph) metricGraph.src = '';
  }

  // Show feedback area
  elShow(feedback, true);
}

// Next / Retry handlers
if(nextQuestionBtn) nextQuestionBtn.addEventListener('click', async ()=>{ await loadQuestion(); });
if(nextBtn) nextBtn.addEventListener('click', async ()=>{ await loadQuestion(); });
if(retryBtn) retryBtn.addEventListener('click', ()=>{ resetUIForRecord(); });

// kick off first question on load
window.addEventListener('load', ()=>{
  loadQuestion();
});
