const faceInput        = document.getElementById("faceInput");
const faceImg          = document.getElementById("faceImg");
const facePlaceholder  = document.getElementById("facePlaceholder");
const faceChooseLabel  = document.getElementById("faceChooseLabel");
const faceRemoveBtn    = document.getElementById("faceRemoveBtn");
const faceStatus       = document.getElementById("faceStatus");
const resPreset        = document.getElementById("resPreset");
const fpsPreset        = document.getElementById("fpsPreset");
const startBtn         = document.getElementById("startBtn");
const stopBtn          = document.getElementById("stopBtn");
const sessionStatus    = document.getElementById("sessionStatus");
const obsCard          = document.getElementById("obsCard");
const mjpegUrl         = document.getElementById("mjpegUrl");
const viewerUrl        = document.getElementById("viewerUrl");
const copyMjpeg        = document.getElementById("copyMjpeg");
const copyViewer       = document.getElementById("copyViewer");
const inputVideo       = document.getElementById("inputVideo");
const outputCanvas     = document.getElementById("outputCanvas");
const mFps             = document.getElementById("mFps");
const mSent            = document.getElementById("mSent");
const logBody          = document.getElementById("logBody");
const logCount         = document.getElementById("logCount");
const logClear         = document.getElementById("logClear");

let facePath = null;
let streamSecret = "secret";
let sid = null;
let ws = null;
let mediaStream = null;
let captureTimer = null;
let statusTimer = null;
let metricsTimer = null;
let captureCanvas = null;
let captureCtx = null;
let outCtx = null;
let framesSent = 0;
let framesRecv = 0;
let fpsLastCheck = performance.now();

let lastSendTime = 0;

// per-20s window
let windowStart      = 0;
let windowSentFrames = 0;
let windowSentBytes  = 0;
let windowFramesRecv = 0;
let windowLatencies  = [];
let windowRenderMs   = [];

// session totals
let sessionStart        = 0;
let sessionSentFrames   = 0;
let sessionOutputFrames = 0;
let sessionLatSum       = 0;
let sessionLatCount     = 0;
let sessionRenderSum    = 0;
let sessionRenderCount  = 0;

// ── Activity log (SSE) ────────────────────────────────────────────────────────

let totalLogEntries = 0;

function appendLog(level, ts, msg) {
  totalLogEntries++;
  logCount.textContent = totalLogEntries;
  const row = document.createElement("div");
  row.className = `log-row log-${level}`;
  const tsEl = document.createElement("span");
  tsEl.className = "log-ts";
  tsEl.textContent = ts;
  const msgEl = document.createElement("span");
  msgEl.className = "log-msg";
  msgEl.textContent = msg;
  row.append(tsEl, msgEl);
  logBody.appendChild(row);
  logBody.scrollTop = logBody.scrollHeight;
}

logClear.addEventListener("click", () => {
  logBody.innerHTML = "";
  totalLogEntries = 0;
  logCount.textContent = "0";
});

(function connectLog() {
  const es = new EventSource("/api/log");
  es.onmessage = (e) => {
    const [level, ts, ...rest] = e.data.split("|");
    appendLog(level, ts, rest.join("|"));
  };
  es.onerror = () => { setTimeout(connectLog, 3000); es.close(); };
})();

function localTs() {
  return new Date().toTimeString().slice(0, 8);
}

resPreset.addEventListener("change", () => {
  appendLog("info", localTs(), `Resolution set to ${resPreset.options[resPreset.selectedIndex].text}`);
});

fpsPreset.addEventListener("change", () => {
  appendLog("info", localTs(), `Frame rate set to ${fpsPreset.options[fpsPreset.selectedIndex].text}`);
});

// ── Status panel ──────────────────────────────────────────────────────────────

const STATUS_DOT   = { idle: "dot-idle", loading: "dot-loading", ready: "dot-ready",
                       active: "dot-active", warning: "dot-warning", failed: "dot-failed" };
const STATUS_BADGE = { idle: "—", loading: "Loading…", ready: "Ready",
                       active: "Active", warning: "Warning", failed: "Failed" };

function setRow(rowId, status, subText) {
  const row = document.getElementById(rowId);
  if (!row) return;
  row.querySelector(".status-dot").className = `status-dot ${STATUS_DOT[status] || "dot-idle"}`;
  row.querySelector(".status-badge").textContent = STATUS_BADGE[status] || "—";
  if (subText !== undefined) {
    const sub = row.querySelector(".status-sub");
    if (sub) sub.textContent = subText;
  }
}

function renderServerStatus(data) {
  const c = data.components || {};

  const fd = c.face_detector || {};
  setRow("st-face-detector", fd.status || "idle");

  const sm = c.swap_model || {};
  setRow("st-swap-model", sm.status || "idle", sm.detail || "ONNX");

  const sf = c.source_face || {};
  setRow("st-source-face", sf.status || "idle", sf.detail || "No face loaded");

  const gp = c.gpu_provider || {};
  setRow("st-gpu-provider", gp.status || "idle", gp.detail || "—");
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    renderServerStatus(data);
  } catch (_) {}
}

function startStatusPolling() {
  pollStatus();
  statusTimer = setInterval(pollStatus, 2000);
}

function stopStatusPolling() {
  clearInterval(statusTimer);
  statusTimer = null;
}

startStatusPolling();

// ── Face image upload ─────────────────────────────────────────────────────────

faceInput.addEventListener("change", async () => {
  const file = faceInput.files[0];
  if (!file) return;
  faceStatus.textContent = "Uploading…";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/uploadImage", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    facePath = data.path;
    faceImg.src = data.url;
    faceImg.hidden = false;
    facePlaceholder.hidden = true;
    faceStatus.textContent = "Ready";
    startBtn.disabled = false;
    faceRemoveBtn.hidden = false;
  } catch (e) {
    faceStatus.textContent = "Upload failed";
    console.error(e);
  }
});

faceRemoveBtn.addEventListener("click", () => {
  facePath = null;
  faceImg.src = "";
  faceImg.hidden = true;
  facePlaceholder.hidden = false;
  faceStatus.textContent = "";
  faceInput.value = "";
  startBtn.disabled = true;
  faceRemoveBtn.hidden = true;
});

// ── Session lifecycle ─────────────────────────────────────────────────────────

startBtn.addEventListener("click", startSession);
stopBtn.addEventListener("click", stopSession);

async function startSession() {
  if (!facePath) return;
  startBtn.disabled = true;
  resPreset.disabled = true;
  fpsPreset.disabled = true;
  sessionStatus.textContent = "Loading models… (first run may take ~30s)";
  try {
    const res = await fetch("/api/session/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ face_path: facePath, max_swap_fps: parseInt(fpsPreset.value, 10) }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    sid = data.sid;
    streamSecret = data.stream_secret || streamSecret;
    faceRemoveBtn.disabled = true;

    await pollStatus();

    sessionStatus.textContent = "Session active";
    startBtn.hidden = true;
    stopBtn.hidden = false;

    const mjpeg  = `${location.origin}/stream.mjpeg?sid=${sid}&key=${streamSecret}`;
    const viewer = `${location.origin}/viewer.html?sid=${sid}`;
    mjpegUrl.textContent  = mjpeg;
    viewerUrl.textContent = viewer;
    obsCard.hidden = false;
    copyMjpeg.onclick  = () => navigator.clipboard.writeText(mjpeg);
    copyViewer.onclick = () => navigator.clipboard.writeText(viewer);

    await startWebcam();
    openSwapSocket();
  } catch (e) {
    sessionStatus.textContent = `Error: ${e.message}`;
    startBtn.disabled = false;
    startBtn.hidden = false;
    resPreset.disabled = false;
    fpsPreset.disabled = false;
    faceRemoveBtn.disabled = false;
  }
}

async function stopSession() {
  logSessionSummary();
  cleanup();
  if (sid) {
    await fetch("/api/session/close", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid }),
    }).catch(() => {});
    sid = null;
  }
  sessionStatus.textContent = "Stopped";
  startBtn.hidden = false;
  startBtn.disabled = false;
  stopBtn.hidden = true;
  obsCard.hidden = true;
  resPreset.disabled = false;
  fpsPreset.disabled = false;
  faceRemoveBtn.disabled = false;
  mFps.textContent = "—";
  mSent.textContent = "—";
  setRow("st-webcam", "idle", "—");
  setRow("st-swap-ws", "idle");
  await pollStatus();
}

function cleanup() {
  clearInterval(captureTimer);
  captureTimer = null;
  clearInterval(metricsTimer);
  metricsTimer = null;
  if (ws) { ws.close(); ws = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  inputVideo.srcObject = null;
  framesSent = 0;
  framesRecv = 0;
  lastSendTime = 0;
}

// ── Webcam ────────────────────────────────────────────────────────────────────

function getResConstraints() {
  const [w, h] = resPreset.value.split("x").map(Number);
  return { width: { ideal: w }, height: { ideal: h } };
}

function getSendInterval() {
  const fps = parseInt(fpsPreset.value, 10);
  return fps === 0 ? 16 : 1000 / fps;
}

async function startWebcam() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { ...getResConstraints(), frameRate: { ideal: parseInt(fpsPreset.value, 10) } },
    audio: false,
  });
  inputVideo.srcObject = mediaStream;
  await new Promise(r => { inputVideo.onloadedmetadata = r; });

  const w = inputVideo.videoWidth;
  const h = inputVideo.videoHeight;
  const fpsLabel = fpsPreset.value === "0" ? "MAX fps" : `${fpsPreset.value} fps`;
  setRow("st-webcam", "active", `${w}×${h} @ ${fpsLabel}`);

  captureCanvas = document.createElement("canvas");
  captureCanvas.width = w;
  captureCanvas.height = h;
  captureCtx = captureCanvas.getContext("2d");

  outputCanvas.width = w;
  outputCanvas.height = h;
  outCtx = outputCanvas.getContext("2d");
}

// ── WebSocket swap ────────────────────────────────────────────────────────────

function openSwapSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/swap?sid=${sid}`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    sessionStart        = performance.now();
    windowStart         = performance.now();
    windowSentFrames    = 0;
    windowSentBytes     = 0;
    windowFramesRecv    = 0;
    windowLatencies     = [];
    windowRenderMs      = [];
    sessionSentFrames   = 0;
    sessionOutputFrames = 0;
    sessionLatSum       = 0;
    sessionLatCount     = 0;
    sessionRenderSum    = 0;
    sessionRenderCount  = 0;
    setRow("st-swap-ws", "active");
    captureTimer = setInterval(sendFrame, getSendInterval());
    metricsTimer = setInterval(logPeriodicMetrics, 20000);
    const resLabel = resPreset.options[resPreset.selectedIndex].text;
    const fpsTxt   = fpsPreset.options[fpsPreset.selectedIndex].text;
    appendLog("info", localTs(), `Stream started — ${resLabel} | ${fpsTxt}`);
  };

  ws.onmessage = (e) => {
    const recvTime = performance.now();
    framesRecv++;
    windowFramesRecv++;
    sessionOutputFrames++;
    if (lastSendTime > 0) {
      const lat = recvTime - lastSendTime;
      windowLatencies.push(lat);
      sessionLatSum   += lat;
      sessionLatCount++;
    }
    const blob = new Blob([e.data], { type: "image/jpeg" });
    const renderStart = performance.now();
    createImageBitmap(blob).then(bitmap => {
      outCtx.drawImage(bitmap, 0, 0, outputCanvas.width, outputCanvas.height);
      const renderMs = performance.now() - renderStart;
      windowRenderMs.push(renderMs);
      sessionRenderSum   += renderMs;
      sessionRenderCount++;
      bitmap.close();
      tickMetrics();
    });
  };

  ws.onerror = () => {
    setRow("st-swap-ws", "failed");
    sessionStatus.textContent = "Connection error";
  };
  ws.onclose = () => {
    setRow("st-swap-ws", "idle");
    clearInterval(captureTimer);
    captureTimer = null;
  };
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !captureCtx) return;
  captureCtx.drawImage(inputVideo, 0, 0, captureCanvas.width, captureCanvas.height);
  captureCanvas.toBlob(blob => {
    if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
    blob.arrayBuffer().then(buf => {
      windowSentFrames++;
      windowSentBytes += buf.byteLength;
      sessionSentFrames++;
      lastSendTime = performance.now();
      ws.send(buf);
      framesSent++;
    });
  }, "image/jpeg", 0.85);
}

// ── Metrics ───────────────────────────────────────────────────────────────────

function tickMetrics() {
  const now = performance.now();
  const elapsed = (now - fpsLastCheck) / 1000;
  if (elapsed >= 1) {
    mFps.textContent = (framesRecv / elapsed).toFixed(1);
    framesRecv = 0;
    fpsLastCheck = now;
  }
  mSent.textContent = framesSent;
}

function avgMs(arr) {
  return arr.length > 0 ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
}

function logPeriodicMetrics() {
  const elapsed    = (performance.now() - windowStart) / 1000;
  const sendFps    = elapsed > 0 ? (windowSentFrames / elapsed).toFixed(1) : "0.0";
  const kbps       = elapsed > 0 ? ((windowSentBytes / 1024) / elapsed).toFixed(0) : "0";
  const recvFps    = elapsed > 0 ? (windowFramesRecv / elapsed).toFixed(1) : "0.0";
  const avgLat     = avgMs(windowLatencies);
  const avgRender  = avgMs(windowRenderMs);
  appendLog("info", localTs(),
    `[Browser]  Send: ${sendFps} fps (${kbps} KB/s)  →  Output: ${recvFps} fps  |  RTT: ${avgLat} ms  |  Render: ${avgRender} ms`);
  windowStart      = performance.now();
  windowSentFrames = 0;
  windowSentBytes  = 0;
  windowFramesRecv = 0;
  windowLatencies  = [];
  windowRenderMs   = [];
}

function logSessionSummary() {
  if (sessionStart === 0) return;
  const elapsed    = (performance.now() - sessionStart) / 1000;
  const avgSend    = elapsed > 0 ? (sessionSentFrames   / elapsed).toFixed(1) : "0.0";
  const avgRecv    = elapsed > 0 ? (sessionOutputFrames / elapsed).toFixed(1) : "0.0";
  const avgLat     = sessionLatCount    > 0 ? Math.round(sessionLatSum    / sessionLatCount)    : 0;
  const avgRender  = sessionRenderCount > 0 ? Math.round(sessionRenderSum / sessionRenderCount) : 0;
  appendLog("info", localTs(),
    `[Session]  Send avg: ${avgSend} fps  →  Output avg: ${avgRecv} fps  |  Avg RTT: ${avgLat} ms  |  Avg Render: ${avgRender} ms  |  Duration: ${elapsed.toFixed(0)}s`);
  sessionStart = 0;
}
