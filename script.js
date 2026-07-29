// ==========================================================================
// GESTURE // SPEAK — runs entirely in the browser. No backend.
// Hand gestures via MediaPipe HandLandmarker, face emotion via
// MediaPipe FaceLandmarker (blendshapes), speech via Web Speech API.
// ==========================================================================

import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// ---------- Tunable thresholds ----------
const THRESH = {
  pinchOK: 0.55,
  pinchAll: 0.8,
  extendRatio: 1.25,
  thumbExtendRatio: 1.2,
  thumbUpDown: 0.5,
  fistTightRatio: 1.15,
  twoHandPinch: 0.8,
  twoHandPoint: 1.2,
  customMatch: 0.28,
};

const HOLD_FRAMES = 4;       // frames a gesture must hold before it's spoken/logged
const EMOTION_HOLD_FRAMES = 3; // frames an emotion must hold before the label updates

// ---------- DOM ----------
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const viewfinder = document.getElementById("viewfinder");
const viewfinderMessage = document.getElementById("viewfinderMessage");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const gestureReadout = document.getElementById("gestureReadout");
const cornerReadout = document.getElementById("cornerReadout");
const voiceToggle = document.getElementById("voiceToggle");
const mirrorToggle = document.getElementById("mirrorToggle");
const cameraToggle = document.getElementById("cameraToggle");
const emotionToggle = document.getElementById("emotionToggle");
const logList = document.getElementById("logList");
const customNameInput = document.getElementById("customNameInput");
const recordBtn = document.getElementById("recordBtn");
const recordStatus = document.getElementById("recordStatus");
const customList = document.getElementById("customList");
const exportBtn = document.getElementById("exportBtn");
const importInput = document.getElementById("importInput");
const themeToggle = document.getElementById("themeToggle");

// ---------- Theme ----------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeToggle.querySelector(".theme-icon").textContent = theme === "dark" ? "☀️" : "🌙";
  themeToggle.setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
  );
  localStorage.setItem("gestureSpeak.theme", theme);
}
applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
});

// ---------- Hand topology (for drawing) ----------
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const MCP = { index: 5, middle: 9, ring: 13, pinky: 17 };

// ---------- Geometry helpers ----------
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function handScale(landmarks) {
  return dist(landmarks[0], landmarks[9]) || 1;
}

// ---------- Finger state (orientation-invariant) ----------
function fingerStates(landmarks) {
  const wrist = landmarks[0];
  const s = {};
  for (const f of ["index", "middle", "ring", "pinky"]) {
    const tipDist = dist(landmarks[TIP[f]], wrist);
    const mcpDist = dist(landmarks[MCP[f]], wrist);
    s[f] = tipDist > mcpDist * THRESH.extendRatio;
  }
  const palmRef = landmarks[17];
  const thumbTipDist = dist(landmarks[4], palmRef);
  const thumbBaseDist = dist(landmarks[2], palmRef);
  s.thumb = thumbTipDist > thumbBaseDist * THRESH.thumbExtendRatio;
  return s;
}

function isTightFist(landmarks) {
  const wrist = landmarks[0];
  let tipSum = 0;
  let mcpSum = 0;
  for (const f of ["index", "middle", "ring", "pinky"]) {
    tipSum += dist(landmarks[TIP[f]], wrist);
    mcpSum += dist(landmarks[MCP[f]], wrist);
  }
  return tipSum < mcpSum * THRESH.fistTightRatio;
}

// ---------- Single-hand gesture classification ----------
function classifyGesture(landmarks) {
  const scale = handScale(landmarks);
  const f = fingerStates(landmarks);
  const { thumb, index, middle, ring, pinky } = f;

  const thumbTipY = landmarks[4].y;
  const thumbMcpY = landmarks[2].y;

  if (dist(landmarks[4], landmarks[8]) < THRESH.pinchOK * scale && middle && ring && pinky) {
    return "OK 👌";
  }

  const tips = ["thumb", "index", "middle", "ring", "pinky"].map((n) => landmarks[TIP[n]]);
  let maxSpread = 0;
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      maxSpread = Math.max(maxSpread, dist(tips[i], tips[j]));
    }
  }
  const avgTipY = tips.reduce((sum, t) => sum + t.y, 0) / 5;
  if (maxSpread < THRESH.pinchAll * scale && avgTipY < landmarks[0].y) {
    return "Pinched Fingers 🤌🏻";
  }

  if (!index && !middle && !ring && !pinky) {
    if (!thumb) {
      return isTightFist(landmarks) ? "Help ✊" : "Unknown";
    }
    if (thumbTipY < thumbMcpY - THRESH.thumbUpDown * scale) return "I Agree 👍";
    if (thumbTipY > thumbMcpY + THRESH.thumbUpDown * scale) return "Nope 👎";
    return "Unknown";
  }

  if (thumb && pinky && !index && !middle && !ring) return "Call Me 🤙🏻";
  if (middle && !thumb && !index && !ring && !pinky) return "🖕";
  if (index && middle && !ring && !pinky && !thumb) return "Peace ✌️";
  if (thumb && index && pinky && !middle && !ring) return "I Love You 🤟";
  if (index && !thumb && !middle && !ring && !pinky) return "Point ☝️";
  if (thumb && index && middle && ring && pinky) return "Hello ✋";

  return "Unknown";
}

// ---------- Two-hand gesture classification ----------
function classifyTwoHandGesture(handsLandmarks) {
  if (handsLandmarks.length !== 2) return null;

  const [lm1, lm2] = handsLandmarks;
  const f1 = fingerStates(lm1);
  const f2 = fingerStates(lm2);
  const avgScale = (handScale(lm1) + handScale(lm2)) / 2;

  const thumbDist = dist(lm1[TIP.thumb], lm2[TIP.thumb]);
  const indexDist = dist(lm1[TIP.index], lm2[TIP.index]);

  if (
    thumbDist < THRESH.twoHandPinch * avgScale && indexDist < THRESH.twoHandPinch * avgScale &&
    !f1.middle && !f1.ring && !f1.pinky &&
    !f2.middle && !f2.ring && !f2.pinky
  ) {
    return "Heart Hands 🫶🏻";
  }

  const isPoint = (f) => f.index && !f.thumb && !f.middle && !f.ring && !f.pinky;
  if (isPoint(f1) && isPoint(f2) && indexDist < THRESH.twoHandPoint * avgScale) {
    return "Shy 👉🏻👈🏻";
  }

  return null;
}

// ---------- Custom gestures ----------
const CUSTOM_KEY = "gestureSpeak.customGestures";

function loadCustomGestures() {
  try {
    return JSON.parse(localStorage.getItem(CUSTOM_KEY)) || [];
  } catch {
    return [];
  }
}

function saveCustomGestures() {
  localStorage.setItem(CUSTOM_KEY, JSON.stringify(customGestures));
}

let customGestures = loadCustomGestures();

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const scale = handScale(landmarks);
  return landmarks.map((lm) => ({
    x: (lm.x - wrist.x) / scale,
    y: (lm.y - wrist.y) / scale,
  }));
}

function vectorDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
  }
  return sum / a.length;
}

function matchCustomGesture(landmarks) {
  if (!customGestures.length) return null;
  const live = normalizeLandmarks(landmarks);
  let best = null;
  let bestDist = Infinity;
  for (const g of customGestures) {
    const d = vectorDistance(live, g.vector);
    if (d < bestDist) {
      bestDist = d;
      best = g;
    }
  }
  return bestDist < THRESH.customMatch ? best.label : null;
}

function renderCustomList() {
  customList.innerHTML = "";
  if (!customGestures.length) {
    const li = document.createElement("li");
    li.className = "empty-hint";
    li.textContent = "No custom gestures yet — hold a pose and name it above.";
    customList.appendChild(li);
    return;
  }
  customGestures.forEach((g, i) => {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = g.label;
    const del = document.createElement("button");
    del.className = "delete-btn";
    del.setAttribute("aria-label", `Remove ${g.label}`);
    del.textContent = "✕";
    del.addEventListener("click", () => {
      customGestures.splice(i, 1);
      saveCustomGestures();
      renderCustomList();
    });
    li.appendChild(span);
    li.appendChild(del);
    customList.appendChild(li);
  });
}

// ---------- Recording flow ----------
let recording = null;

function startRecording(name) {
  recording = { name, samples: [] };
  recordStatus.textContent = `Hold the pose still for "${name}"…`;
  recordBtn.disabled = true;
  customNameInput.disabled = true;
}

function captureRecordingFrame(landmarks) {
  if (!recording) return;
  recording.samples.push(normalizeLandmarks(landmarks));
  recordStatus.textContent = `Capturing "${recording.name}"… ${recording.samples.length}/20`;
  if (recording.samples.length >= 20) {
    const avg = recording.samples[0].map((_, i) => {
      const xs = recording.samples.map((s) => s[i].x);
      const ys = recording.samples.map((s) => s[i].y);
      return {
        x: xs.reduce((a, b) => a + b, 0) / xs.length,
        y: ys.reduce((a, b) => a + b, 0) / ys.length,
      };
    });
    customGestures.push({ label: recording.name, vector: avg });
    saveCustomGestures();
    renderCustomList();
    recordStatus.textContent = `Saved "${recording.name}".`;
    recording = null;
    recordBtn.disabled = false;
    customNameInput.disabled = false;
    customNameInput.value = "";
    setTimeout(() => {
      if (!recording) recordStatus.textContent = "";
    }, 2500);
  }
}

// ---------- Face emotion classification (from blendshape scores) ----------
function classifyEmotion(categories) {
  const map = {};
  for (const c of categories) map[c.categoryName] = c.score;

  const smile = ((map.mouthSmileLeft || 0) + (map.mouthSmileRight || 0)) / 2;
  const frown = ((map.mouthFrownLeft || 0) + (map.mouthFrownRight || 0)) / 2;
  const browDown = ((map.browDownLeft || 0) + (map.browDownRight || 0)) / 2;
  const browInnerUp = map.browInnerUp || 0;
  const jawOpen = map.jawOpen || 0;
  const eyeWide = ((map.eyeWideLeft || 0) + (map.eyeWideRight || 0)) / 2;

  if (jawOpen > 0.4 && eyeWide > 0.3) return "surprised";
  if (smile > 0.4) return "happy";
  if (frown > 0.3 && browInnerUp > 0.3) return "sad";
  if (browDown > 0.4 && frown > 0.15) return "angry";
  return "neutral";
}

// ---------- Drawing ----------
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHandLandmarks(handsLandmarks) {
  for (const landmarks of handsLandmarks) {
    const pts = landmarks.map((lm) => [lm.x * canvas.width, lm.y * canvas.height]);

    ctx.strokeStyle = "rgba(47, 111, 94, 0.8)";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }

    ctx.fillStyle = "#C98A3D";
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// Thin rounded-rect border around the face, with the emotion word sitting
// in small letters directly on the border line. Drawn only on the canvas
// overlay — never added to the side panel or spoken aloud.
function drawFaceEmotionBox(faceLandmarks, emotionLabel) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const lm of faceLandmarks) {
    const x = lm.x * canvas.width;
    const y = lm.y * canvas.height;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = (maxX - minX) * 0.08;
  const x = minX - pad;
  const y = minY - pad * 1.4;
  const w = (maxX - minX) + pad * 2;
  const h = (maxY - minY) + pad * 2.2;
  const radius = 14;

  ctx.save();
  ctx.strokeStyle = "rgba(95, 217, 196, 0.85)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
  ctx.stroke();

  // small label sitting on the border, top-left
  ctx.font = "11px 'IBM Plex Mono', monospace";
  const textWidth = ctx.measureText(emotionLabel).width;
  const labelX = x + 14;
  const labelY = y;

  ctx.fillStyle = "rgba(17, 20, 22, 0.85)";
  ctx.fillRect(labelX - 4, labelY - 8, textWidth + 8, 14);

  ctx.fillStyle = "#5FD9C4";
  ctx.textBaseline = "middle";
  ctx.fillText(emotionLabel, labelX, labelY - 1);
  ctx.restore();
}

// ---------- Speech ----------
let lastSpoken = null;
function speak(label) {
  if (!voiceToggle.checked) return;
  const spokenText = label.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\uFE0F\u{1F3FB}-\u{1F3FF}]/gu, "").trim();
  if (!spokenText) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(spokenText);
  utter.rate = 1.0;
  window.speechSynthesis.speak(utter);
}

// ---------- Log ----------
function logDetection(label) {
  const time = new Date().toLocaleTimeString([], { hour12: false });
  const li = document.createElement("li");
  const timeSpan = document.createElement("span");
  timeSpan.className = "t";
  timeSpan.textContent = time;
  li.appendChild(timeSpan);
  li.appendChild(document.createTextNode(label));
  logList.appendChild(li);
  while (logList.children.length > 12) {
    logList.removeChild(logList.firstChild);
  }
  logList.scrollTop = logList.scrollHeight;
}

// ---------- Stability smoothing ----------
let candidateLabel = null;
let candidateCount = 0;

function stableGesture(rawLabel) {
  if (rawLabel === candidateLabel) {
    candidateCount++;
  } else {
    candidateLabel = rawLabel;
    candidateCount = 1;
  }
  return candidateCount >= HOLD_FRAMES ? candidateLabel : null;
}

let emotionCandidate = null;
let emotionCandidateCount = 0;
let stableEmotionLabel = "neutral";

function stableEmotion(rawLabel) {
  if (rawLabel === emotionCandidate) {
    emotionCandidateCount++;
  } else {
    emotionCandidate = rawLabel;
    emotionCandidateCount = 1;
  }
  if (emotionCandidateCount >= EMOTION_HOLD_FRAMES) {
    stableEmotionLabel = emotionCandidate;
  }
  return stableEmotionLabel;
}

// ---------- Camera + model lifecycle ----------
let handLandmarker = null;
let faceLandmarker = null;
let running = false;
let currentFacingMode = "user";
let rafId = null;

async function initModels() {
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);

  async function createWithFallback(Ctor, modelUrl, extraOptions) {
    const common = {
      baseOptions: { modelAssetPath: modelUrl },
      runningMode: "VIDEO",
      ...extraOptions,
    };
    try {
      return await Ctor.createFromOptions(filesetResolver, {
        ...common,
        baseOptions: { ...common.baseOptions, delegate: "GPU" },
      });
    } catch (err) {
      console.warn("GPU delegate failed, falling back to CPU:", err);
      return await Ctor.createFromOptions(filesetResolver, {
        ...common,
        baseOptions: { ...common.baseOptions, delegate: "CPU" },
      });
    }
  }

  if (!handLandmarker) {
    handLandmarker = await createWithFallback(HandLandmarker, HAND_MODEL_URL, {
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
  }
  if (!faceLandmarker) {
    faceLandmarker = await createWithFallback(FaceLandmarker, FACE_MODEL_URL, {
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    });
  }
}

async function startCamera(facingMode = currentFacingMode) {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720, facingMode: { ideal: facingMode } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  currentFacingMode = facingMode;
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  clearCanvas();
  window.speechSynthesis.cancel();
  lastSpoken = null;
  candidateLabel = null;
  candidateCount = 0;
  emotionCandidate = null;
  emotionCandidateCount = 0;
  stableEmotionLabel = "neutral";
  recording = null;
  recordStatus.textContent = "";
  recordBtn.disabled = true;
  customNameInput.disabled = true;
  setViewfinderState("idle");
  viewfinderMessage.classList.remove("hidden");
  viewfinderMessage.querySelector("p").textContent = "Camera is off";
  startBtn.disabled = false;
  startBtn.textContent = "Turn on camera";
  stopBtn.disabled = true;
  gestureReadout.textContent = "— · —";
  cornerReadout.textContent = "Hands: 0";
}

function setViewfinderState(state) {
  viewfinder.dataset.state = state;
}

function predictLoop() {
  if (!running) return;

  const now = performance.now();
  const handResult = handLandmarker.detectForVideo(video, now);
  const faceResult = emotionToggle.checked ? faceLandmarker.detectForVideo(video, now) : null;

  clearCanvas();

  let gestureText = "No hand detected";
  let handCount = 0;

  if (handResult.landmarks && handResult.landmarks.length > 0) {
    handCount = handResult.landmarks.length;
    drawHandLandmarks(handResult.landmarks);

    if (recording) {
      captureRecordingFrame(handResult.landmarks[0]);
    }

    const twoHandLabel = classifyTwoHandGesture(handResult.landmarks);

    if (twoHandLabel) {
      gestureText = twoHandLabel;
    } else if (handCount === 2) {
      const g1 = matchCustomGesture(handResult.landmarks[0]) || classifyGesture(handResult.landmarks[0]);
      const g2 = matchCustomGesture(handResult.landmarks[1]) || classifyGesture(handResult.landmarks[1]);
      gestureText = g1 === g2 ? g1 : `${g1} / ${g2}`;
    } else {
      gestureText = matchCustomGesture(handResult.landmarks[0]) || classifyGesture(handResult.landmarks[0]);
    }
  }

  if (
    faceResult &&
    faceResult.faceLandmarks &&
    faceResult.faceLandmarks.length > 0 &&
    faceResult.faceBlendshapes &&
    faceResult.faceBlendshapes.length > 0
  ) {
    const rawEmotion = classifyEmotion(faceResult.faceBlendshapes[0].categories);
    const emotion = stableEmotion(rawEmotion);
    drawFaceEmotionBox(faceResult.faceLandmarks[0], emotion);
  }

  cornerReadout.textContent = `Hands: ${handCount}`;

  if (recording) {
    rafId = requestAnimationFrame(predictLoop);
    return;
  }

  const stable = stableGesture(gestureText);

  if (stable) {
    gestureReadout.textContent = stable;
    if (stable !== "No hand detected" && stable !== "Unknown") {
      setViewfinderState("locked");
      if (stable !== lastSpoken) {
        speak(stable);
        logDetection(stable);
        lastSpoken = stable;
      }
    } else {
      setViewfinderState("scanning");
      lastSpoken = null;
    }
  }

  rafId = requestAnimationFrame(predictLoop);
}

// ---------- Boot ----------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";
  let modelsReady = false;
  let cameraReady = false;
  try {
    await initModels();
    modelsReady = true;
    await startCamera(currentFacingMode);
    cameraReady = true;

    viewfinderMessage.classList.add("hidden");
    running = true;
    setViewfinderState("scanning");
    stopBtn.disabled = false;
    recordBtn.disabled = false;
    customNameInput.disabled = false;
    rafId = requestAnimationFrame(predictLoop);

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === "videoinput");
      cameraToggle.disabled = videoInputs.length < 2;
    } catch {
      cameraToggle.disabled = true;
    }
  } catch (err) {
    console.error(err);
    if (cameraReady && video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    startBtn.disabled = false;
    startBtn.textContent = "Turn on camera";
    viewfinderMessage.querySelector("p").textContent =
      modelsReady
        ? "Camera access denied or unavailable — check browser permissions"
        : "Could not load the recognition models — check your connection";
  }
});

stopBtn.addEventListener("click", stopCamera);

mirrorToggle.addEventListener("change", () => {
  viewfinder.dataset.mirror = mirrorToggle.checked ? "true" : "false";
});
viewfinder.dataset.mirror = "true";

voiceToggle.addEventListener("change", () => {
  if (!voiceToggle.checked) window.speechSynthesis.cancel();
});

emotionToggle.addEventListener("change", () => {
  if (!emotionToggle.checked) {
    emotionCandidate = null;
    emotionCandidateCount = 0;
    stableEmotionLabel = "neutral";
  }
});

cameraToggle.addEventListener("change", async () => {
  if (!running) return;
  const nextMode = cameraToggle.checked ? "environment" : "user";
  cameraToggle.disabled = true;
  try {
    await startCamera(nextMode);
    mirrorToggle.checked = nextMode === "user";
    viewfinder.dataset.mirror = mirrorToggle.checked ? "true" : "false";
  } catch (err) {
    console.error(err);
    cameraToggle.checked = !cameraToggle.checked;
  } finally {
    cameraToggle.disabled = false;
  }
});

recordBtn.addEventListener("click", () => {
  const name = customNameInput.value.trim();
  if (!name) {
    recordStatus.textContent = "Give the gesture a name first.";
    return;
  }
  if (!running) {
    recordStatus.textContent = "Turn on the camera first.";
    return;
  }
  startRecording(name);
});

exportBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(customGestures, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "my-gestures.json";
  a.click();
  URL.revokeObjectURL(url);
});

importInput.addEventListener("change", () => {
  const file = importInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const imported = JSON.parse(reader.result);
      if (!Array.isArray(imported)) throw new Error("not an array");
      customGestures = customGestures.concat(imported);
      saveCustomGestures();
      renderCustomList();
      recordStatus.textContent = `Imported ${imported.length} gesture(s).`;
    } catch (err) {
      console.error(err);
      recordStatus.textContent = "That file doesn't look like a valid gesture export.";
    }
  };
  reader.readAsText(file);
  importInput.value = "";
});

// initial render
renderCustomList();
