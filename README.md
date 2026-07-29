# Gesture // Speak — Hand Gesture Recognizer + Face Emotion Overlay

A single-page, pure HTML/CSS/JS app. No Python, no backend, no build step —
runs entirely in the browser using MediaPipe's official JavaScript/WASM
build, and speaks gesture detections aloud using the browser's built-in
Web Speech API.

## 📁 Files

```
gesture_web/
├── index.html   # Structure
├── style.css    # Clean, warm theme (light + dark), Fraunces + Inter + IBM Plex Mono
└── script.js    # Camera, hand tracking, face/emotion tracking, gesture logic, speech, log
```

## ✋ Built-in Gestures

**Single hand:** Hello (open palm), Help (fist), I Agree (thumbs up), Nope
(thumbs down), Peace, Point, OK, I Love You, Call Me 🤙🏻, Pinched Fingers 🤌🏻, 🖕

**Two hands together:** Heart Hands 🫶🏻, Shy 👉🏻👈🏻

Every distance threshold lives in the `THRESH` object at the top of
`script.js`, scaled relative to the hand's own size so detection holds up
regardless of how close the hand is to the camera. Finger-extended state
is measured as a fingertip-to-wrist vs. knuckle-to-wrist ratio rather than
screen position, so it stays accurate even when the hand is tilted.

## 🖐️ Your Own Custom Gestures

No code editing required — in the "Your custom gestures" panel:

1. Type a name, click **Record**, hold the pose steady for about a second.
2. It's saved to your browser's local storage and recognized from then on,
   taking priority over the built-in gestures.
3. **Export** downloads your set as `.json`; **Import** loads someone
   else's file — a way to share gesture packs without any server.

## 🙂 Face Emotion Overlay (on-camera only)

Toggle **"Show face emotion"** in the controls panel. When on, a thin
rounded border is drawn around your face directly on the camera feed,
with the detected emotion in small lowercase letters sitting on the
border itself — `happy`, `sad`, `angry`, `surprised`, or `neutral`.

- This is a **visual-only overlay**: it is never spoken aloud, never
  logged, and never shown anywhere in the side panel — exactly mirroring
  what appears on the camera and nothing more.
- Built on MediaPipe's `FaceLandmarker` with facial blendshapes (52
  expression coefficients — smile, brow position, jaw openness, eye
  width, etc.). Simple threshold rules on those scores decide the label;
  see `classifyEmotion()` in `script.js` if you want to tune the
  thresholds or add more emotion categories.
- Runs independently of hand tracking — turning it off (or leaving the
  camera pointed away from a face) doesn't affect gesture recognition at
  all, and vice versa.

## 🌗 Theme Toggle

The circular button in the top-right corner switches between a light and
dark version of the same palette (same teal/gold accents both ways).
Your choice is remembered via local storage, so it stays consistent on
your next visit.

## 🚀 Run Locally

Browsers block webcam access (`getUserMedia`) on plain `file://` pages, so
serve the folder over local HTTP:

```
cd gesture_web
python3 -m http.server 8000
```

Then open **http://localhost:8000**. The first load downloads the two
small MediaPipe models (hand + face) from Google's CDN, so you'll need
internet the first time.

### 📷 Front / rear camera

If your device has more than one camera, a "Use rear camera" toggle
appears once the camera is running. Switching is instant — no model
reload, just a stream swap. Mirroring auto-adjusts (front mirrors, rear
doesn't) and can still be overridden manually.

### ⚙️ A note on performance

This build runs **two** MediaPipe models per frame (hands + face) instead
of one, so it's noticeably heavier than the hands-only version — expect
higher CPU/GPU/battery use, especially on older phones. If it feels
sluggish on your device, turning off "Show face emotion" removes the
face-tracking model's per-frame cost entirely while leaving gesture
recognition untouched.

## ☁️ Deploy for free (GitHub Pages)

```
cd gesture_web
git init
git add .
git commit -m "Add theme toggle and face emotion overlay"
git branch -M main
git remote add origin https://github.com/Amateur-HEHE/HandGestureSpeaker.git
git push -u origin main
```
Then on GitHub: **Settings → Pages → Source: Deploy from a branch → main
→ / (root) → Save**. Camera and microphone-adjacent APIs require HTTPS,
which GitHub Pages provides automatically.

## 🛠️ Customization

- **Colors:** CSS variables at the top of `style.css` — both the
  `:root` (light) and `:root[data-theme="dark"]` blocks
- **Type:** swap the Google Fonts `<link>` in `index.html` and the
  `font-family` values in `style.css`
- **Gesture sensitivity:** the `THRESH` object in `script.js`
- **Emotion sensitivity:** the threshold numbers inside `classifyEmotion()`
  in `script.js` (e.g. `smile > 0.4`) — lower to make an emotion trigger
  more easily, raise to require a more exaggerated expression
- **Hold time:** `HOLD_FRAMES` (gestures) and `EMOTION_HOLD_FRAMES`
  (emotion label) in `script.js`
- **New built-in gestures:** add a branch to `classifyGesture()` or
  `classifyTwoHandGesture()`

Enjoy! 🚀
