import { FaceLandmarker, FaceDetector, HandLandmarker, ObjectDetector, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
import { processGaze, drawIris, resetCalibration,
         LI_C,LI_T,LI_R,LI_B,LI_L,
         RI_C,RI_T,RI_R,RI_B,RI_L } from './gaze.js'

const PW = 640, PH = 360

// ── Capture config ────────────────────────────────────────────────────────
const API_ENDPOINT        = 'http://localhost:8080/incident'
const CAPTURE_TOTAL       = 6
const CAPTURE_GAP_MS      = 1000
const CAPTURE_COOLDOWN_MS = 3000    // 3s between captures — an incident can re-trigger 3s after the last (background AI retry + provisional verdicts absorb the extra LLM load)

// ── Layer 1: real-time visible-phone detector (EfficientDet-Lite0, CPU) ─────
const PHONE_MIN_SCORE   = 0.42   // ObjectDetector confidence to count a "cell phone" (lowered: a phone held in selfie/angled pose often scores 0.4–0.5; same-spot debounce + LLM confirmation keep false positives down)
const PHONE_STRONG_SCORE = 0.62  // a SINGLE high-confidence sighting is enough to fire — so a phone flashed for <½s (which may only be caught on one detector run) still triggers a capture
const PHONE_BANNER_HOLD = 1500   // ms to keep the WARNING BANNER after last sighting (text debounce only)
const PHONE_FIRE_STREAK = 2      // consecutive detections before firing a capture (lower-confidence path)
const PHONE_SAME_SPOT_TOL  = 0.06    // box-centre within 6% of frame ⇒ the SAME fixed object (e.g. a phone pictured on a poster)
const PHONE_SAME_SPOT_HOLD = 120000  // ms: once judged, don't re-fire on a phone that never moves — saves LLM quota, kills dashboard spam

// ── Layer 2b: framing manipulation — camera/lid tilted so the torso & hands leave frame ──
const FRAMING_CHIN_Y     = 0.82   // chin landmark in the bottom 18% ⇒ no room below for shoulders/hands
const FRAMING_SUSTAIN_MS = 2500   // must persist this long before flagging (ignores brief reaching/adjusting)
const FRAMING_REARM_MS   = 60000  // don't re-fire a framing capture more often than this

// ── Adaptive phone-detector cadence — guarantees the loop never starves below the FPS floor ──
const FPS_FLOOR          = 40     // if smoothed FPS dips to/under this, the phone detector backs off (loop is capped at 48, so >40 = healthy)

// ── Layer 2: covert-device detector (phone NOT in frame) ────────────────────
// A phone in the lap/under the desk has ZERO pixels to detect — it can only be
// inferred from BEHAVIOUR: a head clearly tilted DOWN (not just eyes glancing at
// a lower screen) that repeats, or stares for an abnormally long time. We score
// these episodes over a rolling window and raise a NEEDS-REVIEW flag (never an
// auto-accusation — we cannot see the device, so a human decides).
const COVERT_PITCH_DEG   = 30     // head pitched-down beyond this = looking at lap/desk (raised: normal desk/laptop posture tilts the head ~15-20°, which must NOT trigger)
const COVERT_MIN_EPISODE = 1200   // ms — a sustained down-look, not a glance/flick
const COVERT_WINDOW_MS   = 30000  // rolling observation window
const COVERT_MIN_GLANCES = 4      // repeated head-down glances in the window → suspect
const COVERT_LONG_STARE  = 6000   // one continuous head-down longer than this → suspect

// ── FPS cap ─────────────────────────────────────────────────────────────
// 45 FPS is the REQUIRED minimum. Running flat-out at 60-70 wastes CPU and
// triggers thermal throttling (which is what caused the sub-30 crashes), so we
// cap — but at 48, not 45: the cap is the loop's AVERAGE, and second-to-second
// jitter of ±1-2 under a 45 cap reads 43-44 on the counter. 48 keeps the
// counter at 45+ at all times while still shedding ~20% load vs 60.
const TARGET_FPS   = 48
const MIN_FRAME_MS = 1000 / TARGET_FPS   // ~20.8ms
let   _lastProcessT = 0

// ── DOM ───────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('main-canvas')
const ctx         = canvas.getContext('2d')
const video       = document.getElementById('video')
const gazeEl      = document.getElementById('gaze-text')
const fpsEl       = document.getElementById('fps-text')
const faceEl      = document.getElementById('face-text')
const handEl      = document.getElementById('hand-text')
const statusEl    = document.getElementById('status-text')
const handBanner  = document.getElementById('hand-banner')
const faceBanner  = document.getElementById('face-banner')
const phoneBanner = document.getElementById('phone-banner')
const covertBanner= document.getElementById('covert-banner')
const framingBanner = document.getElementById('framing-banner')
const loadOverlay = document.getElementById('loading-overlay')
const loadMsg     = document.getElementById('loading-msg')
const loadSub     = document.getElementById('loading-sub')
const errOverlay  = document.getElementById('error-overlay')
const errMsg      = document.getElementById('error-msg')

// ── State ─────────────────────────────────────────────────────────────────
let faceLandmarker = null
let faceDetector   = null
let handLandmarker = null
let objectDetector = null    // EfficientDet-Lite0 — real-time phone detection (CPU)
let fpsAcc = 0, fpsT0 = performance.now()
let _emaFps = 60   // smoothed FPS — drives the adaptive phone-detector cadence
let _debug = false
let _frameN = 0
let _lastGs         = null
let _lastDetections = []
let _lastHands      = { landmarks: [], handedness: [] }
let _primaryDetIdx  = 0

// ── Phone-detector state (Layer 1) ──────────────────────────────────────────
let _phoneVisible  = false   // phone seen within the debounce hold
let _phoneScore    = 0       // confidence of the last sighting
let _phoneBox      = null    // {originX,originY,width,height} in video px — for drawing
let _phoneLastSeen = 0       // performance.now() of last sighting
let _phoneStreak   = 0       // consecutive detector runs that saw a phone
let _phoneBoxFresh = false   // true ONLY on frames the detector actually saw a phone — drives the box (no ghosting)
let _lastFiredPhoneCenter = null  // {x,y} normalised centre of the last phone that fired a capture
let _lastFiredPhoneAt     = 0     // when it fired — for the same-spot re-fire hold

// ── Covert-device state (Layer 2) ───────────────────────────────────────────
let _downStart      = 0      // when the current continuous head-down episode began (0 = not down)
let _covertGlances  = []     // timestamps of completed head-down episodes (pruned to the window)

// ── Framing-manipulation state (Layer 2b) ───────────────────────────────────
let _framingSince   = 0      // when the torso-hidden condition began (0 = torso visible)
let _framingHidden  = false  // sustained torso-hidden flag (drives the banner + trigger)
let _framingFiredAt = -FRAMING_REARM_MS   // last framing capture time (re-arm guard); seeded negative so the FIRST tilt can fire immediately

// ── Intruder / face-count state ───────────────────────────────────────────
let _lastFaceCount   = 1     // total faces in frame (candidate + intruders), set each render
let _intruderSince   = 0     // timestamp when 2+ faces first seen (sustain debounce)
let _captureMaxFaces = 1     // peak face count observed during the current capture window
let _captureReason   = 'hand_visible_and_gaze_toward_wrist'  // what fired this capture

// ── Capture state ─────────────────────────────────────────────────────────
let _captureActive        = false
let _captureCanvases      = []     // raw canvas per frame — used to build collage
let _captureFrames        = []     // base64 JPEG per frame — sent to API
let _captureGazeLabels    = []     // gaze direction string per frame — written on collage
let _captureGazeData      = []     // full gaze object per frame { dir, h, v, yaw, pitch }
let _captureCount         = 0
let _captureCooldownUntil = 0
let _prevHandCount        = 0
let _captureAllBBoxes     = []
let _captureHandPosture   = 'unknown'  // hand posture at the moment trigger fired
let _capturePhoneSeen     = false      // real-time detector saw a phone during this capture window
let _capturePhoneFrames   = []         // per-frame on-device phone state — gives a per-frame table on EVERY incident
let _lastKnownIntruderBBoxes = []  // fallback when FaceDetector briefly misses a present intruder
let _lastKnownIntruderAt     = 0   // when it was last set — the fallback EXPIRES so a one-off (even false) intruder box can't blur a phone/notes out of every later capture for the whole session

// ── Layer 1: continuous lightweight signal buffer (the "memory buffer") ─────
// GOLDEN RULE: detect in layers, but WRITE EVERYTHING into the report. Every
// processed frame logs its light signals here at ~zero cost — face count, gaze,
// hand presence, motion, on-device phone hint — while heavy AI (Layer 3) runs
// only on triggered captures. At End Session this whole buffer is posted to the
// API and rendered as the report's SESSION OVERVIEW + FRAME TIMELINE, so nothing
// is discarded — only layered.
const MOTION_HIGH = 35   // motion score (0-100) above which a frame counts as "high motion"
const _sig = {
  t0: 0, tLast: 0, frames: 0,
  gaze: { AHEAD: 0, LEFT: 0, RIGHT: 0, UP: 0, DOWN: 0, NO_FACE: 0, UNKNOWN: 0 },
  face: { none: 0, one: 0, multi: 0 },
  handFrames: 0, phoneFrames: 0,
  qualityGood: 0, qualityPoor: 0,
  motionSum: 0, motionMax: 0, motionSamples: 0, motionHigh: 0,
  events: { phone_detected_realtime: 0, intruder_detected: 0,
            hand_visible_and_gaze_toward_wrist: 0, covert_device_suspected: 0,
            framing_torso_hidden: 0 },
  timeline: [],          // one finalized bucket per wall-clock second
  _curSec: -1, _bucket: null
}

// Motion: cheap luminance frame-diff on a 32×18 downscale (576 px — negligible
// cost). Returns 0-100. This is Layer-1 "motion" — pure arithmetic, no model.
const _motionCanvas = document.createElement('canvas')
_motionCanvas.width = 32; _motionCanvas.height = 18
const _motionCtx = _motionCanvas.getContext('2d', { willReadFrequently: true })
let _motionPrev = null
function computeMotion() {
  try {
    _motionCtx.drawImage(video, 0, 0, 32, 18)
    const cur = _motionCtx.getImageData(0, 0, 32, 18).data
    if (!_motionPrev) { _motionPrev = cur.slice(0); return 0 }
    let diff = 0
    for (let i = 0; i < cur.length; i += 4) {
      const l  = cur[i]        * 0.299 + cur[i + 1]     * 0.587 + cur[i + 2]     * 0.114
      const lp = _motionPrev[i] * 0.299 + _motionPrev[i + 1] * 0.587 + _motionPrev[i + 2] * 0.114
      diff += Math.abs(l - lp)
    }
    _motionPrev.set(cur)
    // 576 px × 255 max per-px diff; ×6 gain so ordinary movement reads on the 0-100 scale
    return Math.min(100, Math.round((diff / (576 * 255)) * 100 * 6))
  } catch (_) { return 0 }
}

function _finalizeBucket(b) {
  let g = 'AHEAD', gm = -1
  for (const k in b.gaze) if (b.gaze[k] > gm) { gm = b.gaze[k]; g = k }
  return {
    sec: b.sec, face: b.faceMax, gaze: g, hand: b.hand,
    motion: Math.round(b.motionSum / Math.max(1, b.n)), phone: b.phone
  }
}

// logSignalFrame records ONE frame's light signals into the buffer. Called every
// processed frame from the loop — must stay arithmetic-only (no DOM, no model).
function logSignalFrame(now, gs, motion) {
  if (_sig.t0 === 0) _sig.t0 = now
  _sig.tLast = now
  _sig.frames++

  const dir = (gs && gs.dir) || 'UNKNOWN'
  _sig.gaze[dir] = (_sig.gaze[dir] || 0) + 1

  const fc = _lastFaceCount
  if (fc <= 0) _sig.face.none++
  else if (fc === 1) _sig.face.one++
  else _sig.face.multi++

  const hand = !!(_lastHands && _lastHands.landmarks && _lastHands.landmarks.length > 0)
  if (hand) _sig.handFrames++
  if (_phoneVisible) _sig.phoneFrames++

  // Quality = a frame the detectors could actually trust (calibrated + a face present).
  const good = !!(gs && gs.calibrated && dir !== 'NO_FACE')
  if (good) _sig.qualityGood++; else _sig.qualityPoor++

  _sig.motionSum += motion; _sig.motionSamples++
  if (motion > _sig.motionMax) _sig.motionMax = motion
  if (motion > MOTION_HIGH) _sig.motionHigh++

  // 1 Hz timeline bucket — the full-truth per-second sample.
  const sec = Math.floor((now - _sig.t0) / 1000)
  if (sec !== _sig._curSec) {
    if (_sig._bucket) _sig.timeline.push(_finalizeBucket(_sig._bucket))
    _sig._curSec = sec
    _sig._bucket = { sec, n: 0, faceMax: 0, gaze: {}, hand: false, motionSum: 0, phone: false }
  }
  const bk = _sig._bucket
  bk.n++
  if (fc > bk.faceMax) bk.faceMax = fc
  bk.gaze[dir] = (bk.gaze[dir] || 0) + 1
  if (hand) bk.hand = true
  bk.motionSum += motion
  if (_phoneVisible) bk.phone = true
}

// _bucketRank scores how "interesting" a timeline sample is, so when the emitted
// timeline is downsampled for long sessions the anomaly (not a bland AHEAD row)
// survives — nothing suspicious is ever downsampled away.
function _bucketRank(s) {
  if (!s) return -1
  if (s.phone) return 4
  if (s.face >= 2) return 3
  if (s.gaze !== 'AHEAD' && s.gaze !== 'UNKNOWN' && s.gaze !== 'NO_FACE') return 2
  if (s.motion > MOTION_HIGH) return 1
  return 0
}

// getSessionSignals finalizes the buffer into the compact summary the report
// renders. The timeline is adaptively downsampled to ≤180 rows (keeping the most
// informative sample per window) so a long exam stays legible without dropping
// any anomaly.
function getSessionSignals() {
  if (_sig._bucket) { _sig.timeline.push(_finalizeBucket(_sig._bucket)); _sig._bucket = null; _sig._curSec = -1 }
  const durationSecs = _sig.t0 ? Math.max(0, Math.round((_sig.tLast - _sig.t0) / 1000)) : 0
  const avgFps = durationSecs > 0 ? +(_sig.frames / durationSecs).toFixed(1) : 0

  const full = _sig.timeline
  const step = Math.max(1, Math.ceil(full.length / 180))
  const tl = []
  for (let i = 0; i < full.length; i += step) {
    let pick = full[i]
    for (let j = i + 1; j < Math.min(i + step, full.length); j++) {
      if (_bucketRank(full[j]) > _bucketRank(pick)) pick = full[j]
    }
    if (pick) tl.push(pick)
  }

  let evTotal = 0
  for (const k in _sig.events) evTotal += _sig.events[k]

  return {
    captured_at: new Date().toISOString(),
    duration_secs: durationSecs,
    total_frames: _sig.frames,
    avg_fps: avgFps,
    sample_secs: step,
    gaze: _sig.gaze,
    face: _sig.face,
    hand_present_frames: _sig.handFrames,
    phone_hint_frames: _sig.phoneFrames,
    quality: { good: _sig.qualityGood, poor: _sig.qualityPoor },
    motion: {
      avg: _sig.motionSamples ? +(_sig.motionSum / _sig.motionSamples).toFixed(1) : 0,
      max: _sig.motionMax, high_frames: _sig.motionHigh
    },
    events: { ..._sig.events, total: evTotal },
    timeline: tl
  }
}
window.__getSessionSignals = getSessionSignals

// ── Keys ──────────────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  if (e.key === 'd' || e.key === 'D') _debug = !_debug
  if (e.key === 'c' || e.key === 'C') { resetCalibration(); console.log('[main] recalibrating') }
})

function setLoad(msg, sub = '') { loadMsg.textContent = msg; loadSub.textContent = sub }
function showError(msg) { errMsg.textContent = msg; errOverlay.classList.remove('hidden') }

// Change-gated DOM writes — only touch the DOM when the value actually changes.
// (Per-frame textContent/className writes were thrashing style recalc → FPS drop.)
function setText(el, txt) { if (el && el._t !== txt) { el.textContent = txt; el._t = txt } }
function setCls(el, cls)  { if (el && el._c !== cls) { el.className   = cls; el._c = cls } }

// ── Camera ────────────────────────────────────────────────────────────────
let _camStream   = null    // held so End Session can release the webcam
let _sessionEnded = false   // once true, the loop stops processing (session is over)

async function initCamera() {
  setLoad('Requesting camera…')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: 'user' }, audio: false
  })
  _camStream = stream
  video.srcObject = stream
  await new Promise(r => { video.onloadedmetadata = r })
  await video.play()
}

// endSessionCleanup releases the webcam and freezes the feed — called by the
// dashboard once the session has finalized. The camera light turns OFF and the
// split-screen shows a clear "SESSION ENDED" state so the candidate knows they
// are no longer being recorded.
function endSessionCleanup() {
  if (_sessionEnded) return
  _sessionEnded = true
  try {
    if (_camStream) { _camStream.getTracks().forEach(t => t.stop()); _camStream = null }
    if (video) { try { video.pause() } catch (_) {} video.srcObject = null }
  } catch (_) {}
  try {
    const canvas = document.getElementById('main-canvas')
    const ctx = canvas && canvas.getContext('2d')
    if (ctx) {
      ctx.fillStyle = '#0b0f17'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#e5e7eb'
      ctx.textAlign = 'center'
      ctx.font = '600 28px system-ui, sans-serif'
      ctx.fillText('SESSION ENDED', canvas.width / 2, canvas.height / 2 - 8)
      ctx.fillStyle = '#94a3b8'
      ctx.font = '15px system-ui, sans-serif'
      ctx.fillText('Camera stopped — recording finished.', canvas.width / 2, canvas.height / 2 + 24)
    }
  } catch (_) {}
  // Reflect it in the top-bar status chips.
  try {
    const rec = document.querySelector('.pulse-dot')?.parentElement
    if (rec) rec.style.opacity = '0.4'
    const st = document.getElementById('status-text')
    if (st) { st.textContent = 'ENDED'; st.className = 'stat-v stat-mono' }
  } catch (_) {}
}
window.__endSessionCleanup = endSessionCleanup

// ── MediaPipe init ────────────────────────────────────────────────────────
async function initModels() {
  setLoad('Loading MediaPipe…', 'Downloads models on first run')
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm')

  ;[faceLandmarker, faceDetector, handLandmarker] = await Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'CPU'
      },
      outputFaceBlendshapes: false,
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.3,
      minFacePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3
    }),
    FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.5
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      numHands: 2,
      // Defaults are 0.5 — weak enough that facial features occasionally read as
      // knuckles. Real hands score 0.8+, so 0.6 sheds phantoms, never real hands.
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6
    })
  ])

  // Phone detector loads LAST and independently. If its model download fails,
  // core proctoring (gaze / hand / intruder) still runs — phone detection just
  // no-ops. EfficientDet-Lite0 (~4MB, COCO) on the CPU delegate — no GPU.
  try {
    setLoad('Loading phone detector…', 'EfficientDet-Lite0 · CPU')
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      scoreThreshold: PHONE_MIN_SCORE,
      maxResults: 8
    })
    console.log('[boot] phone detector ready (EfficientDet-Lite0, CPU)')
  } catch (e) {
    console.warn('[boot] phone detector failed to load — continuing without it:', e?.message || e)
    objectDetector = null
  }
}

// ── Calibration overlay ───────────────────────────────────────────────────
function drawCalibrationOverlay(calPct) {
  const barW = PW * 0.6, barH = 14
  const barX = PW + (PW - barW) * 0.5
  const barY = PH * 0.5 + 30

  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.fillRect(PW, 0, PW, PH)

  ctx.save()
  ctx.font = 'bold 15px Segoe UI,monospace'
  ctx.fillStyle = '#f9a825'
  ctx.textAlign = 'center'
  ctx.fillText('STABILISING — look directly at the camera', PW + PW/2, PH*0.5 - 10)
  ctx.font = '11px Segoe UI,monospace'
  ctx.fillStyle = '#888'
  ctx.fillText('Press  C  to reset', PW + PW/2, PH*0.5 + 10)
  ctx.restore()

  ctx.fillStyle = '#222'; ctx.fillRect(barX, barY, barW, barH)
  ctx.fillStyle = '#43a047'; ctx.fillRect(barX, barY, barW * calPct, barH)
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.strokeRect(barX, barY, barW, barH)

  ctx.save()
  ctx.font = '11px Segoe UI,monospace'
  ctx.fillStyle = '#ccc'; ctx.textAlign = 'center'
  ctx.fillText(`${Math.round(calPct*100)}%`, PW + PW/2, barY + barH + 14)
  ctx.restore()
}

// ── Debug HUD ─────────────────────────────────────────────────────────────
function drawDebugHUD(gs) {
  const { yaw, pitch, h, v, sL, sR, sU, sD, score, mirrored, calibrated } = gs
  const f = x => (typeof x==='number' ? (x>=0?'+':'')+x.toFixed(2) : '---')
  const n = x => (typeof x==='number' ? x.toFixed(2) : '--')
  const line = calibrated
    ? `Yaw:${f(yaw)}  Pitch:${f(pitch)}  h:${f(h)}  v:${f(v)}  ` +
      `L:${n(sL)} R:${n(sR)} U:${n(sU)} D:${n(sD)}  score:${n(score)}  ${mirrored?'MIR':'RAW'}`
    : `settling… yaw:${f(yaw)} pitch:${f(pitch)} h:${f(h)} v:${f(v)}`

  const x = 6, y = PH - 6
  ctx.save()
  ctx.font = '10px Consolas,monospace'
  ctx.fillStyle = 'rgba(0,0,0,0.7)'
  const tw = ctx.measureText(line).width
  ctx.fillRect(x-2, y-12, tw+6, 16)
  ctx.fillStyle = '#aaa'
  ctx.fillText(line, x, y)
  ctx.restore()
}

// ── Draw hand bounding box + wrist dot ───────────────────────────────────
function drawHand(lm, handedness) {
  let mnX=1, mnY=1, mxX=0, mxY=0
  for (const p of lm) {
    if (p.x < mnX) mnX=p.x; if (p.x > mxX) mxX=p.x
    if (p.y < mnY) mnY=p.y; if (p.y > mxY) mxY=p.y
  }
  const bx = PW + mnX*PW - 6, by = mnY*PH - 6
  const bw = (mxX-mnX)*PW + 12, bh = (mxY-mnY)*PH + 12

  ctx.strokeStyle = '#f9a825'; ctx.lineWidth = 2
  ctx.strokeRect(bx, by, bw, bh)

  const wx = PW + lm[0].x*PW, wy = lm[0].y*PH
  ctx.beginPath(); ctx.arc(wx, wy, 5, 0, Math.PI*2)
  ctx.fillStyle = '#f9a825'; ctx.fill()

  ctx.font = 'bold 12px Segoe UI,monospace'
  ctx.fillStyle = '#f9a825'
  ctx.fillText(`HAND (${handedness})`, bx+4, Math.max(by-5, 12))
}

// ── Hand false-positive filter ────────────────────────────────────────────
// BlazeFace sometimes detects fists/hands as faces.
// If any hand landmark falls inside a detection box → it's a hand, not an intruder.
function isHandBox(bb, handLandmarks, vW, vH) {
  const bL = bb.originX / vW, bR = (bb.originX + bb.width)  / vW
  const bT = bb.originY / vH, bB = (bb.originY + bb.height) / vH
  for (const lm of handLandmarks) {
    let inside = 0
    for (const pt of lm) {
      if (pt.x >= bL && pt.x <= bR && pt.y >= bT && pt.y <= bB) inside++
    }
    if (inside >= 5) return true  // fist-as-face has 15+ pts inside; nearby hand has 2-3
  }
  return false
}

// ── Phantom-hand filter ─────────────────────────────────────────────────────
// HandLandmarker occasionally hallucinates a "hand" ON the candidate's face
// (nose/mouth reads as knuckles). A real hand touching the face always extends
// beyond it — palm and wrist reach below the chin — so a detection whose 21
// landmarks sit almost entirely inside the face's own landmark bbox is a false
// positive. Dropped at the source, so the HAND banner, the stat card, the
// wrist-look trigger and the hand_posture hint all stay phantom-free.
function filterPhantomHands(h, faceResult) {
  const hands = { landmarks: h.landmarks || [], handedness: h.handedness || [] }
  const lm = faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks[0]
  if (!lm || hands.landmarks.length === 0) return hands
  let fL = 1, fR = 0, fT = 1, fB = 0
  for (const p of lm) {
    if (p.x < fL) fL = p.x
    if (p.x > fR) fR = p.x
    if (p.y < fT) fT = p.y
    if (p.y > fB) fB = p.y
  }
  const keepL = [], keepH = []
  for (let i = 0; i < hands.landmarks.length; i++) {
    let inside = 0
    for (const pt of hands.landmarks[i]) {
      if (pt.x >= fL && pt.x <= fR && pt.y >= fT && pt.y <= fB) inside++
    }
    // A hand-on-chin puts ~8-14 pts in the face box; a phantom puts 20-21.
    if (inside < 18) { keepL.push(hands.landmarks[i]); keepH.push(hands.handedness[i]) }
  }
  return { landmarks: keepL, handedness: keepH }
}

// ── Real-face validator ─────────────────────────────────────────────────────
// BlazeFace also fires on phones (the dark camera module looks like a face),
// which caused false intruders AND blurred the phone out of the LLM collage.
// Reject a detection unless it's confident AND its keypoints form a real face
// (two eyes sitting above the mouth, sanely separated).
const FACE_MIN_SCORE = 0.55
function looksLikeFace(det, vW) {
  const score = det?.categories?.[0]?.score
  if (typeof score === 'number' && score < FACE_MIN_SCORE) return false
  const kp = det?.keypoints
  if (!kp || kp.length < 4) return true   // no keypoints to check → trust the score gate
  // BlazeFace order: 0=right eye, 1=left eye, 2=nose, 3=mouth, 4/5=ear tragions
  const rEye = kp[0], lEye = kp[1], mouth = kp[3]
  if (!rEye || !lEye || !mouth) return true
  if (!(rEye.y < mouth.y && lEye.y < mouth.y)) return false   // eyes must be above the mouth
  const eyeDx = Math.abs(rEye.x - lEye.x)
  const boxW  = det.boundingBox.width / (vW || 1280)
  if (eyeDx < boxW * 0.12 || eyeDx > boxW * 1.0) return false // eyes implausibly close/far
  return true
}

// A detection is a true intruder face only if it isn't a hand AND looks like a real face.
function isIntruderFace(det, handLandmarks, vW, vH) {
  if (isHandBox(det.boundingBox, handLandmarks, vW, vH)) return false
  return looksLikeFace(det, vW)
}

// ── Capture: grab current video frame, pixelate ALL faces ────────────────
const PIXEL_BLOCK = 8    // mosaic block size
const FACE_PAD    = 0.06 // small padding — hug the face so the blur never spills
                         // onto held notes/phones or the surroundings (was 0.22,
                         // then 0.10). Combined with the ELLIPSE mask in
                         // pixelateFace, ONLY the face oval is blurred — not the
                         // rectangle of neck/hair/background around it.

// Convert normalized bbox → padded normalized bbox
function padBBox(minX, minY, maxX, maxY) {
  const padX = (maxX - minX) * FACE_PAD
  const padY = (maxY - minY) * FACE_PAD
  return {
    minX: Math.max(0, minX - padX),
    maxX: Math.min(1, maxX + padX),
    minY: Math.max(0, minY - padY),
    maxY: Math.min(1, maxY + padY)
  }
}

// Snapshot ALL face bboxes at capture start — candidate (from landmarks) + intruders (from detector)
// Must copy coordinates NOW — MediaPipe buffers get recycled on next detectForVideo call
function snapshotAllFaceBBoxes() {
  const bboxes = []
  const vW = video.videoWidth || 1280, vH = video.videoHeight || 720

  // Candidate — FaceLandmarker landmarks (accurate, updates every frame)
  try {
    const faces = _lastGs && _lastGs.faces
    if (faces && faces.length > 0) {
      const lm = faces[0]
      if (lm && lm.length >= 455) {
        const xs = [lm[10].x, lm[152].x, lm[234].x, lm[454].x]
        const ys = [lm[10].y, lm[152].y, lm[234].y, lm[454].y]
        bboxes.push(padBBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)))
      }
    }
  } catch (_) {}

  // Intruders — use current FaceDetector results, fall back to last known if none found.
  // Only blur boxes that are REAL faces (not phones/hands) — otherwise we'd pixelate
  // the phone out of the collage and the LLM could never identify it.
  const currentIntruders = []
  try {
    for (let i = 0; i < _lastDetections.length; i++) {
      if (i === _primaryDetIdx) continue
      if (!isIntruderFace(_lastDetections[i], _lastHands.landmarks, vW, vH)) continue
      const bb = _lastDetections[i].boundingBox
      currentIntruders.push(padBBox(
        bb.originX / vW,
        bb.originY / vH,
        (bb.originX + bb.width)  / vW,
        (bb.originY + bb.height) / vH
      ))
    }
  } catch (_) {}

  // If FaceDetector missed them (extreme angle, head down), fall back to the last
  // known position — but ONLY while it is fresh. A stale box (a transient/false
  // second face that already left) must NOT keep pixelating a region every capture,
  // or it could blur a real phone/notes out of the frames the LLM sees.
  const _fallbackFresh = (performance.now() - _lastKnownIntruderAt) < 1500
  bboxes.push(...(currentIntruders.length > 0 ? currentIntruders : (_fallbackFresh ? _lastKnownIntruderBBoxes : [])))

  return bboxes
}

// Pixelate a bbox region on canvas oc from video source
function pixelateFace(oc, bbox) {
  const { minX, minY, maxX, maxY } = bbox
  const bx = minX * PW, by = minY * PH
  const bw = (maxX - minX) * PW, bh = (maxY - minY) * PH
  if (bw < 1 || bh < 1) return

  const vW = video.videoWidth || 1280, vH = video.videoHeight || 720
  const tW = Math.max(1, Math.round(bw / PIXEL_BLOCK))
  const tH = Math.max(1, Math.round(bh / PIXEL_BLOCK))
  const tmp = document.createElement('canvas')
  tmp.width = tW; tmp.height = tH
  const tc = tmp.getContext('2d')
  tc.drawImage(video, minX * vW, minY * vH, (maxX - minX) * vW, (maxY - minY) * vH, 0, 0, tW, tH)

  // Clip the mosaic to an ELLIPSE inscribed in the box so we blur ONLY the face
  // oval — the rectangular corners (background, neck, hair edges) stay sharp.
  oc.save()
  oc.beginPath()
  oc.ellipse(bx + bw / 2, by + bh / 2, bw / 2, bh / 2, 0, 0, Math.PI * 2)
  oc.clip()
  oc.imageSmoothingEnabled = false
  oc.drawImage(tmp, 0, 0, tW, tH, bx, by, bw, bh)
  oc.imageSmoothingEnabled = true
  oc.restore()
}

// Returns offscreen canvas with pixelated faces — caller encodes to JPEG
function captureCanvas() {
  const off = document.createElement('canvas')
  off.width = PW; off.height = PH
  const oc = off.getContext('2d')
  oc.drawImage(video, 0, 0, PW, PH)
  for (const bbox of _captureAllBBoxes) {
    try { pixelateFace(oc, bbox) } catch (_) {}
  }
  return off
}

// ── Collage: 3×2 grid of frames with gaze labels ─────────────────────────
const CW = 400, CH = 225   // each frame cell size (scaled from 640×360)
const LH = 55              // label bar height — two lines: gaze direction + iris h/v
const GP = 10              // gap between cells

function createCollage(canvases, gazeLabels, gazeData = []) {
  const COLS = 3, ROWS = 2
  const totalW = COLS * CW + (COLS + 1) * GP
  const totalH = ROWS * (CH + LH) + (ROWS + 1) * GP + 50  // +50 for header

  const coll = document.createElement('canvas')
  coll.width = totalW; coll.height = totalH
  const cc = coll.getContext('2d')

  // background
  cc.fillStyle = '#0d0d0d'
  cc.fillRect(0, 0, totalW, totalH)

  // header
  cc.fillStyle = '#1a1a1a'
  cc.fillRect(0, 0, totalW, 44)
  cc.fillStyle = '#f9a825'
  cc.font = 'bold 16px Segoe UI,monospace'
  cc.textAlign = 'left'
  cc.fillText('PROCTOR INCIDENT REPORT', GP, 28)
  cc.fillStyle = '#555'
  cc.font = '12px Consolas,monospace'
  cc.textAlign = 'right'
  cc.fillText(new Date().toLocaleString(), totalW - GP, 28)

  for (let i = 0; i < canvases.length; i++) {
    const col = i % COLS, row = Math.floor(i / COLS)
    const x = GP + col * (CW + GP)
    const y = 50 + GP + row * (CH + LH + GP)

    // frame image
    cc.drawImage(canvases[i], 0, 0, PW, PH, x, y, CW, CH)

    // frame border
    cc.strokeStyle = '#333'; cc.lineWidth = 1
    cc.strokeRect(x, y, CW, CH)

    // frame number badge (top-left)
    cc.fillStyle = 'rgba(0,0,0,0.75)'
    cc.fillRect(x + 4, y + 4, 30, 19)
    cc.fillStyle = '#aaa'
    cc.font = 'bold 11px Consolas,monospace'
    cc.textAlign = 'left'
    cc.fillText(`#${i + 1}`, x + 9, y + 16)

    // gaze label bar
    const dir = gazeLabels[i] || 'UNKNOWN'
    const gd  = gazeData && gazeData[i] ? gazeData[i] : null
    const labelCol = dir === 'AHEAD' ? '#43a047'
                   : dir === 'NO_FACE' ? '#f9a825'
                   : '#e53935'
    cc.fillStyle = '#161616'
    cc.fillRect(x, y + CH, CW, LH)
    cc.strokeStyle = labelCol; cc.lineWidth = 2
    cc.beginPath(); cc.moveTo(x, y + CH); cc.lineTo(x + CW, y + CH); cc.stroke()
    cc.fillStyle = labelCol
    cc.font = 'bold 14px Segoe UI,monospace'
    cc.textAlign = 'center'
    const tStr = gd && gd.t
      ? new Date(gd.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : ''
    cc.fillText(`FRAME ${i + 1}${tStr ? '  ·  ' + tStr : ''}  —  ${dir === 'NO_FACE' ? 'NO FACE' : `LOOKING ${dir}`}`, x + CW / 2, y + CH + 20)
    // second line: iris h/v + head yaw/pitch
    if (gd) {
      const f = v => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(2) : '--')
      cc.fillStyle = '#888'
      cc.font = '11px Consolas,monospace'
      cc.fillText(`iris h:${f(gd.h)}  v:${f(gd.v)}  yaw:${f(gd.yaw)}  pitch:${f(gd.pitch)}`, x + CW / 2, y + CH + 38)
    }
  }

  return coll.toDataURL('image/jpeg', 0.92)
}

// ── Hand posture from 21 landmarks (zero-cost, no extra model) ────────────
// Returns "holding_object" if fingers are curled (gripping), else "open_hand".
function computeHandPosture(lm) {
  try {
    const wrist = lm[0]
    const tips  = [8, 12, 16, 20]   // index, middle, ring, pinky tips
    const mcps  = [5,  9, 13, 17]   // corresponding MCP knuckles
    let curled  = 0
    for (let i = 0; i < tips.length; i++) {
      const tip = lm[tips[i]], mcp = lm[mcps[i]]
      const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y)
      const dMcp = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
      if (dTip < dMcp * 1.5) curled++  // tip closer to wrist than 1.5× MCP = curled
    }
    return curled >= 3 ? 'holding_object' : 'open_hand'
  } catch (_) { return 'unknown' }
}

function startCapture() {
  _captureActive     = true
  window.__captureBusy = true   // signals dashboard's End Session to wait for this upload
  _captureCanvases   = []
  _captureFrames     = []
  _captureGazeLabels = []
  _captureGazeData   = []
  _captureCount      = 0
  _captureAllBBoxes  = []
  _captureMaxFaces   = _lastFaceCount   // seed with the count at trigger moment
  _capturePhoneSeen  = _phoneVisible    // seed with the phone flag at trigger moment
  _capturePhoneFrames = []

  // Snapshot hand posture at trigger moment
  _captureHandPosture = 'unknown'
  try {
    if (_lastHands.landmarks && _lastHands.landmarks.length > 0) {
      _captureHandPosture = computeHandPosture(_lastHands.landmarks[0])
    }
  } catch (_) {}

  console.log(`[capture] started (${_captureReason}) — ${CAPTURE_TOTAL} frames × ${CAPTURE_GAP_MS}ms  posture:${_captureHandPosture}  faces:${_captureMaxFaces}`)
  setTimeout(captureNextFrame, 0)
}

function captureNextFrame() {
  _captureAllBBoxes = snapshotAllFaceBBoxes()   // refresh every frame — intruder may have entered
  if (_lastFaceCount > _captureMaxFaces) _captureMaxFaces = _lastFaceCount   // track peak people in frame
  if (_phoneVisible) _capturePhoneSeen = true   // phone seen at any point during the capture
  const canvas = captureCanvas()
  const gs  = _lastGs || {}
  const dir = gs.dir || 'UNKNOWN'
  const gd  = {
    dir:   dir,
    t:     new Date().toISOString(),
    h:     typeof gs.h     === 'number' ? +gs.h.toFixed(4)     : null,
    v:     typeof gs.v     === 'number' ? +gs.v.toFixed(4)     : null,
    yaw:   typeof gs.yaw   === 'number' ? +gs.yaw.toFixed(2)   : null,
    pitch: typeof gs.pitch === 'number' ? +gs.pitch.toFixed(2) : null
  }

  _captureCanvases.push(canvas)
  _captureGazeLabels.push(dir)
  _captureGazeData.push(gd)
  _captureFrames.push(canvas.toDataURL('image/jpeg', 0.92)) // high quality — sent to LLM as separate frames; small-object legibility matters
  // Per-frame on-device phone state from the FRESH detector recency, NOT the 1.5s
  // banner debounce — otherwise a phone seen only in the previous frame would bleed
  // its flag onto this (phone-free) frame and mis-stamp that row "phone".
  _capturePhoneFrames.push(_phoneLastSeen > 0 && (performance.now() - _phoneLastSeen) < CAPTURE_GAP_MS)
  _captureCount++
  console.log(`[capture] frame ${_captureCount}/${CAPTURE_TOTAL} — ${dir}  h:${gd.h}  v:${gd.v}  yaw:${gd.yaw}  pitch:${gd.pitch}`)

  if (_captureCount < CAPTURE_TOTAL) {
    setTimeout(captureNextFrame, CAPTURE_GAP_MS)
  } else {
    sendToAPI()
  }
}

async function sendToAPI() {
  console.log('[capture] building collage…')
  let collage = ''
  try { collage = createCollage(_captureCanvases, _captureGazeLabels, _captureGazeData) } catch (_) {}

  console.log('[capture] sending to API…')
  try {
    const res = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frames:         _captureFrames,
        collage:        collage,
        gazeLabels:     _captureGazeLabels,
        gazeData:       _captureGazeData,
        timestamp:         new Date().toISOString(),
        phone_detected:    _capturePhoneSeen,
        frame_phones:      _capturePhoneFrames,
        hand_posture:      _captureHandPosture,
        face_count:        _captureMaxFaces,
        intruder_detected: _captureMaxFaces >= 2,
        trigger_reason:    _captureReason
      })
    })
    const data = await res.json()
    console.log('[capture] saved:', data)
  } catch (e) {
    console.error('[capture] API error:', e.message)
  } finally {
    _captureActive        = false
    _captureCanvases      = []
    _captureFrames        = []
    _captureGazeLabels    = []
    _captureGazeData      = []
    _captureCooldownUntil = performance.now() + CAPTURE_COOLDOWN_MS
    window.__captureBusy  = false   // upload finished — End Session may proceed
  }
}

// computePrimaryDetIdx returns the index of the detection nearest the candidate's
// own face (so it is excluded from the intruder logic). It MUST be recomputed
// against the CURRENT detections each time — BlazeFace ordering is not stable, so
// a cached index from a previous frame can point at the wrong face (skipping a
// real intruder or pixelating the candidate).
function computePrimaryDetIdx(detections, faces, vW, vH) {
  if (!detections || detections.length === 0 || !faces || faces.length === 0) return 0
  const fx = faces[0][1].x, fy = faces[0][1].y
  let minDist = Infinity, idx = 0
  for (let i = 0; i < detections.length; i++) {
    const bb = detections[i].boundingBox
    const cx = (bb.originX + bb.width * 0.5) / vW
    const cy = (bb.originY + bb.height * 0.5) / vH
    const d = (cx - fx) * (cx - fx) + (cy - fy) * (cy - fy)
    if (d < minDist) { minDist = d; idx = i }
  }
  return idx
}

// ── Render ────────────────────────────────────────────────────────────────
function render(gs, detections, hands, now) {
  const { dir, faces, calPct, calibrated } = gs
  const vW = video.videoWidth || 1280, vH = video.videoHeight || 720
  const realFaces = detections
    ? detections.filter(d => isIntruderFace(d, hands?.landmarks || [], vW, vH)).length
    : 0
  const numFaces = realFaces
  const numHands = hands && hands.landmarks ? hands.landmarks.length : 0
  _lastFaceCount = numFaces   // expose to the loop's intruder trigger + capture snapshot

  ctx.drawImage(video, 0,  0, PW, PH)
  ctx.drawImage(video, PW, 0, PW, PH)

  // ── Candidate iris ───────────────────────────────────────────────────
  if (faces && faces.length > 0) {
    const col = (dir !== 'AHEAD' && dir !== 'NO_FACE') ? '#e53935' : '#43a047'
    drawIris(ctx, faces[0], LI_C,LI_T,LI_R,LI_B,LI_L, col, PW, 0, PW, PH)
    drawIris(ctx, faces[0], RI_C,RI_T,RI_R,RI_B,RI_L, col, PW, 0, PW, PH)
  }

  // ── Intruder boxes — compute + store primary index ───────────────────
  if (detections && detections.length > 0) {
    const primaryIdx = computePrimaryDetIdx(detections, faces, vW, vH)
    _primaryDetIdx = primaryIdx   // store for blurredFrame()

    ctx.font = 'bold 12px Segoe UI,monospace'
    let intruderId = 1
    for (let i = 0; i < detections.length; i++) {
      if (i === primaryIdx) continue
      if (!isIntruderFace(detections[i], _lastHands.landmarks, vW, vH)) continue  // skip hands & phones
      const bb = detections[i].boundingBox
      const bx = PW + (bb.originX/vW)*PW, by = (bb.originY/vH)*PH
      const bw = (bb.width/vW)*PW,        bh = (bb.height/vH)*PH
      ctx.strokeStyle = '#e53935'; ctx.lineWidth = 2
      ctx.strokeRect(bx, by, bw, bh)
      ctx.fillStyle = '#e53935'
      ctx.fillText(`! INTRUDER ${intruderId++}`, bx+4, Math.max(by-5, 12))
    }
  }

  // ── Hands ────────────────────────────────────────────────────────────
  if (hands && hands.landmarks) {
    for (let hi = 0; hi < hands.landmarks.length; hi++) {
      drawHand(hands.landmarks[hi], hands.handedness[hi]?.[0]?.displayName || '?')
    }
  }

  // ── Phone box (real-time detector) ────────────────────────────────────
  // Draw ONLY on frames the detector actually saw a phone (_phoneBoxFresh). The
  // banner may hold longer to debounce the text, but the box must never freeze
  // on a stale position — that "ghost box" trailing reality reads as latency.
  // FPS-independent: the box clears at the very next detector run.
  if (_phoneBox && _phoneBoxFresh) {
    const bx = PW + (_phoneBox.originX / vW) * PW, by = (_phoneBox.originY / vH) * PH
    const bw = (_phoneBox.width / vW) * PW,        bh = (_phoneBox.height / vH) * PH
    ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 3
    ctx.strokeRect(bx, by, bw, bh)
    ctx.font = 'bold 12px Segoe UI,monospace'
    ctx.fillStyle = '#dc2626'
    ctx.fillText(`PHONE ${Math.round(_phoneScore * 100)}%`, bx + 4, Math.max(by - 5, 12))
  }

  // ── Capture progress indicator ───────────────────────────────────────
  if (_captureActive) {
    const label = `● CAPTURING  ${_captureCount}/${CAPTURE_TOTAL}`
    ctx.save()
    ctx.font = 'bold 13px Segoe UI,monospace'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = 'rgba(0,0,0,0.75)'
    ctx.fillRect(PW + PW/2 - tw/2 - 10, PH - 38, tw + 20, 26)
    ctx.fillStyle = '#f9a825'
    ctx.textAlign = 'center'
    ctx.fillText(label, PW + PW/2, PH - 20)
    ctx.restore()
  }

  if (!calibrated) drawCalibrationOverlay(calPct)

  if (calibrated && dir !== 'AHEAD') {
    ctx.save()
    ctx.font = 'bold 24px Segoe UI,monospace'
    ctx.fillStyle = dir === 'NO_FACE' ? '#f9a825' : '#e53935'
    ctx.textAlign = 'center'
    ctx.fillText(dir === 'NO_FACE' ? 'NO FACE' : `LOOKING ${dir}`, PW + PW/2, 34)
    ctx.restore()
  }

  if (_debug) drawDebugHUD(gs)

  ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(PW, 0); ctx.lineTo(PW, PH); ctx.stroke()

  // ── Stat cards (change-gated) ─────────────────────────────────────────
  let gTxt, gCls
  if (!calibrated)            { gTxt = `Stabilising ${Math.round(calPct*100)}%`; gCls = 'status-nof' }
  else if (dir === 'NO_FACE') { gTxt = 'NO FACE';       gCls = 'status-nof' }
  else if (dir === 'AHEAD')   { gTxt = 'LOOKING AHEAD'; gCls = 'status-ok'  }
  else                        { gTxt = `LOOKING ${dir}`; gCls = 'status-warn' }
  setText(gazeEl, gTxt); setCls(gazeEl, `stat-v ${gCls}`)

  const intruderCount = Math.max(0, numFaces - 1)
  const fTxt = numFaces > 1
    ? `${numFaces} FACES — ${intruderCount} INTRUDER${intruderCount>1?'S':''}`
    : numFaces === 1 ? '1 FACE' : 'NO FACE'
  setText(faceEl, fTxt); setCls(faceEl, `stat-v ${numFaces > 1 ? 'status-warn' : ''}`.trim())

  const hTxt = numHands > 0 ? `${numHands} HAND${numHands>1?'S':''} DETECTED` : 'NO HAND'
  setText(handEl, hTxt); setCls(handEl, `stat-v ${numHands > 0 ? 'status-warn' : 'status-ok'}`)

  handBanner.classList.toggle('hidden', numHands === 0)
  faceBanner.classList.toggle('hidden', numFaces <= 1)
  phoneBanner?.classList.toggle('hidden', !_phoneVisible)
  covertBanner?.classList.toggle('hidden', _covertGlances.length < 3)
  framingBanner?.classList.toggle('hidden', !_framingHidden)

  const sTxt = _captureActive
    ? `CAPTURING ${_captureCount}/${CAPTURE_TOTAL}`
    : (!calibrated ? 'STABILISING' : 'MONITORING')
  setText(statusEl, sTxt); setCls(statusEl, 'stat-v stat-mono')

  fpsAcc++
  const elapsed = now - fpsT0
  if (elapsed >= 1000) {
    const fps = fpsAcc * 1000 / elapsed
    _emaFps = _emaFps * 0.6 + fps * 0.4   // smoothed → drives the adaptive detector cadence
    fpsEl.textContent = `${Math.round(fps)} FPS`
    fpsAcc = 0; fpsT0 = now
  }
}

// ── Is candidate looking toward the detected wrist? ──────────────────────
// Three signals, each requiring TWO confirmations — never fires on one sensor alone.
//
// Raw frame coords: large x = anatomical LEFT, small x = anatomical RIGHT.
// Iris signs (gaze.js, non-mirrored): h<0=LEFT h>0=RIGHT v<0=DOWN.
// Head: yaw>0=turned-LEFT, pitch>0=pitched-DOWN.
//
// T_SP=0.10: wrist must be clearly spatially away from nose before any check.
function isLookingAtHand() {
  if (!_lastGs || !_lastGs.faces || _lastGs.faces.length === 0) return false
  if (!_lastHands || !_lastHands.landmarks || _lastHands.landmarks.length === 0) return false
  if (!_lastGs.calibrated) return false
  if (_lastGs.dir === 'NO_FACE') return false

  const dir   = _lastGs.dir
  const h     = typeof _lastGs.h     === 'number' ? _lastGs.h     : 0
  const v     = typeof _lastGs.v     === 'number' ? _lastGs.v     : 0
  const yaw   = typeof _lastGs.yaw   === 'number' ? _lastGs.yaw   : 0
  const pitch = typeof _lastGs.pitch === 'number' ? _lastGs.pitch : 0
  const nose  = _lastGs.faces[0][1]

  const T_SP    = 0.10   // wrist must be clearly off-face before any gaze check
  const T_IRIS  = 0.09   // iris deviation (strong — avoids casual glances)
  const T_YAW   = 18     // degrees head yaw toward wrist
  const T_PITCH = 13     // degrees head pitch down toward wrist

  for (const hand of _lastHands.landmarks) {
    try {
      const wrist = hand[0]
      const dx = wrist.x - nose.x   // > 0 = wrist to anatomical LEFT
      const dy = wrist.y - nose.y   // > 0 = wrist below face

      // Gate: wrist must be clearly away from face in the relevant direction
      const wristLeft  = dx >  T_SP
      const wristRight = dx < -T_SP
      const wristBelow = dy >  T_SP

      if (!wristLeft && !wristRight && !wristBelow) continue

      // SIGNAL A — combined gaze dir (head + iris fused by gaze.js) → most reliable
      if (wristLeft  && dir === 'LEFT')  return true
      if (wristRight && dir === 'RIGHT') return true
      if (wristBelow && dir === 'DOWN')  return true

      // SIGNAL B — iris clearly pointing at wrist AND head at least slightly following
      if (wristLeft  && h < -T_IRIS && yaw   >  5) return true
      if (wristRight && h >  T_IRIS && yaw   < -5) return true
      if (wristBelow && v < -T_IRIS && pitch >  5) return true

      // SIGNAL C — head clearly turned at wrist AND iris at least slightly following
      if (wristLeft  && yaw   >  T_YAW   && h < -0.04) return true
      if (wristRight && yaw   < -T_YAW   && h >  0.04) return true
      if (wristBelow && pitch >  T_PITCH  && v < -0.04) return true
    } catch (_) {}
  }
  return false
}

// ── Layer 1: run the phone detector (called on a staggered frame) ──────────
// Picks the highest-confidence "cell phone" detection, drives the banner/box,
// and tracks a streak so a single fluke frame doesn't fire a capture. The LLM
// still judges the captured frames — this is an INSTANT corroborating tripwire,
// not the verdict.
function runPhoneDetector(now) {
  if (!objectDetector) return
  let best = null
  try {
    const res = objectDetector.detectForVideo(video, now)
    for (const d of (res?.detections || [])) {
      const c = d.categories && d.categories[0]
      if (!c) continue
      const name = (c.categoryName || c.displayName || '').toLowerCase()
      if (!name.includes('phone')) continue          // COCO "cell phone"
      if (c.score < PHONE_MIN_SCORE) continue
      if (!best || c.score > best.score) best = { score: c.score, box: d.boundingBox }
    }
  } catch (_) { /* detector hiccup — ignore this frame */ }

  if (best) {
    _phoneVisible  = true
    _phoneScore    = best.score
    _phoneBox      = best.box
    _phoneLastSeen = now
    _phoneStreak++
    _phoneBoxFresh = true
    if (_captureActive) _capturePhoneSeen = true
  } else {
    _phoneStreak   = 0
    _phoneBoxFresh = false   // box clears at the next detector run — never a stale ghost
    if (now - _phoneLastSeen > PHONE_BANNER_HOLD) { _phoneVisible = false; _phoneBox = null }
  }
}

// Normalised centre of the current phone box (for the same-spot debounce).
function phoneCenterNow() {
  if (!_phoneBox) return null
  const vW = video.videoWidth || 1280, vH = video.videoHeight || 720
  return {
    x: (_phoneBox.originX + _phoneBox.width  * 0.5) / vW,
    y: (_phoneBox.originY + _phoneBox.height * 0.5) / vH
  }
}

// ── Layer 2: accumulate covert "looking-down-at-a-hidden-device" behaviour ──
// Keys on HEAD pitch, not eye gaze: glancing at a lower screen is eyes-only with
// the head upright, whereas reading a lap/desk phone tilts the head clearly down.
// Counts discrete down-episodes in a rolling window; the loop fires NEEDS-REVIEW
// when they repeat or one stare runs abnormally long.
function updateCovert(now) {
  const gs = _lastGs
  if (!gs || !gs.calibrated || gs.dir === 'NO_FACE') { _downStart = 0; return }
  const pitch = typeof gs.pitch === 'number' ? gs.pitch : null
  const headDown = pitch !== null ? pitch > COVERT_PITCH_DEG : gs.dir === 'DOWN'

  if (headDown) {
    if (_downStart === 0) _downStart = now
  } else if (_downStart !== 0) {
    if (now - _downStart >= COVERT_MIN_EPISODE) _covertGlances.push(now)
    _downStart = 0
  }
  const cutoff = now - COVERT_WINDOW_MS
  while (_covertGlances.length && _covertGlances[0] < cutoff) _covertGlances.shift()
}

function covertTriggered(now) {
  const longStare = _downStart !== 0 && (now - _downStart) >= COVERT_LONG_STARE
  return _covertGlances.length >= COVERT_MIN_GLANCES || longStare
}

// ── Layer 2b: framing manipulation — is the torso/hands tilted out of frame? ──
// Uses the chin landmark (152) FaceLandmarker already produces every frame: if
// the chin sits in the bottom band there is physically no room below it for the
// shoulders/hands, i.e. the camera (or laptop lid) was tilted up to hide what's
// on the lap/desk. Pure arithmetic — zero added model cost.
function updateFraming(now, faceResult) {
  const lm   = faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks[0]
  const chin = lm && lm[152]
  const torsoHidden = !!chin && chin.y > FRAMING_CHIN_Y
  if (torsoHidden) {
    if (_framingSince === 0) _framingSince = now
    if (now - _framingSince >= FRAMING_SUSTAIN_MS) _framingHidden = true
  } else {
    _framingSince  = 0
    _framingHidden = false
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────
function loop(now) {
  if (_sessionEnded) return        // session finalized — camera released, stop processing
  requestAnimationFrame(loop)   // re-queue FIRST — loop never dies even if below throws
  const _dt = now - _lastProcessT
  if (_dt < MIN_FRAME_MS) return   // FPS cap → steady, no thermal throttle
  // Carry the timing remainder. A plain `_lastProcessT = now` reset quantizes
  // the loop to the display's vsync grid — on a 60Hz screen (16.7ms ticks) a
  // ~21ms budget then skips EVERY OTHER tick, halving the loop to 30 FPS. The
  // carry lets short and long gaps alternate so the true average IS the cap,
  // on any refresh rate.
  _lastProcessT = now - (_dt % MIN_FRAME_MS)
  try {
    _frameN++

    const faceResult = faceLandmarker.detectForVideo(video, now)
    const gs = processGaze(faceResult)
    _lastGs = gs   // store for snapshotFaceBBox()

    if (_frameN % 4 === 1) {
      const h = handLandmarker.detectForVideo(video, now)
      _lastHands = filterPhantomHands(h, faceResult)   // drop "hands" hallucinated on the face
    }

    if (_frameN % 4 === 3) {
      const d = faceDetector.detectForVideo(video, now)
      _lastDetections = d.detections || []
      // Cache intruder bboxes — updated every 4th frame so capture frame 1 has fresh data.
      // Recompute the primary (own-face) index against the FRESH detections — the
      // cached _primaryDetIdx is from the previous render and BlazeFace ordering is
      // not stable, so reusing it could skip a real intruder or cache the own face.
      const _vW = video.videoWidth || 1280, _vH = video.videoHeight || 720
      const _primIdx = computePrimaryDetIdx(_lastDetections, gs.faces, _vW, _vH)
      const _intruders = []
      for (let _i = 0; _i < _lastDetections.length; _i++) {
        if (_i === _primIdx) continue
        try {
          if (isIntruderFace(_lastDetections[_i], _lastHands.landmarks, _vW, _vH)) {
            const _bb = _lastDetections[_i].boundingBox
            _intruders.push(padBBox(_bb.originX/_vW, _bb.originY/_vH,
              (_bb.originX+_bb.width)/_vW, (_bb.originY+_bb.height)/_vH))
          }
        } catch (_) {}
      }
      if (_intruders.length > 0) { _lastKnownIntruderBBoxes = _intruders; _lastKnownIntruderAt = now }
    }

    // Phone detector — staggered to its own frame so no single frame runs two
    // heavy models. CRITICAL: the stride MUST be a MULTIPLE OF 4 so the offset-2
    // slot never drifts onto the hand frame (%4===1) or the face-detector frame
    // (%4===3) — a non-multiple like 45 (45%4===1) would slide across both and
    // run 3 CPU models on the same frame, spiking latency exactly when FPS is low.
    // Adaptive cadence: every 24th frame (~0.5s @45fps) when healthy, backing off
    // to every 44th when FPS dips under the floor — the detector can never be why
    // FPS drops, and the phone is still caught via the streak + per-frame capture.
    // Every 12th frame (~267ms @45fps) when healthy — tight enough that a phone
    // flashed for well under a second is sampled at least twice. Backs off to every
    // 44th only when FPS dips under the floor, so the detector can never be the cause
    // of an FPS drop. Stride MUST stay a multiple of 4 (see stagger note above).
    const phoneStride = _emaFps <= FPS_FLOOR ? 44 : 12
    if (_frameN % phoneStride === 2) runPhoneDetector(now)

    // Behavioural accumulators — pure arithmetic on already-computed data, run
    // every frame (zero added model cost).
    updateCovert(now)
    updateFraming(now, faceResult)

    const curHandCount = _lastHands.landmarks.length

    // Intruder sustain — 2+ faces must persist ~1s before firing (filters detector glitches)
    if (_lastFaceCount >= 2) {
      if (_intruderSince === 0) _intruderSince = now
    } else {
      _intruderSince = 0
    }
    const intruderSustained = _intruderSince !== 0 && (now - _intruderSince) > 1000

    // window.__finalizing is set by the dashboard's End Session flow: once the
    // session is being finalized, NO new incidents are allowed to start — the
    // door is closed while incidents already in the pipeline finish. This is what
    // lets the finalization gate ever reach queue_depth==0 (a fresh trigger would
    // otherwise keep re-opening the queue).
    if (!_captureActive && !window.__finalizing && performance.now() > _captureCooldownUntil) {
      const handTrigger  = curHandCount > 0 && isLookingAtHand()
      // Fire on a sustained sighting (streak) OR one HIGH-confidence sighting, so a
      // brief phone that the detector only catches once still triggers a capture.
      let   phoneTrigger = _phoneVisible && (_phoneStreak >= PHONE_FIRE_STREAK ||
                           (_phoneScore >= PHONE_STRONG_SCORE && _phoneStreak >= 1))
      // Same-spot debounce: a phone that never moves from a spot we already
      // captured is almost certainly a FIXED object (e.g. a phone pictured on a
      // poster) the LLM has judged — suppress repeat fires for a while. A phone
      // that MOVES to a new position re-fires normally. Saves the LLM quota.
      const pc = phoneCenterNow()
      if (phoneTrigger && pc && _lastFiredPhoneCenter &&
          (now - _lastFiredPhoneAt) < PHONE_SAME_SPOT_HOLD &&
          Math.abs(pc.x - _lastFiredPhoneCenter.x) < PHONE_SAME_SPOT_TOL &&
          Math.abs(pc.y - _lastFiredPhoneCenter.y) < PHONE_SAME_SPOT_TOL) {
        phoneTrigger = false
      }
      const covertTrig  = covertTriggered(now)
      const framingTrig = _framingHidden && (now - _framingFiredAt) > FRAMING_REARM_MS
      if (phoneTrigger || intruderSustained || handTrigger || covertTrig || framingTrig) {
        // Precedence: concrete signals first (visible phone, intruder), then the
        // wrist-look heuristic, then the review-only behavioural signals (covert
        // look-down, then framing/torso-hidden) which fire only if nothing
        // concrete did.
        _captureReason = phoneTrigger      ? 'phone_detected_realtime'
                       : intruderSustained ? 'intruder_detected'
                       : handTrigger       ? 'hand_visible_and_gaze_toward_wrist'
                       : covertTrig        ? 'covert_device_suspected'
                       :                     'framing_torso_hidden'
        if (_captureReason === 'phone_detected_realtime' && pc) {
          _lastFiredPhoneCenter = pc; _lastFiredPhoneAt = now   // remember this spot
        }
        if (_captureReason === 'covert_device_suspected') {
          _covertGlances = []; _downStart = 0   // reset so it doesn't immediately re-fire
        }
        if (_captureReason === 'framing_torso_hidden') {
          _framingFiredAt = now; _framingSince = 0; _framingHidden = false
        }
        if (_sig.events[_captureReason] != null) _sig.events[_captureReason]++   // Layer-1 event tally
        startCapture()
      }
    }
    _prevHandCount = curHandCount

    render(gs, _lastDetections, _lastHands, now)

    // Layer 1 — log this frame's light signals into the continuous buffer. Runs
    // AFTER render so it captures the state actually shown. Pure arithmetic +
    // one tiny 32×18 frame-diff — it can never be why FPS drops.
    logSignalFrame(now, gs, computeMotion())
  } catch (e) {
    console.error('[loop] error:', e)
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────
;(async () => {
  try {
    await initCamera()
    await initModels()
    loadOverlay.classList.add('hidden')
    console.log('[boot] ready — CPU only, wrist capture → Go API')
    requestAnimationFrame(loop)
  } catch (e) {
    console.error(e)
    showError(
      e.name === 'NotAllowedError'
        ? 'Camera access denied.\nAllow camera and reload.'
        : `Init failed:\n${e.message}`
    )
  }
})()
