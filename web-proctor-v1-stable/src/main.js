import { FaceLandmarker, FaceDetector, HandLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
import { processGaze, drawIris, resetCalibration,
         LI_C,LI_T,LI_R,LI_B,LI_L,
         RI_C,RI_T,RI_R,RI_B,RI_L } from './gaze.js'

const PW = 640, PH = 360

// ── Capture config ────────────────────────────────────────────────────────
const API_ENDPOINT        = 'http://localhost:8080/incident'
const CAPTURE_TOTAL       = 6
const CAPTURE_GAP_MS      = 1000
const CAPTURE_COOLDOWN_MS = 5000    // 5s between captures

// ── FPS cap ─────────────────────────────────────────────────────────────
// Cap the loop at ~45 FPS. Running flat-out at 60-70 wastes CPU and triggers
// thermal throttling (which is what caused the sub-30 crashes). A steady cap
// keeps it in the 30-45 band with headroom to spare.
const TARGET_FPS   = 60
const MIN_FRAME_MS = 1000 / TARGET_FPS   // ~16.7ms — phone detector removed, more headroom
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
const loadOverlay = document.getElementById('loading-overlay')
const loadMsg     = document.getElementById('loading-msg')
const loadSub     = document.getElementById('loading-sub')
const errOverlay  = document.getElementById('error-overlay')
const errMsg      = document.getElementById('error-msg')

// ── State ─────────────────────────────────────────────────────────────────
let faceLandmarker = null
let faceDetector   = null
let handLandmarker = null
let fpsAcc = 0, fpsT0 = performance.now()
let _debug = false
let _frameN = 0
let _lastGs         = null
let _lastDetections = []
let _lastHands      = { landmarks: [], handedness: [] }
let _primaryDetIdx  = 0

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
let _lastKnownIntruderBBoxes = []  // persists between frames — fallback when FaceDetector misses

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
async function initCamera() {
  setLoad('Requesting camera…')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: 'user' }, audio: false
  })
  video.srcObject = stream
  await new Promise(r => { video.onloadedmetadata = r })
  await video.play()
}

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
      numHands: 2
    })
  ])
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
const FACE_PAD    = 0.10 // 10% padding — tight to the face so it doesn't blur
                         // notes/phones held right next to the face (was 0.22 and
                         // spilled onto held objects, making them unidentifiable)

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

  // If FaceDetector missed them (extreme angle, head down), use last known position
  bboxes.push(...(currentIntruders.length > 0 ? currentIntruders : _lastKnownIntruderBBoxes))

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

  oc.imageSmoothingEnabled = false
  oc.drawImage(tmp, 0, 0, tW, tH, bx, by, bw, bh)
  oc.imageSmoothingEnabled = true
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
    let primaryIdx = 0
    if (faces && faces.length > 0) {
      const fx = faces[0][1].x, fy = faces[0][1].y
      let minDist = Infinity
      for (let i = 0; i < detections.length; i++) {
        const bb = detections[i].boundingBox
        const cx = (bb.originX + bb.width*0.5) / vW
        const cy = (bb.originY + bb.height*0.5) / vH
        const d = (cx-fx)*(cx-fx) + (cy-fy)*(cy-fy)
        if (d < minDist) { minDist = d; primaryIdx = i }
      }
    }
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

  const sTxt = _captureActive
    ? `CAPTURING ${_captureCount}/${CAPTURE_TOTAL}`
    : (!calibrated ? 'STABILISING' : 'MONITORING')
  setText(statusEl, sTxt); setCls(statusEl, 'stat-v stat-mono')

  fpsAcc++
  const elapsed = now - fpsT0
  if (elapsed >= 1000) {
    fpsEl.textContent = `${Math.round(fpsAcc * 1000 / elapsed)} FPS`
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

// ── Main loop ─────────────────────────────────────────────────────────────
function loop(now) {
  requestAnimationFrame(loop)   // re-queue FIRST — loop never dies even if below throws
  if (now - _lastProcessT < MIN_FRAME_MS) return   // FPS cap → steady, no thermal throttle
  _lastProcessT = now
  try {
    _frameN++

    const faceResult = faceLandmarker.detectForVideo(video, now)
    const gs = processGaze(faceResult)
    _lastGs = gs   // store for snapshotFaceBBox()

    if (_frameN % 4 === 1) {
      const h = handLandmarker.detectForVideo(video, now)
      _lastHands = { landmarks: h.landmarks || [], handedness: h.handedness || [] }
    }

    if (_frameN % 4 === 3) {
      const d = faceDetector.detectForVideo(video, now)
      _lastDetections = d.detections || []
      // Cache intruder bboxes — updated every 4th frame so capture frame 1 has fresh data
      const _vW = video.videoWidth || 1280, _vH = video.videoHeight || 720
      const _intruders = []
      for (let _i = 0; _i < _lastDetections.length; _i++) {
        if (_i === _primaryDetIdx) continue
        try {
          if (isIntruderFace(_lastDetections[_i], _lastHands.landmarks, _vW, _vH)) {
            const _bb = _lastDetections[_i].boundingBox
            _intruders.push(padBBox(_bb.originX/_vW, _bb.originY/_vH,
              (_bb.originX+_bb.width)/_vW, (_bb.originY+_bb.height)/_vH))
          }
        } catch (_) {}
      }
      if (_intruders.length > 0) _lastKnownIntruderBBoxes = _intruders
    }

    const curHandCount = _lastHands.landmarks.length

    // Intruder sustain — 2+ faces must persist ~1s before firing (filters detector glitches)
    if (_lastFaceCount >= 2) {
      if (_intruderSince === 0) _intruderSince = now
    } else {
      _intruderSince = 0
    }
    const intruderSustained = _intruderSince !== 0 && (now - _intruderSince) > 1000

    if (!_captureActive && performance.now() > _captureCooldownUntil) {
      const handTrigger = curHandCount > 0 && isLookingAtHand()
      if (handTrigger || intruderSustained) {
        _captureReason = handTrigger ? 'hand_visible_and_gaze_toward_wrist' : 'intruder_detected'
        startCapture()
      }
    }
    _prevHandCount = curHandCount

    render(gs, _lastDetections, _lastHands, now)
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
