import { FaceLandmarker, FaceDetector, HandLandmarker, PoseLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
import { processGaze, drawIris, resetCalibration,
         LI_C,LI_T,LI_R,LI_B,LI_L,
         RI_C,RI_T,RI_R,RI_B,RI_L } from './gaze.js'

const PW = 640, PH = 360

// ── DOM ───────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('main-canvas')
const ctx         = canvas.getContext('2d')
const video       = document.getElementById('video')
const gazeEl      = document.getElementById('gaze-text')
const fpsEl       = document.getElementById('fps-text')
const faceEl      = document.getElementById('face-text')
const handEl      = document.getElementById('hand-text')
const handBanner  = document.getElementById('hand-banner')
const faceBanner  = document.getElementById('face-banner')
const loadOverlay = document.getElementById('loading-overlay')
const loadMsg     = document.getElementById('loading-msg')
const loadSub     = document.getElementById('loading-sub')
const errOverlay  = document.getElementById('error-overlay')
const errMsg      = document.getElementById('error-msg')

// ── State ─────────────────────────────────────────────────────────────────
let faceLandmarker = null   // single instance — numFaces=4, same settings for all
let faceDetector   = null
let handLandmarker = null
let poseLandmarker = null
let fpsAcc = 0, fpsT0 = performance.now()
let _debug   = false
let _frameN  = 0
let _lastAllFaces   = []
let _lastPrimaryIdx = 0
let _lastGs         = null
let _lastDetections = []
let _lastHands      = { landmarks: [], handedness: [] }
let _lastPoses      = []

// ── Keys ──────────────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  if (e.key === 'd' || e.key === 'D') _debug = !_debug
  if (e.key === 'c' || e.key === 'C') { resetCalibration(); console.log('[main] recalibrating') }
})

function setLoad(msg, sub = '') { loadMsg.textContent = msg; loadSub.textContent = sub }
function showError(msg) { errMsg.textContent = msg; errOverlay.classList.remove('hidden') }

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

// ── MediaPipe init ─────────────────────────────────────────────────────────
async function initModels() {
  setLoad('Loading MediaPipe…', 'Downloads models on first run')
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm')

  // All 4 models — same CPU delegate, same quality settings across all people
  ;[faceLandmarker, faceDetector, handLandmarker, poseLandmarker] = await Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'CPU'
      },
      outputFaceBlendshapes: false,
      runningMode: 'VIDEO',
      numFaces: 4,
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
      minDetectionConfidence: 0.3
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      numHands: 4
    }),
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'CPU'
      },
      runningMode: 'VIDEO',
      numPoses: 4,
      minPoseDetectionConfidence: 0.3,
      minPosePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
      outputSegmentationMasks: false
    })
  ])
}

// ── Primary face: largest face area = closest to camera = candidate ────────
function findPrimaryIdx(allFaces) {
  let bestIdx = 0, bestArea = 0
  for (let i = 0; i < allFaces.length; i++) {
    const lm = allFaces[i]
    const w = Math.abs(lm[234].x - lm[454].x)
    const h = Math.abs(lm[10].y  - lm[152].y)
    const area = w * h
    if (area > bestArea) { bestArea = area; bestIdx = i }
  }
  return bestIdx
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

// ── Draw hand ─────────────────────────────────────────────────────────────
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

// ── Pose skeleton ─────────────────────────────────────────────────────────
const POSE_CONN = [
  [11,12],
  [11,13],[13,15],[15,17],[15,19],[17,19],
  [12,14],[14,16],[16,18],[16,20],[18,20],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[27,31],
  [24,26],[26,28],[28,30],[28,32],
]
const POSE_COLORS = [
  '#00bcd4','#ff9800','#4caf50','#e040fb','#f44336','#ffeb3b',
  '#2196f3','#76ff03','#ff4081','#18ffff','#ff6d00','#b388ff',
  '#69f0ae','#ff80ab','#82b1ff','#ccff90','#ea80fc','#80d8ff',
  '#a7ffeb','#ffd180'
]

function drawPose(lms, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 2
  for (const [a, b] of POSE_CONN) {
    const pa = lms[a], pb = lms[b]
    if (!pa || !pb) continue
    if ((pa.visibility ?? 1) < 0.3 || (pb.visibility ?? 1) < 0.3) continue
    ctx.beginPath()
    ctx.moveTo(PW + pa.x * PW, pa.y * PH)
    ctx.lineTo(PW + pb.x * PW, pb.y * PH)
    ctx.stroke()
  }
  for (let i = 11; i <= 32; i++) {
    const p = lms[i]
    if (!p || (p.visibility ?? 1) < 0.3) continue
    ctx.beginPath(); ctx.arc(PW + p.x * PW, p.y * PH, 3, 0, Math.PI*2)
    ctx.fillStyle = color; ctx.fill()
  }
}

// ── Render ────────────────────────────────────────────────────────────────
function render(gs, detections, hands, allFaces, primaryIdx, poses, now) {
  const { dir, calPct, calibrated } = gs
  const numFaces = detections ? detections.length : 0
  const numHands = hands && hands.landmarks ? hands.landmarks.length : 0

  ctx.drawImage(video, 0,  0, PW, PH)
  ctx.drawImage(video, PW, 0, PW, PH)

  // ── All faces: primary gets green/red iris, others get purple ────────
  if (allFaces.length > 0) {
    for (let i = 0; i < allFaces.length; i++) {
      const lm = allFaces[i]
      if (i === primaryIdx) {
        const col = (dir !== 'AHEAD' && dir !== 'NO_FACE') ? '#e53935' : '#43a047'
        drawIris(ctx, lm, LI_C,LI_T,LI_R,LI_B,LI_L, col, PW, 0, PW, PH)
        drawIris(ctx, lm, RI_C,RI_T,RI_R,RI_B,RI_L, col, PW, 0, PW, PH)
      } else {
        drawIris(ctx, lm, LI_C,LI_T,LI_R,LI_B,LI_L, '#ab47bc', PW, 0, PW, PH)
        drawIris(ctx, lm, RI_C,RI_T,RI_R,RI_B,RI_L, '#ab47bc', PW, 0, PW, PH)
      }
    }
  }

  // ── Pose skeletons ───────────────────────────────────────────────────
  for (let i = 0; i < poses.length; i++) {
    drawPose(poses[i], POSE_COLORS[i % POSE_COLORS.length])
  }

  // ── Intruder boxes ───────────────────────────────────────────────────
  if (detections && detections.length > 0) {
    let primaryDetIdx = 0
    if (allFaces.length > 0) {
      const lm = allFaces[primaryIdx]
      const fx = lm[1].x, fy = lm[1].y
      let minDist = Infinity
      for (let i = 0; i < detections.length; i++) {
        const bb = detections[i].boundingBox
        const vW = video.videoWidth || 1280, vH = video.videoHeight || 720
        const cx = (bb.originX + bb.width*0.5) / vW
        const cy = (bb.originY + bb.height*0.5) / vH
        const d = (cx-fx)*(cx-fx) + (cy-fy)*(cy-fy)
        if (d < minDist) { minDist = d; primaryDetIdx = i }
      }
    }
    ctx.font = 'bold 12px Segoe UI,monospace'
    let intId = 1
    for (let i = 0; i < detections.length; i++) {
      if (i === primaryDetIdx) continue
      const bb = detections[i].boundingBox
      const vW = video.videoWidth || 1280, vH = video.videoHeight || 720
      const bx = PW + (bb.originX/vW)*PW, by = (bb.originY/vH)*PH
      const bw = (bb.width/vW)*PW,        bh = (bb.height/vH)*PH
      ctx.strokeStyle = '#e53935'; ctx.lineWidth = 2
      ctx.strokeRect(bx, by, bw, bh)
      ctx.fillStyle = '#e53935'
      ctx.fillText(`! INTRUDER ${intId++}`, bx+4, Math.max(by-5, 12))
    }
  }

  // ── Hands ────────────────────────────────────────────────────────────
  if (hands && hands.landmarks) {
    for (let hi = 0; hi < hands.landmarks.length; hi++) {
      drawHand(hands.landmarks[hi], hands.handedness[hi]?.[0]?.displayName || '?')
    }
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

  // ── Status bar ───────────────────────────────────────────────────────
  if (!calibrated) {
    gazeEl.textContent = `Stabilising… ${Math.round(calPct*100)}%`
    gazeEl.className   = 'status-nof'
  } else if (dir === 'NO_FACE') {
    gazeEl.textContent = 'NO FACE'; gazeEl.className = 'status-nof'
  } else if (dir === 'AHEAD') {
    gazeEl.textContent = 'LOOKING AHEAD'; gazeEl.className = 'status-ok'
  } else {
    gazeEl.textContent = `LOOKING ${dir}`; gazeEl.className = 'status-warn'
  }

  const intruderCount = Math.max(0, numFaces - 1)
  faceEl.textContent = numFaces > 1
    ? `${numFaces} FACES — ${intruderCount} INTRUDER${intruderCount>1?'S':''}!`
    : numFaces === 1 ? '1 FACE' : 'NO FACE'

  if (numHands > 0) {
    handEl.textContent = `${numHands} HAND${numHands>1?'S':''} DETECTED`
    handEl.className = 'status-warn'
  } else {
    handEl.textContent = 'NO HAND'; handEl.className = 'status-ok'
  }

  handBanner.classList.toggle('hidden', numHands === 0)
  faceBanner.classList.toggle('hidden', numFaces <= 1)

  fpsAcc++
  const elapsed = now - fpsT0
  if (elapsed >= 1000) {
    fpsEl.textContent = `${Math.round(fpsAcc * 1000 / elapsed)} FPS`
    fpsAcc = 0; fpsT0 = now
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────
function loop(now) {
  _frameN++
  const fn3 = _frameN % 3
  const fn6 = _frameN % 6

  // FaceLandmarker(4): every 3rd frame — all people, equal settings
  if (fn3 === 0) {
    const r = faceLandmarker.detectForVideo(video, now)
    _lastAllFaces = r.faceLandmarks || []
    if (_lastAllFaces.length > 0) {
      _lastPrimaryIdx = findPrimaryIdx(_lastAllFaces)
      _lastGs = processGaze({ faceLandmarks: [_lastAllFaces[_lastPrimaryIdx]] })
    } else {
      _lastGs = processGaze({ faceLandmarks: [] })
    }
  }

  // HandLandmarker: every 3rd frame offset 1
  if (fn3 === 1) {
    const h = handLandmarker.detectForVideo(video, now)
    _lastHands = { landmarks: h.landmarks || [], handedness: h.handedness || [] }
  }

  // FaceDetector: every 6th frame offset 2
  if (fn6 === 2) {
    const d = faceDetector.detectForVideo(video, now)
    _lastDetections = d.detections || []
  }

  // PoseLandmarker: every 6th frame offset 5
  if (fn6 === 5) {
    const p = poseLandmarker.detectForVideo(video, now)
    _lastPoses = p.landmarks || []
  }

  if (_lastGs) render(_lastGs, _lastDetections, _lastHands, _lastAllFaces, _lastPrimaryIdx, _lastPoses, now)
  requestAnimationFrame(loop)
}

// ── Boot ──────────────────────────────────────────────────────────────────
;(async () => {
  try {
    await initCamera()
    await initModels()
    loadOverlay.classList.add('hidden')
    console.log('[boot] all models ready — 4 people, equal settings, CPU')
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
