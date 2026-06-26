// ── MediaPipe landmark indices ─────────────────────────────────────────────
const LI_C=468,LI_T=469,LI_R=470,LI_B=471,LI_L=472
const RI_C=473,RI_T=474,RI_R=475,RI_B=476,RI_L=477
const LE_O=33, LE_I=133, LE_T=159, LE_B=145
const RE_I=362, RE_O=263, RE_T=386, RE_B=374
const NOSE=1, LCHEEK=234, RCHEEK=454, FORE=10, CHIN=152

// ── Thresholds ────────────────────────────────────────────────────────────
const HEAD_YAW     = 16
const HEAD_PITCH_U = 3      // degrees UP
const HEAD_PITCH_D = 15     // degrees DOWN — natural forward-facing posture reads ~10-12, need margin
const IRIS_H       = 0.10
const HEAD_W       = 0.60
const IRIS_W       = 0.40
const FIRE_H       = 0.63
export const FIRE_THRESH = FIRE_H
const IRIS_ONLY_H  = 0.26   // raised → less sensitive to small lateral glances
const IRIS_ONLY_V  = 0.28   // vertical iris-only — requires clear upward eye roll
const SMOOTH_N     = 3      // median of 3 → 2-frame response at 90fps, still stable

// ── Smoothing — matches Python deque(maxlen=5) ────────────────────────────
const _yQ=[], _pQ=[], _hQ=[], _vQ=[]
function _median(q) {
  const s=[...q].sort((a,b)=>a-b), m=s.length>>1
  return s.length&1 ? s[m] : (s[m-1]+s[m])*0.5
}
function _push(q,v) { q.push(v); if(q.length>SMOOTH_N) q.shift(); return _median(q) }

// ── Settle period (web-only: 30 frames for MediaPipe to stabilise) ─────────
// Python has no calibration — it just starts immediately with raw values.
// We wait 30 frames so the first result is never garbage from model warmup.
const SETTLE_FRAMES = 15
let _settleCount = 0, _settled = false

export function resetCalibration() {
  _settleCount=0; _settled=false
  _yQ.length=0; _pQ.length=0; _hQ.length=0; _vQ.length=0
  console.log('[gaze] reset')
}
export function isCalibrated() { return _settled }
export function calProgress()  { return Math.min(_settleCount/SETTLE_FRAMES, 1) }

// ── Mirror detection ───────────────────────────────────────────────────────
function isMirrored(lm) { return lm[33].x < lm[263].x }

// ── Head pose — exact Python head_pose() ──────────────────────────────────
function headPose(lm) {
  const nose=lm[NOSE], lc=lm[LCHEEK], rc=lm[RCHEEK], fore=lm[FORE], chin=lm[CHIN]
  const midX  = (lc.x+rc.x)*0.5
  const halfW = Math.max(Math.abs(lc.x-rc.x)*0.5, 1e-6)
  let yaw = Math.max(-1,Math.min(1,(nose.x-midX)/halfW))*90
  const faceH = Math.max(Math.abs(chin.y-fore.y), 1e-6)
  const pitch = ((nose.y-fore.y)/faceH - 0.48)*120
  if (isMirrored(lm)) yaw = -yaw
  return { yaw, pitch }
}

// ── Iris offsets — exact Python iris_offsets() ────────────────────────────
// h > 0 → RIGHT  |  h < 0 → LEFT  |  v > 0 → DOWN  |  v < 0 → UP
function irisOffsets(lm) {
  function _h(o,i,ic) {
    const lo=Math.min(lm[o].x,lm[i].x), hi=Math.max(lm[o].x,lm[i].x), ew=hi-lo
    return ew>0.005 ? (lm[ic].x-(lo+hi)*0.5)/ew : 0
  }
  function _v(t,b,ic) {
    const eh=lm[b].y-lm[t].y
    return eh>0.005 ? (lm[ic].y-(lm[t].y+lm[b].y)*0.5)/eh : 0
  }
  let h = (_h(LE_O,LE_I,LI_C)+_h(RE_I,RE_O,RI_C))*0.5
  const v = (_v(LE_T,LE_B,LI_C)+_v(RE_T,RE_B,RI_C))*0.5
  if (isMirrored(lm)) h = -h
  return { h, v }
}

// ── Gaze direction ────────────────────────────────────────────────────────
function gazeDirection(yaw, pitch, h, v) {
  const clamp = (x,lo,hi) => Math.min(Math.max(x,lo),hi)

  // ── Vertical: head pitch primary ─────────────────────────────────
  if (pitch < -HEAD_PITCH_U) return { dir:'UP',   score:1, sL:0, sR:0, sU:1, sD:0 }
  if (pitch >  HEAD_PITCH_D) return { dir:'DOWN', score:1, sL:0, sR:0, sU:0, sD:1 }

  // ── Vertical: iris UP fast path (eyes-only upward glance, pitch guard prevents DOWN false fire)
  if (v < -IRIS_ONLY_V && pitch <= 2) return { dir:'UP', score:1, sL:0, sR:0, sU:1, sD:0 }

  // ── Horizontal: iris-only fast path ──────────────────────────────
  if (h < -IRIS_ONLY_H) return { dir:'LEFT',  score:1, sL:1, sR:0, sU:0, sD:0 }
  if (h >  IRIS_ONLY_H) return { dir:'RIGHT', score:1, sL:0, sR:1, sU:0, sD:0 }

  // ── Horizontal: combined head + iris ─────────────────────────────
  const hL = clamp( yaw/HEAD_YAW, 0,1)
  const hR = clamp(-yaw/HEAD_YAW, 0,1)
  const iL = clamp(-h/IRIS_H,     0,1)
  const iR = clamp( h/IRIS_H,     0,1)
  const sL = hL*HEAD_W + iL*IRIS_W
  const sR = hR*HEAD_W + iR*IRIS_W

  const best = Math.max(sL, sR)
  if (best < FIRE_H) return { dir:'AHEAD', score:best, sL, sR, sU:0, sD:0 }
  return { dir: sL >= sR ? 'LEFT' : 'RIGHT', score:best, sL, sR, sU:0, sD:0 }
}

// ── Draw iris ring ─────────────────────────────────────────────────────────
export function drawIris(ctx, lm, c, t, r, b, l, color, ox, oy, sx, sy) {
  const px=i=>ox+lm[i].x*sx, py=i=>oy+lm[i].y*sy
  const cx=px(c), cy=py(c)
  const rad=Math.max((Math.hypot(cx-px(t),cy-py(t))+Math.hypot(cx-px(r),cy-py(r))+
             Math.hypot(cx-px(b),cy-py(b))+Math.hypot(cx-px(l),cy-py(l)))*0.25, 4)
  ctx.beginPath(); ctx.arc(cx,cy,rad,0,Math.PI*2)
  ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.stroke()
  ctx.beginPath(); ctx.arc(cx,cy,2.5,0,Math.PI*2)
  ctx.fillStyle=color; ctx.fill()
}

// ── Process one FaceLandmarker result ─────────────────────────────────────
export function processGaze(result) {
  const faces = result.faceLandmarks
  if (!faces || faces.length === 0) {
    return {
      dir:'NO_FACE', score:0, faces:[], mirrored:false,
      yaw:0, pitch:0, h:0, v:0, dYaw:0, dPitch:0, dH:0, dV:0,
      sL:0, sR:0, sU:0, sD:0, calibrated:_settled, calPct:calProgress()
    }
  }

  const lm = faces[0]
  const { yaw, pitch } = headPose(lm)
  const { h, v }       = irisOffsets(lm)

  const sy = _push(_yQ, yaw)
  const sp = _push(_pQ, pitch)
  const sh = _push(_hQ, h)
  const sv = _push(_vQ, v)

  // Settle — suppress alerts for first 30 frames while MediaPipe stabilises
  if (!_settled) {
    _settleCount++
    if (_settleCount >= SETTLE_FRAMES) {
      _settled = true
      console.log(`[gaze] settled  yaw=${sy.toFixed(1)} pitch=${sp.toFixed(1)} h=${sh.toFixed(3)} v=${sv.toFixed(3)} mirrored=${isMirrored(lm)}`)
    }
    return {
      dir:'AHEAD', score:0, faces, mirrored:isMirrored(lm),
      yaw:sy, pitch:sp, h:sh, v:sv, dYaw:0, dPitch:0, dH:0, dV:0,
      sL:0, sR:0, sU:0, sD:0, calibrated:false, calPct:calProgress()
    }
  }

  const { dir, score, sL, sR, sU, sD } = gazeDirection(sy, sp, sh, sv)
  return {
    dir, score, faces, mirrored:isMirrored(lm),
    yaw:sy, pitch:sp, h:sh, v:sv, dYaw:sy, dPitch:sp, dH:sh, dV:sv,
    sL, sR, sU, sD, calibrated:true, calPct:1
  }
}

// ── Landmark exports ───────────────────────────────────────────────────────
export { LI_C,LI_T,LI_R,LI_B,LI_L, RI_C,RI_T,RI_R,RI_B,RI_L }
