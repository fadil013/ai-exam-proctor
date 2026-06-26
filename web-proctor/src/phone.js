// All ORT inference runs in a Web Worker — main thread never blocked
const INPUT_SIZE   = 640
const MIN_INTERVAL = 2000   // ms between phone checks

let _worker    = null
let _ready     = false
let _inferring = false
let _lastRun   = 0
let _provider  = 'wasm'
let _canvas    = null
let _ctx       = null

let _loadResolve = null, _loadReject = null
let _detectResolve = null

function _initCanvas() {
  _canvas       = document.createElement('canvas')
  _canvas.width = _canvas.height = INPUT_SIZE
  _ctx = _canvas.getContext('2d', { willReadFrequently: true })
}

// Letterbox: scale video to fit 640×640 without stretching, black-pad remainder.
// Returns { scale, padX, padY } so the worker can invert the transform on boxes.
function _letterbox(video) {
  const vW = video.videoWidth  || INPUT_SIZE
  const vH = video.videoHeight || INPUT_SIZE
  const scale = Math.min(INPUT_SIZE / vW, INPUT_SIZE / vH)
  const dW    = Math.round(vW * scale)
  const dH    = Math.round(vH * scale)
  const padX  = (INPUT_SIZE - dW) / 2
  const padY  = (INPUT_SIZE - dH) / 2

  _ctx.fillStyle = '#000'
  _ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  _ctx.drawImage(video, padX, padY, dW, dH)

  return { scale, padX, padY }
}

export function loadPhoneModel(modelPath = '/rtdetr-x.onnx') {
  _initCanvas()
  return new Promise((resolve, reject) => {
    _loadResolve = resolve
    _loadReject  = reject

    _worker = new Worker(new URL('./phone-worker.js', import.meta.url), { type: 'module' })
    _worker.onmessage = ({ data }) => {
      if (data.type === 'ready') {
        _ready    = true
        _provider = data.provider
        if (_loadResolve) { _loadResolve(data.provider); _loadResolve = null }
      }
      if (data.type === 'result') {
        _inferring = false
        if (_detectResolve) { _detectResolve(data.boxes); _detectResolve = null }
      }
      if (data.type === 'error') {
        if (_loadReject) { _loadReject(new Error(data.message)); _loadReject = null }
      }
    }
    _worker.onerror = (e) => {
      if (_loadReject) { _loadReject(e); _loadReject = null }
    }
    _worker.postMessage({ type: 'load', modelPath })
  })
}

export function detectPhones(video) {
  const now = performance.now()
  if (!_ready || _inferring || (now - _lastRun) < MIN_INTERVAL)
    return Promise.resolve(null)

  _inferring = true
  _lastRun   = now

  // Letterboxed grab → preserves aspect ratio → better RT-DETR-x accuracy
  const { scale, padX, padY } = _letterbox(video)
  const imgData = _ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
  const buf     = imgData.data.buffer

  return new Promise(resolve => {
    _detectResolve = resolve
    _worker.postMessage(
      { type:'detect', rgba:buf, videoW:video.videoWidth, videoH:video.videoHeight, scale, padX, padY },
      [buf]
    )
  })
}

export function modelReady()    { return _ready }
export function modelProvider() { return _provider }
