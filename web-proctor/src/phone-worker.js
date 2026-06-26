// Runs in a Web Worker — all ORT operations are off the main thread
import * as ort from 'onnxruntime-web'

const PHONE_CLS  = 67
const PHONE_CONF = 0.55
const INPUT_SIZE = 640
const IOU_THRESH = 0.45

let session     = null
let _numBoxes   = 300
let _transposed = false
let _provider   = 'wasm'

// ── IOU + NMS ──────────────────────────────────────────────────────────────
function iou(a, b) {
  const ix1=Math.max(a.x1,b.x1), iy1=Math.max(a.y1,b.y1)
  const ix2=Math.min(a.x2,b.x2), iy2=Math.min(a.y2,b.y2)
  const inter=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1)
  return inter/((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter+1e-6)
}
function nms(boxes) {
  boxes.sort((a,b)=>b.conf-a.conf)
  const keep=[], gone=new Uint8Array(boxes.length)
  for (let i=0;i<boxes.length;i++){
    if(gone[i]) continue; keep.push(boxes[i])
    for(let j=i+1;j<boxes.length;j++) if(!gone[j]&&iou(boxes[i],boxes[j])>IOU_THRESH) gone[j]=1
  }
  return keep
}

// Inverse-letterbox: map boxes from 640×640 letterboxed space back to video coords
function parseOutput(data, scale, padX, padY) {
  const N=_numBoxes, raw=[]
  for (let i=0; i<N; i++) {
    let x1,y1,x2,y2,conf,cls
    if (_transposed) {
      x1=data[0*N+i]; y1=data[1*N+i]; x2=data[2*N+i]; y2=data[3*N+i]
      conf=data[4*N+i]; cls=Math.round(data[5*N+i])
    } else {
      x1=data[i*6]; y1=data[i*6+1]; x2=data[i*6+2]; y2=data[i*6+3]
      conf=data[i*6+4]; cls=Math.round(data[i*6+5])
    }
    if (conf<PHONE_CONF||cls!==PHONE_CLS) continue
    // Undo letterbox padding and scale to get original video pixel coordinates
    raw.push({
      x1: (x1-padX)/scale, y1: (y1-padY)/scale,
      x2: (x2-padX)/scale, y2: (y2-padY)/scale,
      conf
    })
  }
  return nms(raw)
}

// ── Load ──────────────────────────────────────────────────────────────────
async function loadModel(modelPath) {
  ort.env.wasm.numThreads = Math.max(Math.min(navigator.hardwareConcurrency ?? 4, 8), 2)
  ort.env.wasm.simd = true
  console.log('[phone-worker] loading wasm, threads:', ort.env.wasm.numThreads)

  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  })
  _provider = 'wasm'

  const dummy = new ort.Tensor('float32',
    new Float32Array(3*INPUT_SIZE*INPUT_SIZE), [1,3,INPUT_SIZE,INPUT_SIZE])
  const feeds = { images: dummy }
  if (session.inputNames.includes('orig_img_size'))
    feeds.orig_img_size = new ort.Tensor('int64',
      BigInt64Array.from([BigInt(640),BigInt(640)]), [1,2])
  const warm  = await session.run(feeds)
  const out0  = warm[session.outputNames[0]]
  _transposed = out0.dims[1] === 6
  _numBoxes   = _transposed ? out0.dims[2] : out0.dims[1]
  console.log(`[phone-worker] ready  provider=${_provider} boxes=${_numBoxes}`)
}

// ── Detect ────────────────────────────────────────────────────────────────
async function detect(rgba, scale, padX, padY) {
  const N=INPUT_SIZE*INPUT_SIZE
  const t=new Float32Array(3*N)
  const u8=new Uint8Array(rgba)
  for (let i=0;i<N;i++) {
    t[i]     = u8[i*4]   / 255
    t[i+N]   = u8[i*4+1] / 255
    t[i+N*2] = u8[i*4+2] / 255
  }
  const tensor = new ort.Tensor('float32', t, [1,3,INPUT_SIZE,INPUT_SIZE])
  const feeds  = { images: tensor }
  if (session.inputNames.includes('orig_img_size'))
    feeds.orig_img_size = new ort.Tensor('int64',
      BigInt64Array.from([BigInt(640),BigInt(640)]), [1,2])
  const res = await session.run(feeds)
  const out = res[session.outputNames[0]]
  return parseOutput(out.data, scale, padX, padY)
}

// ── Message router ─────────────────────────────────────────────────────────
self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    try {
      await loadModel(data.modelPath)
      self.postMessage({ type:'ready', provider:_provider })
    } catch (e) {
      self.postMessage({ type:'error', message:e.message })
    }
  }
  if (data.type === 'detect') {
    try {
      const boxes = await detect(data.rgba, data.scale, data.padX, data.padY)
      self.postMessage({ type:'result', boxes })
    } catch {
      self.postMessage({ type:'result', boxes:[] })
    }
  }
}
