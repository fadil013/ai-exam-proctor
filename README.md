# AI Exam Proctor

A browser-based AI proctoring system that watches a candidate's webcam during an online exam, detects suspicious behaviour **entirely in the browser**, and escalates real incidents to a vision-LLM for a written report. Built for low cost and privacy: detection runs client-side, and only flagged incidents (not the full video) are ever sent to the backend.

## How it works

```
Browser (candidate)                         Go API server                    Cloud
─────────────────────                       ─────────────                    ─────
webcam ─▶ MediaPipe Tasks (in-browser, CPU)
          • gaze / head pose   (FaceLandmarker)
          • hands              (HandLandmarker)
          • 2nd person         (FaceDetector)
          • phone              (ObjectDetector / EfficientDet-Lite0)
          • covert / framing   (pure math)
              │
              │ on a triggered incident → POST 6 frames + signals
              ▼
        /incident  ──────────────▶  Go server
                                     • writes a PROVISIONAL verdict instantly
                                     • enriches with a vision LLM (async, serial queue)
                                     • decideVerdict() recomputes the verdict DETERMINISTICALLY
                                     • append-only event log + replayable decision trace
                                     • persists to Supabase ───────────▶ Supabase (Postgres)
                                     • local-disk fallback ◀── dashboard reads session report
              ▲
   End Session → finalization gate (waits for queue quiescence) → locked snapshot → PDF
```

- **Client-side detection** — MediaPipe Tasks (FaceLandmarker / HandLandmarker / FaceDetector / ObjectDetector) run **entirely on the CPU** in the browser, staggered so no frame runs two heavy models; video never leaves the device until something is flagged.
- **Deterministic verdict** — the AI is an **observer only**: it supplies per-frame observations, and `decideVerdict()` recomputes the verdict from rules, so the model can neither hallucinate a conviction nor bury a real object.
- **Incident reporting** — the Go backend forwards flagged frames to a vision LLM (**Groq** primary, **Gemini** fallback) which returns structured per-frame observations.
- **Audit & replay** — every incident has an append-only `events.jsonl` and a stored `ReplayInput`/`DecisionTrace`; `GET /replay?id=` re-derives the verdict deterministically and proves it matches (forensic "why was I flagged?").
- **Finalization** — End Session waits (backend-verified, with a quiescence window) until the AI queue is idle, then freezes a **locked snapshot**; the report is built only from that — reports show **COMPLETE or FAILED**, never "pending".
- **Persistence + dashboard** — incidents are written to **Supabase** with a **local-disk fallback**; a live dashboard surfaces them in real time.

## Stack

**Frontend (live):** Vanilla JS + Vite · **MediaPipe Tasks** (WASM, CPU delegate) · html2pdf
**Backend:** Go (stdlib `net/http`, zero framework) · Groq + Gemini vision APIs · Supabase REST
**Detectors:** FaceLandmarker, HandLandmarker, FaceDetector (BlazeFace), ObjectDetector (EfficientDet-Lite0)

## Repo layout

| Path | What |
|---|---|
| `web-proctor-v1-stable/` | **The live frontend** — MediaPipe in-browser detection (`src/main.js`, `gaze.js`, `dashboard.js`) |
| `api-server/main.go` | Go incident API — deterministic verdict engine, LLM queue, event log, replay, Supabase |
| `api-server/main_test.go` | Verdict/classifier/replay tests |
| `api-server/supabase_schema.sql` | Postgres schema (idempotent, self-migrating) |
| `web-proctor/` | ⚠️ **Legacy** experimental frontend (ONNX RT-DETR/YOLOv8) — superseded by `web-proctor-v1-stable/`, kept for reference |
| `proctor_lite.py` | 🔒 Standalone Python desktop reference (RT-DETR-x + YOLO + MediaPipe) |

## Running it

> **Secrets** go in `api-server/.env` (copy from `.env.example`) — gitignored, never committed.

**Backend:**
```bash
cd api-server
cp .env.example .env        # fill in Supabase + Groq/Gemini keys
go run main.go              # serves on :8080
```

**Frontend (live):**
```bash
cd web-proctor-v1-stable
npm install
npm run dev                 # Vite dev server (MediaPipe models load from CDN)
```

## Security

All secrets are read from environment variables (`os.Getenv`) and loaded from a local `.env` that is **gitignored** — no keys are ever committed. See [`api-server/.env.example`](api-server/.env.example) for the required variables.

---

Built by **Muhammad Fadil Falak Awan** — [portfolio](https://fadil-awan.vercel.app) · [github.com/fadil013](https://github.com/fadil013)
