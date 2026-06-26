# AI Exam Proctor

A browser-based AI proctoring system that watches a candidate's webcam during an online exam, detects suspicious behaviour **entirely in the browser**, and escalates real incidents to a vision-LLM for a written report. Built for low cost and privacy: detection runs client-side, and only flagged incidents (not the full video) are ever sent to the backend.

## How it works

```
Browser (candidate)                         Go API server                    Cloud
─────────────────────                       ─────────────                    ─────
webcam ─▶ ONNX models (in-browser)
          • gaze / head pose  (RT-DETR)
          • phone + person    (YOLOv8)
              │
              │ on suspicious event → POST incident (frames + signals)
              ▼
        /incident  ──────────────▶  Go server
                                     • dedupes + stores incident
                                     • calls Groq or Gemini vision LLM
                                       to write a human-readable report
                                     • persists to Supabase ───────────▶ Supabase (Postgres)
                                                                          dashboard reads from here
```

- **Client-side detection** — `onnxruntime-web` runs RT-DETR (gaze/pose) and YOLOv8 (phone/person) in a Web Worker, so video never leaves the device until something is flagged.
- **Incident reporting** — the Go backend forwards flagged frames to a vision LLM (**Groq** primary, **Gemini** fallback) which returns a structured, per-frame report.
- **Persistence + dashboard** — incidents are written to **Supabase**; a live dashboard surfaces them in real time.

## Stack

**Frontend:** Vanilla JS + Vite · `onnxruntime-web` (WebAssembly/SIMD) · Web Workers
**Backend:** Go (stdlib `net/http`, zero framework) · Groq + Gemini vision APIs · Supabase REST
**Models:** RT-DETR-X, YOLOv8n (exported to ONNX)

## Repo layout

| Path | What |
|---|---|
| `api-server/` | Go incident API — LLM calls, Supabase persistence (`main.go`) |
| `api-server/supabase_schema.sql` | Postgres schema for the incidents table |
| `web-proctor/` | Vite frontend — in-browser detection (`src/`) |
| `proctor_lite.py` | Standalone Python proctor (single-file reference implementation) |

## Running it

> **Model weights and `node_modules` are not committed** (they're large / regenerable). Export the ONNX models with `web-proctor/export_onnx.py` and place them under `web-proctor/public/`.

**Backend:**
```bash
cd api-server
cp .env.example .env        # fill in Supabase + Groq/Gemini keys
go run main.go              # serves on :8080
```

**Frontend:**
```bash
cd web-proctor
npm install
npm run dev                 # Vite dev server
```

## Security

All secrets are read from environment variables (`os.Getenv`) and loaded from a local `.env` that is **gitignored** — no keys are ever committed. See [`api-server/.env.example`](api-server/.env.example) for the required variables.

---

Built by **Muhammad Fadil Falak Awan** — [portfolio](https://fadil-awan.vercel.app) · [github.com/fadil013](https://github.com/fadil013)
