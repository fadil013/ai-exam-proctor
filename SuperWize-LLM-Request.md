# Request for SuperWize LLM Access — SLAC Cheating-Prevention System

**To:** Dr. Sohail
**Date:** 3 July 2026

---

## The request

I'm requesting access to the **SuperWize LLM** for the **SLAC exam cheating-prevention system**. The system is fully built and working — the only remaining blocker is the LLM tier. I'm currently on **free-tier models (Groq / Gemini)**, and I've hit their limits.

## Why I need it

The free-tier models are failing on the two things that matter most:

1. **Rate limits** — free tiers cap requests per minute/day. During a live exam, the quota runs out and **real flagged incidents go unanalysed** (they get pushed to slow manual review). The automation breaks under load exactly when it's needed.
2. **Lower vision accuracy** — the free 17B model **misses small objects** (phones at an angle, earbuds, notes) and **confuses paper with phone**. In live testing it even returned "No Violation" on a frame where the candidate was clearly holding a phone.
3. **Image cap** — the free model accepts only 5 images per request, but each incident captures 6 frames, so **one frame is never analysed**.

## What it's needed for

The LLM is the **vision layer** of the system — it looks at each flagged webcam frame and reports what it sees (phone, notes, second person, etc.). A deterministic rule engine then turns those observations into a verdict. So:

- The LLM **doesn't decide the verdict** — but it supplies the evidence the verdict is built from.
- **If the LLM is weak or unavailable, the evidence is weak or unavailable, and the report is wrong.**

Specifically, the SuperWize LLM is needed to:
- **Analyse every flagged incident reliably** (no quota failures during exams).
- **Accurately identify cheating aids** — phones, earbuds, smartwatches, hidden notes, second person.
- **See all captured frames** per incident (no dropped frames).
- Produce **accurate, defensible reports** examiners can trust and act on.

## What it enables

With the SuperWize LLM in place, the system moves from *"a demo that stalls under load"* to a **deployable, accurate, and defensible** proctoring tool for SLAC — every incident analysed, far fewer missed phones/notes, and far fewer false "needs review" cases.

## Note on usage & cost

Usage is modest — the system sends **one request per flagged incident only** (not per frame, and never for the clean parts of the exam). Most of an exam produces zero LLM calls. The need is for **reliability and accuracy**, not high volume.

---

**In one line:** the SLAC system is built and production-ready; I need the **SuperWize LLM** because the free-tier models are rate-limited (incidents go unanalysed) and inaccurate at vision (phones and notes are missed) — and the LLM is the layer that actually detects the cheating.

I'm happy to give a live demo of the system and show the exact points where the free tier fails.

**— Muhammad Fadil Falak**
