#!/usr/bin/env python3
"""
phone_glance_analysis.py  —  Off-frame phone-use PREDICTION from webcam signals.

The premise (the whole point of this file):
    You do NOT need to see the phone to detect the phone. A candidate looking at a
    phone held low / in the lap / below the desk still shows the webcam their face,
    eyes, head-pitch and hands. The phone is invisible; the BEHAVIOUR of looking at
    it is not. This script infers the invisible from the visible — no video is
    recorded or stored, only the light per-second signals the proctor already logs.

Input:
    A session-signals JSON (the `signals` object returned by the API's
    /session-report, or saved by the browser via window.__getSessionSignals()).
    We use its per-second `timeline` rows: {sec, face, gaze, hand, motion, phone}
    and, when present, per-second `pitch` / `yaw` (head angle in degrees).

    Or run with  --demo  to generate a realistic synthetic session (clear phone
    checking + benign confounds) so you can SEE the output with no live data.

Output:
    1. A 6-panel matplotlib PNG  — the "phone-glance signature" visual proof.
    2. A JSON verdict  — per-episode detections + a fused off-frame-phone
       probability, with the evidence spelled out (nothing is a black box).

Usage:
    python phone_glance_analysis.py --demo
    python phone_glance_analysis.py --input session_signals.json
    python phone_glance_analysis.py --input sig.json --out-prefix candidate_042
"""

import argparse
import json
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 chokes on → · °
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless — never opens a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Thresholds (tunable — calibrate on real labelled sessions) ────────────────
PITCH_DOWN_DEG   = 20.0    # head pitched-down beyond this = a "down look" second
PITCH_LAP_DEG    = 28.0    # steeper than this = target is low/close (lap phone, not desk)
NORMAL_PITCH_DEG = 12.0    # ordinary laptop-posture head tilt (the baseline, benign)
DWELL_MIN_S      = 2.0     # a glance shorter than this = keyboard flick, not reading
DWELL_MAX_S      = 14.0    # longer sustained down-look drifts toward note-writing
YAW_LATERAL_DEG  = 18.0    # strong sideways head turn = looking to a side (paper), not straight-down phone


# ─────────────────────────────────────────────────────────────────────────────
# Signal reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_series(signals):
    """Turn the session-signals `timeline` into dense per-second numpy arrays.

    Fills gaps (downsampled/long sessions) by forward-holding the last sample so
    episode timing stays continuous. Returns a dict of aligned 1-Hz arrays.
    """
    tl = signals.get("timeline") or []
    if not tl:
        raise ValueError("session signals contain no timeline rows")

    secs = [int(r.get("sec", i)) for i, r in enumerate(tl)]
    t_end = max(secs)
    n = t_end + 1

    gaze   = ["UNKNOWN"] * n
    face   = np.zeros(n, dtype=float)
    hand   = np.ones(n, dtype=float)          # assume present unless told otherwise
    motion = np.zeros(n, dtype=float)
    phone  = np.zeros(n, dtype=float)
    pitch  = np.full(n, np.nan, dtype=float)
    yaw    = np.full(n, np.nan, dtype=float)

    for r in tl:
        s = int(r.get("sec", 0))
        if not (0 <= s < n):
            continue
        gaze[s]   = r.get("gaze", "UNKNOWN")
        face[s]   = float(r.get("face", 0) or 0)
        hand[s]   = 1.0 if r.get("hand") else 0.0
        motion[s] = float(r.get("motion", 0) or 0)
        phone[s]  = 1.0 if r.get("phone") else 0.0
        if r.get("pitch") is not None:
            pitch[s] = float(r["pitch"])
        if r.get("yaw") is not None:
            yaw[s] = float(r["yaw"])

    # Forward-fill categorical / numeric gaps so contiguous episodes aren't broken.
    last_g = "UNKNOWN"
    for i in range(n):
        if gaze[i] == "UNKNOWN" and i > 0:
            gaze[i] = last_g
        last_g = gaze[i]

    # Pitch fallback: if the buffer predates per-second pitch, synthesise a proxy
    # from the gaze category so the pipeline still runs (clearly weaker signal).
    pitch_is_real = not np.all(np.isnan(pitch))
    for i in range(n):
        if np.isnan(pitch[i]):
            pitch[i] = 25.0 if gaze[i] == "DOWN" else NORMAL_PITCH_DEG
    yaw = np.nan_to_num(yaw, nan=0.0)

    down = np.array(
        [1.0 if (gaze[i] == "DOWN" or pitch[i] >= PITCH_DOWN_DEG) else 0.0 for i in range(n)],
        dtype=float,
    )

    return {
        "n": n, "gaze": gaze, "face": face, "hand": hand, "motion": motion,
        "phone": phone, "pitch": pitch, "yaw": yaw, "down": down,
        "pitch_is_real": pitch_is_real,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode detection + scoring
# ─────────────────────────────────────────────────────────────────────────────
def find_episodes(S, max_gap=1):
    """Contiguous runs of down-looking seconds → episodes (small gaps bridged)."""
    down = S["down"]
    episodes, start, gap = [], None, 0
    for i, d in enumerate(down):
        if d > 0:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > max_gap:
                    episodes.append((start, i - gap))
                    start = None
    if start is not None:
        episodes.append((start, len(down) - 1))
    return [(a, b) for a, b in episodes if b >= a]


def _bump(x, lo, hi):
    """1.0 inside [lo,hi], tapering to 0 outside — rewards the phone dwell band."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / lo)
    return max(0.0, 1.0 - (x - hi) / hi)


def score_episode(S, a, b):
    """Weighted phone-likelihood for one down-look episode, with transparent parts."""
    dur       = (b - a) + 1
    seg_pitch = S["pitch"][a:b + 1]
    seg_yaw   = S["yaw"][a:b + 1]
    seg_hand  = S["hand"][a:b + 1]
    seg_mot   = S["motion"][a:b + 1]

    peak_pitch  = float(np.max(seg_pitch))
    mean_pitch  = float(np.mean(seg_pitch))
    hand_absent = float(np.mean(seg_hand < 0.5))          # 0..1 fraction hand out of frame
    lateral     = float(np.mean(np.abs(seg_yaw)))
    mean_motion = float(np.mean(seg_mot))

    # Component scores (each 0..1) — the honest, inspectable ingredients.
    pitch_score   = 1.0 / (1.0 + math.exp(-(peak_pitch - PITCH_LAP_DEG) / 4.0))
    dwell_score   = _bump(dur, DWELL_MIN_S, DWELL_MAX_S)
    hand_score    = hand_absent
    lateral_pen   = 1.0 - min(1.0, lateral / YAW_LATERAL_DEG)   # straight-down favours phone
    still_score   = 1.0 - min(1.0, mean_motion / 40.0)          # low gross motion = covert

    # Fusion — pitch & dwell dominate; hand-drop is the corroborator; lateral gates.
    raw = (0.34 * pitch_score + 0.26 * dwell_score +
           0.22 * hand_score + 0.10 * still_score + 0.08 * lateral_pen)
    score = raw * (0.55 + 0.45 * lateral_pen)   # a sideways turn suppresses the whole episode

    if score >= 0.6:
        label = "phone-like"
    elif score >= 0.38:
        label = "suspicious"
    else:
        label = "benign"

    return {
        "start_sec": a, "end_sec": b, "duration_s": dur,
        "peak_pitch_deg": round(peak_pitch, 1), "mean_pitch_deg": round(mean_pitch, 1),
        "hand_absent_frac": round(hand_absent, 2), "lateral_yaw_deg": round(lateral, 1),
        "mean_motion": round(mean_motion, 1),
        "components": {
            "pitch": round(pitch_score, 2), "dwell": round(dwell_score, 2),
            "hand_drop": round(hand_score, 2), "stillness": round(still_score, 2),
            "straight_down": round(lateral_pen, 2),
        },
        "score": round(score, 3), "label": label,
    }


def periodicity(S):
    """Autocorrelation of the down-binary signal → is the checking rhythmic?

    A peak at a non-zero lag means the candidate looks down, returns, and looks
    down again on a beat — the signature of repeated phone checking. Returns the
    best lag (seconds) and its normalised strength (0..1).
    """
    x = S["down"] - S["down"].mean()
    if np.allclose(x, 0):
        return {"best_lag_s": 0, "strength": 0.0}
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1.0)
    lo = 3                                   # ignore trivially short lags
    hi = min(len(ac), 90)                    # look up to 90 s between checks
    if hi <= lo:
        return {"best_lag_s": 0, "strength": 0.0}
    seg = ac[lo:hi]
    k = int(np.argmax(seg)) + lo
    return {"best_lag_s": k, "strength": round(float(max(0.0, ac[k])), 3)}


def analyse(signals):
    S = reconstruct_series(signals)
    eps = find_episodes(S)
    scored = [score_episode(S, a, b) for a, b in eps]
    per = periodicity(S)

    phone_like  = [e for e in scored if e["label"] == "phone-like"]
    suspicious  = [e for e in scored if e["label"] == "suspicious"]
    down_frac   = float(np.mean(S["down"]))
    top_scores  = sorted((e["score"] for e in scored), reverse=True)[:3]
    top_mean    = float(np.mean(top_scores)) if top_scores else 0.0

    # Session-level fusion — evidence, not a single rule.
    prob = (0.45 * top_mean +
            0.20 * min(1.0, len(phone_like) / 3.0) +
            0.20 * per["strength"] +
            0.15 * min(1.0, down_frac / 0.25))
    prob = round(min(1.0, prob), 3)

    if prob >= 0.6:
        verdict = "LIKELY off-frame phone use"
    elif prob >= 0.35:
        verdict = "POSSIBLE off-frame phone use — review"
    else:
        verdict = "No off-frame phone signature"

    evidence = []
    if phone_like:
        evidence.append(f"{len(phone_like)} phone-like down-look episode(s)")
    if per["strength"] >= 0.3:
        evidence.append(f"rhythmic checking every ~{per['best_lag_s']}s (autocorr {per['strength']})")
    steep = [e for e in scored if e["peak_pitch_deg"] >= PITCH_LAP_DEG]
    if steep:
        evidence.append(f"{len(steep)} episode(s) with lap-level head pitch (≥{PITCH_LAP_DEG:.0f}°)")
    hd = [e for e in scored if e["hand_absent_frac"] >= 0.5]
    if hd:
        evidence.append(f"{len(hd)} episode(s) with a hand dropping out of frame")
    if down_frac >= 0.15:
        evidence.append(f"{down_frac*100:.0f}% of the session spent looking down")
    if not evidence:
        evidence.append("no discriminating down-look pattern found")

    report = {
        "off_frame_phone_probability": prob,
        "verdict": verdict,
        "pitch_signal": "measured" if S["pitch_is_real"] else "estimated-from-gaze (weaker)",
        "session": {
            "duration_s": S["n"],
            "down_look_fraction": round(down_frac, 3),
            "episodes_total": len(scored),
            "episodes_phone_like": len(phone_like),
            "episodes_suspicious": len(suspicious),
        },
        "periodicity": per,
        "evidence": evidence,
        "episodes": scored,
        "thresholds": {
            "PITCH_DOWN_DEG": PITCH_DOWN_DEG, "PITCH_LAP_DEG": PITCH_LAP_DEG,
            "DWELL_MIN_S": DWELL_MIN_S, "DWELL_MAX_S": DWELL_MAX_S,
        },
    }
    return S, report


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — the 6-panel "phone-glance signature"
# ─────────────────────────────────────────────────────────────────────────────
def plot(S, report, out_png):
    n = S["n"]
    t = np.arange(n)
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"Off-frame phone-use prediction  —  P(phone) = {report['off_frame_phone_probability']:.2f}"
        f"   [{report['verdict']}]",
        fontsize=15, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

    # A ─ Gaze-pitch timeline (the money shot)
    axA = fig.add_subplot(gs[0, :])
    axA.plot(t, S["pitch"], color="#334155", lw=1.3, label="head pitch (°)")
    axA.axhline(PITCH_DOWN_DEG, color="#f59e0b", ls="--", lw=1, label=f"down look ({PITCH_DOWN_DEG:.0f}°)")
    axA.axhline(PITCH_LAP_DEG, color="#dc2626", ls="--", lw=1, label=f"lap level ({PITCH_LAP_DEG:.0f}°)")
    for e in report["episodes"]:
        c = {"phone-like": "#dc2626", "suspicious": "#f59e0b", "benign": "#94a3b8"}[e["label"]]
        axA.axvspan(e["start_sec"], e["end_sec"] + 1, color=c, alpha=0.20)
    axA.set_title("A · Head-pitch timeline — shaded = down-look episodes (red=phone-like)")
    axA.set_xlabel("session time (s)"); axA.set_ylabel("pitch down (°)")
    axA.legend(loc="upper right", fontsize=8); axA.set_ylim(0, max(45, np.max(S["pitch"]) + 5))

    # B ─ Dwell-time histogram
    axB = fig.add_subplot(gs[1, 0])
    durs = [e["duration_s"] for e in report["episodes"]]
    if durs:
        axB.hist(durs, bins=range(1, max(durs) + 2), color="#6366f1", edgecolor="white")
    axB.axvspan(DWELL_MIN_S, DWELL_MAX_S, color="#dc2626", alpha=0.12, label="phone-reading band")
    axB.set_title("B · Down-look dwell times")
    axB.set_xlabel("episode duration (s)"); axB.set_ylabel("count")
    axB.legend(fontsize=8)

    # C ─ Down-look seconds per minute (bursts)
    axC = fig.add_subplot(gs[1, 1])
    mins = int(math.ceil(n / 60.0))
    per_min = [float(np.sum(S["down"][m * 60:(m + 1) * 60])) for m in range(mins)]
    axC.bar(range(mins), per_min, color="#0ea5e9", edgecolor="white")
    axC.set_title("C · Down-look seconds per minute (clustering = suspicious)")
    axC.set_xlabel("minute"); axC.set_ylabel("down-look s")

    # D ─ Autocorrelation (rhythm)
    axD = fig.add_subplot(gs[2, 0])
    x = S["down"] - S["down"].mean()
    if not np.allclose(x, 0):
        ac = np.correlate(x, x, mode="full")[n - 1:]
        ac = ac / (ac[0] if ac[0] != 0 else 1.0)
        lags = np.arange(len(ac))
        axD.plot(lags[:90], ac[:90], color="#334155", lw=1.2)
        bl = report["periodicity"]["best_lag_s"]
        if bl:
            axD.axvline(bl, color="#dc2626", ls="--", lw=1.2,
                        label=f"rhythm ~{bl}s ({report['periodicity']['strength']})")
            axD.legend(fontsize=8)
    axD.set_title("D · Rhythm of looking down (peak = repeated checking)")
    axD.set_xlabel("lag (s)"); axD.set_ylabel("autocorrelation")

    # E ─ Hand-absence vs down-gaze coincidence
    axE = fig.add_subplot(gs[2, 1])
    axE.fill_between(t, 0, S["down"], step="mid", color="#f59e0b", alpha=0.5, label="looking down")
    axE.fill_between(t, 0, (S["hand"] < 0.5).astype(float) * 0.9, step="mid",
                     color="#dc2626", alpha=0.5, label="hand out of frame")
    axE.set_title("E · Hand leaves frame during down-look (holding phone below)")
    axE.set_xlabel("session time (s)"); axE.set_ylabel("active"); axE.set_ylim(0, 1.1)
    axE.legend(fontsize=8, loc="upper right")

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Demo data — a realistic session with phone checking + benign confounds
# ─────────────────────────────────────────────────────────────────────────────
def demo_signals(seed=7):
    rng = np.random.default_rng(seed)
    dur = 360                                 # 6-minute exam
    tl = []
    for s in range(dur):
        # baseline: on-screen, calm
        pitch = NORMAL_PITCH_DEG + rng.normal(0, 2.0)
        yaw   = rng.normal(0, 3.0)
        gaze  = "AHEAD"
        hand  = True
        motion = max(0, rng.normal(8, 4))
        tl.append([s, pitch, yaw, gaze, hand, motion])

    def paint(a, b, pitch, yaw, gaze, hand, motion):
        for s in range(a, min(b, dur)):
            tl[s] = [s, pitch + rng.normal(0, 2), yaw + rng.normal(0, 2),
                     gaze, hand, max(0, motion + rng.normal(0, 3))]

    # PHONE BURST 1 — 3 rhythmic lap glances around 70–110 s (hand drops out)
    for k, a in enumerate((70, 88, 104)):
        paint(a, a + 6, 33, 2, "DOWN", False, 10)   # look down to lap phone
        paint(a + 6, a + 6 + 4, 12, 0, "AHEAD", True, 12)  # back to screen

    # PHONE BURST 2 — 2 longer glances around 230–260 s
    paint(230, 239, 36, 3, "DOWN", False, 9)
    paint(246, 255, 34, 1, "DOWN", False, 9)

    # BENIGN confound 1 — quick keyboard flicks (short, shallow, hands present)
    for a in (150, 152, 175, 178, 300):
        paint(a, a + 1, 18, 1, "DOWN", True, 22)

    # BENIGN confound 2 — reading a paper note to the SIDE (down + strong yaw)
    paint(320, 335, 26, 26, "DOWN", True, 7)

    tl_rows = [{"sec": r[0], "pitch": round(r[1], 1), "yaw": round(r[2], 1),
                "gaze": r[3], "hand": bool(r[4]), "motion": int(r[5]), "phone": False}
               for r in tl]
    return {
        "captured_at": "demo", "duration_secs": dur, "total_frames": dur * 45,
        "avg_fps": 45.0, "sample_secs": 1, "timeline": tl_rows,
        "note": "SYNTHETIC demo data — 2 phone bursts + keyboard/paper confounds",
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Predict off-frame phone use from webcam signals")
    ap.add_argument("--input", help="session-signals JSON file")
    ap.add_argument("--demo", action="store_true", help="use synthetic demo data")
    ap.add_argument("--out-prefix", default="phone_glance",
                    help="output prefix (writes <prefix>.png and <prefix>.json)")
    args = ap.parse_args()

    if args.demo:
        signals = demo_signals()
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        signals = data.get("signals", data)   # accept a raw /session-report too
    else:
        ap.error("give --input <file> or --demo")

    S, report = analyse(signals)
    png = f"{args.out_prefix}.png"
    js  = f"{args.out_prefix}.json"
    plot(S, report, png)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"P(off-frame phone) = {report['off_frame_phone_probability']:.2f}  →  {report['verdict']}")
    print(f"pitch signal: {report['pitch_signal']}")
    print("evidence:")
    for e in report["evidence"]:
        print(f"  · {e}")
    print(f"\nepisodes ({report['session']['episodes_total']}):")
    for e in report["episodes"]:
        print(f"  [{e['start_sec']:>3}-{e['end_sec']:>3}s] {e['duration_s']:>2}s  "
              f"peak={e['peak_pitch_deg']:>4}°  hand_out={e['hand_absent_frac']:.0%}  "
              f"score={e['score']:.2f}  {e['label']}")
    print(f"\nwrote {png} and {js}")


if __name__ == "__main__":
    main()
