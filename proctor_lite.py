"""
proctor_lite.py  v5
  LEFT  : Student View  (clean)
  RIGHT : Admin Panel   (green iris rings, 4-dir gaze, phone, multi-face)
Press Q or ESC to quit.
"""

import cv2, time, numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PT      = "rtdetr-x.pt"
PHONE_CLS     = 67
PHONE_CONF    = 0.75
IMGSZ         = 640
CAM_ID        = 0
PHONE_LINGER  = 2.0
YOLO_EVERY    = 4        # run RT-DETR-x only every N frames (reduces delay)

# ── Gaze thresholds ─────────────────────────────────────────────────────────────
IRIS_H_THRESH  = 0.10    # iris horizontal contribution threshold
IRIS_V_THRESH  = 0.08    # iris vertical   contribution threshold
HEAD_YAW       = 16.0    # head yaw degrees
HEAD_PITCH_U   = 10.0    # head pitch up degrees
HEAD_PITCH_D   = 13.0    # head pitch down degrees
HEAD_W         = 0.60
IRIS_W         = 0.40
FIRE_THRESH    = 0.72    # combined head+iris score threshold

# Iris-only fire (eyes dart without head moving — must catch this separately)
IRIS_ONLY_H    = 0.18    # iris horizontal alone fires at this deviation
IRIS_ONLY_V    = 0.20    # iris vertical   alone fires at this deviation

SMOOTH_N       = 5       # median smooth window (frames)

# ── Models ─────────────────────────────────────────────────────────────────────
print("\n  Loading RT-DETR-x ...")
detector = YOLO(MODEL_PT)
detector(np.zeros((64,64,3), np.uint8), classes=[PHONE_CLS],
         verbose=False, conf=0.01, imgsz=IMGSZ)
print("  RT-DETR-x ready")
print("  Loading RT-DETR-x class names ...")
print("  Loading MediaPipe FaceMesh ...")
mp_mesh   = mp.solutions.face_mesh
face_mesh = mp_mesh.FaceMesh(
    max_num_faces=4, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("  Ready\n")

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── Panel display size ──────────────────────────────────────────────────────────
PW, PH = 640, 360        # each half panel
TW     = PW * 2

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
GREEN  = ( 40, 220,  40)
RED    = ( 30,  30, 210)
YELLOW = (  0, 210, 255)
WHITE  = (240, 240, 240)
DGRAY  = ( 18,  18,  18)
MGRAY  = ( 55,  55,  55)
LGRAY  = (110, 110, 110)
FONT   = cv2.FONT_HERSHEY_DUPLEX
FONTS  = cv2.FONT_HERSHEY_SIMPLEX

# ── MediaPipe landmark indices ──────────────────────────────────────────────────
# Iris (refine_landmarks=True adds 468-477)
LI_C, LI_T, LI_R, LI_B, LI_L = 468, 469, 470, 471, 472   # left iris
RI_C, RI_T, RI_R, RI_B, RI_L = 473, 474, 475, 476, 477   # right iris
# Eye corners
LE_O, LE_I = 33,  133    # left eye  outer / inner
RE_I, RE_O = 362, 263    # right eye inner / outer
# Eye lid top/bottom
LE_T, LE_B = 159, 145
RE_T, RE_B = 386, 374
# Head pose
NOSE  = 1
LCHEEK, RCHEEK = 234, 454
FORE, CHIN = 10, 152


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAD POSE  (geometry, no solvePnP)
#  yaw  +  → person's LEFT   |  yaw  − → person's RIGHT
#  pitch+  → head DOWN       |  pitch− → head UP
# ═══════════════════════════════════════════════════════════════════════════════
def head_pose(lm):
    nose  = np.array([lm[NOSE].x,    lm[NOSE].y])
    lc    = np.array([lm[LCHEEK].x,  lm[LCHEEK].y])
    rc    = np.array([lm[RCHEEK].x,  lm[RCHEEK].y])
    fore  = np.array([lm[FORE].x,    lm[FORE].y])
    chin  = np.array([lm[CHIN].x,    lm[CHIN].y])
    mid_x  = (lc[0] + rc[0]) / 2.0
    half_w = max(abs(lc[0] - rc[0]) / 2.0, 1e-6)
    yaw    = float(np.clip((nose[0] - mid_x) / half_w, -1.0, 1.0) * 90.0)
    face_h = max(abs(chin[1] - fore[1]), 1e-6)
    pitch  = float(((nose[1] - fore[1]) / face_h - 0.48) * 120.0)
    if lm[33].x < lm[263].x:   # mirrored feed → flip yaw
        yaw = -yaw
    return yaw, pitch


# ═══════════════════════════════════════════════════════════════════════════════
#  IRIS OFFSETS  (person's frame, sign-corrected for mirror)
#  h < 0 → their LEFT  |  h > 0 → their RIGHT
#  v < 0 → UP          |  v > 0 → DOWN
# ═══════════════════════════════════════════════════════════════════════════════
def iris_offsets(lm):
    mirrored = lm[33].x < lm[263].x

    def _h(o, i, ic):
        lo, hi = min(lm[o].x, lm[i].x), max(lm[o].x, lm[i].x)
        ew = hi - lo
        return (lm[ic].x - (lo+hi)/2) / ew if ew > 0.005 else 0.0

    def _v(t, b, ic):
        eh = lm[b].y - lm[t].y
        return (lm[ic].y - (lm[t].y+lm[b].y)/2) / eh if eh > 0.005 else 0.0

    h = (_h(LE_O, LE_I, LI_C) + _h(RE_I, RE_O, RI_C)) / 2.0
    v = (_v(LE_T, LE_B, LI_C) + _v(RE_T, RE_B, RI_C)) / 2.0
    if mirrored:
        h = -h
    return h, v


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED GAZE
# ═══════════════════════════════════════════════════════════════════════════════
def gaze_direction(yaw, pitch, h, v):
    # ── Path 1: iris-only (eyes dart without head turning) ────────────────────
    # Checked first — if eyes alone move far enough, fire immediately
    if h < -IRIS_ONLY_H:  return "LEFT",  1.0, 1.0, 0.0, 0.0, 0.0
    if h >  IRIS_ONLY_H:  return "RIGHT", 1.0, 0.0, 1.0, 0.0, 0.0
    if v < -IRIS_ONLY_V:  return "UP",    1.0, 0.0, 0.0, 1.0, 0.0
    if v >  IRIS_ONLY_V:  return "DOWN",  1.0, 0.0, 0.0, 0.0, 1.0

    # ── Path 2: combined head + iris ──────────────────────────────────────────
    hL = min(max( yaw   / HEAD_YAW,    0.0), 1.0)
    hR = min(max(-yaw   / HEAD_YAW,    0.0), 1.0)
    hU = min(max(-pitch / HEAD_PITCH_U, 0.0), 1.0)
    hD = min(max( pitch / HEAD_PITCH_D, 0.0), 1.0)
    iL = min(max(-h / IRIS_H_THRESH,   0.0), 1.0)
    iR = min(max( h / IRIS_H_THRESH,   0.0), 1.0)
    iU = min(max(-v / IRIS_V_THRESH,   0.0), 1.0)
    iD = min(max( v / IRIS_V_THRESH,   0.0), 1.0)
    sL = hL*HEAD_W + iL*IRIS_W
    sR = hR*HEAD_W + iR*IRIS_W
    sU = hU*HEAD_W + iU*IRIS_W
    sD = hD*HEAD_W + iD*IRIS_W
    best = max(sL, sR, sU, sD)
    if best < FIRE_THRESH:
        return "AHEAD", best, sL, sR, sU, sD
    dir_ = ["LEFT","RIGHT","UP","DOWN"][[sL,sR,sU,sD].index(best)]
    return dir_, best, sL, sR, sU, sD


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW IRIS on scaled panel (uses normalised lm coords × PW/PH)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_iris(panel, lm, c, t, r, b, l, col):
    cx, cy = int(lm[c].x * PW), int(lm[c].y * PH)
    tx, ty = int(lm[t].x * PW), int(lm[t].y * PH)
    rx, ry = int(lm[r].x * PW), int(lm[r].y * PH)
    bx, by = int(lm[b].x * PW), int(lm[b].y * PH)
    lx, ly = int(lm[l].x * PW), int(lm[l].y * PH)
    # True iris radius = average of all 4 spoke distances from center
    rv = int(np.hypot(cx-tx, cy-ty))
    rh = int(np.hypot(cx-rx, cy-ry))
    rb = int(np.hypot(cx-bx, cy-by))
    rl = int(np.hypot(cx-lx, cy-ly))
    rad = max((rv + rh + rb + rl) // 4, 4)
    cv2.circle(panel, (cx,cy), rad, col, 1, cv2.LINE_AA)
    cv2.circle(panel, (cx,cy), 2,   col,-1, cv2.LINE_AA)
    return cx, cy


def blend(panel, x1, y1, x2, y2, color, alpha=0.60):
    roi = panel[y1:y2, x1:x2]
    ov  = np.full_like(roi, color)
    cv2.addWeighted(roi, 1-alpha, ov, alpha, 0, roi)
    panel[y1:y2, x1:x2] = roi


# ── Smoothing ──────────────────────────────────────────────────────────────────
yaw_q, pit_q, h_q, v_q = (deque(maxlen=SMOOTH_N) for _ in range(4))

# ── State ──────────────────────────────────────────────────────────────────────
phone_boxes  = []
t_phone_last = 0.0
frame_count  = 0
fps_t        = time.time()
fps_val      = 0.0

print("  LEFT = Student  |  RIGHT = Admin")
print("  Press Q or ESC to quit\n")

# ══════════════════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fh, fw = frame.shape[:2]
    frame_count += 1
    now = time.time()

    if frame_count % 20 == 0:
        fps_val = 20.0 / max(now - fps_t, 0.001)
        fps_t   = now

    # ── RT-DETR-x  (every YOLO_EVERY frames) ──────────────────────────────────
    if frame_count % YOLO_EVERY == 1:
        res = detector(frame, classes=[PHONE_CLS], verbose=False,
                       conf=PHONE_CONF, imgsz=IMGSZ)[0]
        if res.boxes and len(res.boxes):
            phone_boxes  = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                             int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                             float(b.conf)) for b in res.boxes]
            t_phone_last = now
        else:
            phone_boxes = []
    phone_active = (now - t_phone_last) < PHONE_LINGER

    # ── MediaPipe FaceMesh ─────────────────────────────────────────────────────
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_res   = face_mesh.process(rgb)
    faces      = mesh_res.multi_face_landmarks or []
    n_faces    = len(faces)
    extra_face = n_faces > 1

    # ── Scale panels (draw overlays on scaled versions to avoid tiny circles) ──
    s_panel = cv2.resize(frame, (PW, PH))   # student view — clean
    a_panel = s_panel.copy()                 # admin view  — will get overlays

    gaze_dir   = "NO_FACE"
    gaze_score = 0.0
    yaw = pitch = h_off = v_off = 0.0
    sL = sR = sU = sD = 0.0

    for fi, face_lms in enumerate(faces):
        lm      = face_lms.landmark
        primary = (fi == 0)

        # ── Bounding box ONLY on intruders, never on candidate ────────────────
        if not primary:
            xs = [int(lm[i].x * PW) for i in range(min(468, len(lm)))]
            ys = [int(lm[i].y * PH) for i in range(min(468, len(lm)))]
            if xs and ys:
                bx1 = max(min(xs)-8, 0);    by1 = max(min(ys)-8, 0)
                bx2 = min(max(xs)+8, PW-1); by2 = min(max(ys)+8, PH-1)
                cv2.rectangle(a_panel, (bx1,by1), (bx2,by2), RED, 2)
                cv2.putText(a_panel, f"! INTRUDER {fi}",
                            (bx1, max(by1-8, 14)),
                            FONT, 0.55, RED, 2, cv2.LINE_AA)

        if primary:
            # ── Smoothed head + iris ───────────────────────────────────────────
            yr, pr = head_pose(lm)
            hr, vr = iris_offsets(lm)
            yaw_q.append(yr); pit_q.append(pr)
            h_q.append(hr);   v_q.append(vr)
            yaw   = float(np.median(yaw_q))
            pitch = float(np.median(pit_q))
            h_off = float(np.median(h_q))
            v_off = float(np.median(v_q))

            gaze_dir, gaze_score, sL, sR, sU, sD = gaze_direction(
                yaw, pitch, h_off, v_off)

            # Iris: GREEN = focused, RED = looking away
            iris_col = GREEN if gaze_dir == "AHEAD" else RED
            draw_iris(a_panel, lm, LI_C, LI_T, LI_R, LI_B, LI_L, iris_col)
            draw_iris(a_panel, lm, RI_C, RI_T, RI_R, RI_B, RI_L, iris_col)

            # Gaze text — centered top of panel, red, no arrows
            if gaze_dir != "AHEAD":
                txt = {"LEFT":"LOOKING LEFT","RIGHT":"LOOKING RIGHT",
                       "UP":"LOOKING UP","DOWN":"LOOKING DOWN"}[gaze_dir]
                (tw,_),_ = cv2.getTextSize(txt, FONT, 0.90, 2)
                cv2.putText(a_panel, txt, ((PW-tw)//2, 38),
                            FONT, 0.90, RED, 2, cv2.LINE_AA)

        else:
            # ── Secondary face: red iris ───────────────────────────────────────
            draw_iris(a_panel, lm, LI_C, LI_T, LI_R, LI_B, LI_L, RED)
            draw_iris(a_panel, lm, RI_C, RI_T, RI_R, RI_B, RI_L, RED)

    # ── Phone boxes on admin panel ─────────────────────────────────────────────
    sx, sy = PW/fw, PH/fh
    for x1,y1,x2,y2,conf in phone_boxes:
        px1,py1 = int(x1*sx), int(y1*sy)
        px2,py2 = int(x2*sx), int(y2*sy)
        cv2.rectangle(a_panel,(px1,py1),(px2,py2),RED,3)
        lbl = f" PHONE {conf:.0%} "
        (tw,th),_ = cv2.getTextSize(lbl, FONT, 0.62, 1)
        cv2.rectangle(a_panel,(px1,py1-th-10),(px1+tw+4,py1),RED,-1)
        cv2.putText(a_panel,lbl,(px1+2,py1-5),FONT,0.62,WHITE,1,cv2.LINE_AA)

    # ── Alert banners ──────────────────────────────────────────────────────────
    banner_y = PH  # stack banners upward from bottom
    if extra_face:
        banner_y -= 44
        y0 = max(banner_y - 4, 0)
        blend(a_panel, 0, y0, PW, min(banner_y+32, PH), (50,0,110), 0.75)
        msg = f"!  {n_faces} FACES DETECTED  —  CHEATING RISK  !"
        (mw,_),_ = cv2.getTextSize(msg, FONT, 0.60, 2)
        cv2.putText(a_panel, msg, ((PW-mw)//2, banner_y+22),
                    FONT, 0.60, YELLOW, 2, cv2.LINE_AA)

    if phone_active:
        banner_y -= 44
        y0 = max(banner_y - 4, 0)
        blend(a_panel, 0, y0, PW, min(banner_y+32, PH), (0,0,150), 0.75)
        banner = "!!!  PHONE DETECTED  !!!"
        (bw,_),_ = cv2.getTextSize(banner, FONT, 0.80, 2)
        cv2.putText(a_panel, banner, ((PW-bw)//2, banner_y+22),
                    FONT, 0.80, WHITE, 2, cv2.LINE_AA)

    # ── Label strip ───────────────────────────────────────────────────────────
    LABEL_H = 30
    labels  = np.full((LABEL_H, TW, 3), DGRAY, np.uint8)
    cv2.line(labels,(0,LABEL_H-1),(TW,LABEL_H-1),MGRAY,1)
    cv2.putText(labels,"STUDENT VIEW",(PW//2-68,21),FONT,0.55,LGRAY,1)
    cv2.line(labels,(PW,3),(PW,LABEL_H-3),MGRAY,3)
    cv2.putText(labels,"ADMIN PANEL",(PW+PW//2-62,21),FONT,0.55,YELLOW,1)

    panels = np.hstack([s_panel, a_panel])
    cv2.line(panels,(PW,0),(PW,PH),MGRAY,2)

    # ── Status bar ─────────────────────────────────────────────────────────────
    BAR_H = 86
    bar   = np.full((BAR_H, TW, 3), DGRAY, np.uint8)
    cv2.line(bar,(0,0),(TW,0),MGRAY,2)
    cv2.line(bar,(PW,0),(PW,BAR_H),MGRAY,2)

    # Left — student label
    cv2.putText(bar,"Student feed — no overlays",(12,50),FONTS,0.42,MGRAY,1)

    # Right — gaze + phone
    A = PW + 14
    gmap = {
        "AHEAD"  :("LOOKING AHEAD",  GREEN),
        "LEFT"   :("LOOKING LEFT",   RED),
        "RIGHT"  :("LOOKING RIGHT",  RED),
        "UP"     :("LOOKING UP",     RED),
        "DOWN"   :("LOOKING DOWN",   RED),
        "NO_FACE":("NO FACE",        YELLOW),
    }
    gtxt, gcol = gmap.get(gaze_dir, ("NO FACE", YELLOW))
    cv2.circle(bar,(A+11,22),9,gcol,-1)
    cv2.putText(bar,gtxt,(A+27,28),FONTS,0.56,gcol,1)

    pcol = RED if phone_active else GREEN
    cv2.circle(bar,(A+11,48),9,pcol,-1)
    cv2.putText(bar,("PHONE DETECTED!" if phone_active else "NO PHONE"),
                (A+27,54),FONTS,0.56,pcol,1)

    # Debug values
    dbg=(f"Yaw:{yaw:+5.1f}  Pitch:{pitch:+5.1f}  "
         f"iH:{h_off:+.3f}  iV:{v_off:+.3f}  "
         f"L:{sL:.2f} R:{sR:.2f} U:{sU:.2f} D:{sD:.2f}  "
         f"score:{gaze_score:.2f}/{FIRE_THRESH}")
    cv2.putText(bar,dbg,(A,73),FONTS,0.31,LGRAY,1)

    # Faces + FPS (right side)
    fcol = RED if extra_face else GREEN
    ftxt = f"FACES: {n_faces}" + (f"  — {n_faces-1} INTRUDER(S)!" if extra_face else "  — OK")
    cv2.putText(bar, ftxt,
                (TW-210,28),FONTS,0.54,fcol,1)
    cv2.putText(bar,f"RT-DETR-x  conf={PHONE_CONF}  |  {fps_val:.0f} FPS",
                (TW-210,50),FONTS,0.36,LGRAY,1)

    cv2.imshow("Proctor Lite  v5", np.vstack([labels, panels, bar]))
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("  Stopped.")
