# Dream Cheeky / Thunder USB Turret
# Modes: Manual WASD | Hand Tracking | Face Tracking (with RECOGNITION)
# Windows | PyUSB | OpenCV (DirectShow) | MediaPipe Tasks API (0.10+) | TensorFlow
#
# Requirements:
#   pip install mediapipe opencv-python pyusb tensorflow scikit-learn pillow
#
# SETUP:
#   1. Put face images in:  C:\Users\CSstudent\turretStuff\people\
#      Named like:  danielc1.png  dylanf1.png  dylanf2.png  etc.
#   2. Run:  python train_faces.py        (creates face_embeddings.pkl)
#   3. Run:  python turret.py             (this file)
#
# Controls:
#   .          - Cycle modes: MANUAL → HAND TRACK → FACE TRACK → MANUAL
#   WASD       - Move (manual mode only)
#   SPACE      - Fire (always works, cycles bang → boom → kaboom)
#   P          - Capture face photo and save to people\ folder for training
#   Q / ESC    - Quit
#   q (window) - Quit
#
# Recognition overlay (MANUAL + FACE mode):
#   - Green box  = tracked target (largest face), name shown if recognised
#   - Red box    = other detected faces
#   - "Unknown"  = face detected but confidence below RECOG_THRESHOLD


import time
import ctypes
import threading
import os
import urllib.request
import pickle
import winsound
import re

import usb.core
import usb.util
import usb.backend.libusb1
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# TensorFlow / embedder (optional — gracefully disabled if model not trained yet)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

cv2.setNumThreads(1)
cv2.setUseOptimized(True)


# ==============================
# MODEL AUTO-DOWNLOAD
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL_PATH = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

for path, url, label in [
    (FACE_MODEL_PATH, FACE_MODEL_URL, "face detector (~1 MB)"),
    (HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker (~9 MB)"),
]:
    if not os.path.isfile(path):
        print(f"Downloading {label} to:\n  {path}")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


# ==============================
# FACE RECOGNITION SETUP
# ==============================

RECOG_PKL_PATH      = r"C:\Users\CSstudent\turretStuff\face_embeddings.pkl"
RECOG_THRESHOLD     = 0.75   # minimum SVM probability to show a name (vs "Unknown")
RECOG_CROP_PADDING  = 0.25   # fractional padding around face bounding box
RECOG_IMG_SIZE      = (96, 96)
RECOG_EVERY_N       = 4      # run recognition every N frames (saves CPU)

# Folder where captured training photos are saved (P key)
PEOPLE_FOLDER       = r"C:\Users\CSstudent\turretStuff\people"

_recog_model        = None   # sklearn Pipeline
_recog_label_enc    = None   # LabelEncoder
_recog_embedder     = None   # Keras model


def _load_recognition_model():
    """Load trained classifier + build TF embedder.  Call once at startup."""
    global _recog_model, _recog_label_enc, _recog_embedder

    if not TF_AVAILABLE:
        print("[RECOG] TensorFlow not installed — recognition disabled.")
        return False

    if not os.path.isfile(RECOG_PKL_PATH):
        print("[RECOG] face_embeddings.pkl not found — run train_faces.py first.")
        print("[RECOG] Recognition disabled; turret will track but not identify.")
        return False

    with open(RECOG_PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    _recog_model     = payload["classifier"]
    _recog_label_enc = payload["label_encoder"]

    print("[RECOG] Building MobileNetV2 embedder …")
    base = MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(RECOG_IMG_SIZE[0], RECOG_IMG_SIZE[1], 3),
        pooling="avg",
    )
    base.trainable = False
    _recog_embedder = base
    # Warm-up call so the first real frame isn't slow
    _recog_embedder.predict(
        np.zeros((1, RECOG_IMG_SIZE[0], RECOG_IMG_SIZE[1], 3), dtype=np.float32),
        verbose=0,
    )
    print(f"[RECOG] Ready. Classes: {list(_recog_label_enc.classes_)}")
    return True


RECOGNITION_ENABLED = _load_recognition_model()


def _embed_crop(crop_bgr):
    """BGR face crop → 1280-d embedding vector."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, RECOG_IMG_SIZE)
    x       = preprocess_input(resized.astype(np.float32))
    x       = np.expand_dims(x, 0)
    return _recog_embedder.predict(x, verbose=0)[0]


def recognise_face(crop_bgr):
    """
    Returns (name_str, confidence_float) for the face crop.
    Returns ("Unknown", 0.0) if below threshold or model not loaded.
    """
    if not RECOGNITION_ENABLED:
        return ("Unknown", 0.0)
    try:
        vec   = _embed_crop(crop_bgr)
        probs = _recog_model.predict_proba([vec])[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        if conf < RECOG_THRESHOLD:
            return ("Unknown", conf)
        name = _recog_label_enc.inverse_transform([idx])[0]
        return (name, conf)
    except Exception as e:
        print(f"[RECOG] Error during recognition: {e}")
        return ("Unknown", 0.0)


def extract_face_crop(img_bgr, detection, padding=RECOG_CROP_PADDING):
    """Return padded BGR crop for a MediaPipe detection, or None."""
    h, w = img_bgr.shape[:2]
    bb   = detection.bounding_box
    x1   = int(bb.origin_x);          y1 = int(bb.origin_y)
    x2   = int(bb.origin_x + bb.width)
    y2   = int(bb.origin_y + bb.height)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    x1 = max(0, x1 - px);             y1 = max(0, y1 - py)
    x2 = min(w, x2 + px);             y2 = min(h, y2 + py)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return img_bgr[y1:y2, x1:x2]


# ==============================
# PHOTO CAPTURE  (P key)
# ==============================

def _next_capture_index(folder, label):
    """
    Scan existing files like  <label><N>.png  and return the next N.
    e.g. danielc1.png, danielc2.png → returns 3
    """
    pattern = re.compile(r'^' + re.escape(label) + r'(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
    max_idx = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def capture_face_photo(frame_bgr, detections, people_folder):
    """
    Save the largest detected face crop from frame_bgr into people_folder.
    The user is prompted for a label name via console input (non-blocking thread).
    Returns a status string for the HUD.
    """
    if not detections:
        print("[CAPTURE] No face detected — move closer to the camera.")
        return "CAPTURE FAILED: no face"

    # Pick largest detection
    best = max(detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    crop = extract_face_crop(frame_bgr, best, padding=0.30)
    if crop is None:
        print("[CAPTURE] Face crop too small.")
        return "CAPTURE FAILED: crop too small"

    def _ask_and_save(crop_copy):
        label = input("[CAPTURE] Enter label for this face (e.g. danielc): ").strip().lower()
        if not label:
            print("[CAPTURE] Cancelled — no label entered.")
            return
        os.makedirs(people_folder, exist_ok=True)
        idx      = _next_capture_index(people_folder, label)
        filename = f"{label}{idx}.png"
        filepath = os.path.join(people_folder, filename)
        cv2.imwrite(filepath, crop_copy)
        print(f"[CAPTURE] Saved → {filepath}  (re-run train_faces.py to retrain)")

    # Run in thread so the camera loop isn't frozen while waiting for console input
    threading.Thread(target=_ask_and_save, args=(crop.copy(),), daemon=True).start()
    return "CAPTURE: enter label in console"


# ==============================
# AUDIO SETUP
# ==============================

SFX_DIR    = os.path.join(BASE_DIR, "sfx")
BANG_WAV   = os.path.join(SFX_DIR, "bang.wav")
BOOM_WAV   = os.path.join(SFX_DIR, "boom.wav")
KABOOM_WAV = os.path.join(SFX_DIR, "kaboom.wav")
MOVE_WAV   = os.path.join(SFX_DIR, "move.wav")

shoot_cycle = [BANG_WAV, BOOM_WAV, KABOOM_WAV]
shoot_idx   = 0
move_looping = False


def _check_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing sound file: {path}")


for p in [BANG_WAV, BOOM_WAV, KABOOM_WAV, MOVE_WAV]:
    _check_file(p)


def play_shoot_cycled():
    global shoot_idx
    path = shoot_cycle[shoot_idx]
    shoot_idx = (shoot_idx + 1) % len(shoot_cycle)
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


def set_moving_sound(is_moving: bool):
    global move_looping
    if is_moving and not move_looping:
        winsound.PlaySound(
            MOVE_WAV,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
        )
        move_looping = True
    elif (not is_moving) and move_looping:
        winsound.PlaySound(None, winsound.SND_ASYNC)
        move_looping = False


# ==============================
# USB / TURRET CONFIG
# ==============================

VID = 0x2123
PID = 0x1010

LIBUSB_DLL_PATH = r"C:\Users\CSstudent\turretStuff\libusb-1.0.dll"

backend = usb.backend.libusb1.get_backend(find_library=lambda _: LIBUSB_DLL_PATH)
if backend is None:
    raise RuntimeError("libusb backend failed to load.")

dev = usb.core.find(idVendor=VID, idProduct=PID, backend=backend)
if dev is None:
    raise ValueError("Turret not found.")

dev.set_configuration()
cfg = dev.get_active_configuration()


def pick_iface():
    for candidate in [(0, 0), (1, 0)]:
        try:
            return cfg[candidate].bInterfaceNumber
        except Exception:
            pass
    raise RuntimeError("Interface not found.")


iface = pick_iface()

try:
    usb.util.claim_interface(dev, iface)
except usb.core.USBError:
    pass


def send(cmd):
    cmd = cmd[:8] if len(cmd) >= 8 else cmd + [0] * (8 - len(cmd))
    dev.ctrl_transfer(0x21, 0x09, 0x0200, iface, cmd)


CMD_STOP  = [0x02, 0x20, 0, 0, 0, 0, 0, 0]
CMD_UP    = [0x02, 0x02, 0, 0, 0, 0, 0, 0]
CMD_DOWN  = [0x02, 0x01, 0, 0, 0, 0, 0, 0]
CMD_LEFT  = [0x02, 0x04, 0, 0, 0, 0, 0, 0]
CMD_RIGHT = [0x02, 0x08, 0, 0, 0, 0, 0, 0]
CMD_FIRE  = [0x02, 0x10, 0, 0, 0, 0, 0, 0]


def stop():
    send(CMD_STOP)


def fire(hold_seconds=1.15):
    send(CMD_FIRE)
    time.sleep(hold_seconds)
    stop()


def fire_async(hold_seconds=1.15):
    threading.Thread(target=fire, args=(hold_seconds,), daemon=True).start()


# ==============================
# KEYBOARD (Windows)
# ==============================

user32 = ctypes.windll.user32

VK_W     = 0x57
VK_A     = 0x41
VK_S     = 0x53
VK_D     = 0x44
VK_Period     = 0xBE
VK_P     = 0x50
VK_SPACE = 0x20
VK_Q     = 0x51
VK_ESC   = 0x1B


def is_down(vk: int) -> bool:
    return (user32.GetAsyncKeyState(vk) & 0x8000) != 0


# ==============================
# TUNING
# ==============================

POLL_DELAY    = 0.01
FIRE_COOLDOWN = 0.35

CAM_INDEX         = 2
CAM_W             = 640
CAM_H             = 480
CAM_FPS           = 30
CAM_WARMUP_FRAMES = 10
CAM_PROBE_SECONDS = 2.0
CAM_GRAB_FLUSH    = 2

PULSE_MODE           = True
PULSE_ON             = 0.03
PULSE_OFF            = 0.02
DIAG_PULSE_UPDOWN    = 0.018
DIAG_PULSE_LEFTRIGHT = 0.018
DIAG_OFF             = 0.012

AIM_OFFSET_X = -40
AIM_OFFSET_Y = -40

DEAD_ZONE       = 40
SLOW_ZONE       = 90
TRACK_PULSE_MIN = 0.010
TRACK_PULSE_MAX = 0.060
SLOW_PULSE_MAX  = 0.018
TRACK_PAUSE     = 0.015

OSCILL_FLIP_THRESHOLD = 4
OSCILL_HISTORY        = 6
OSCILL_COOLDOWN       = 0.25

HALF_W = CAM_W // 2
HALF_H = CAM_H // 2

# ==============================
# FIST-TO-SHOOT TUNING
# ==============================

FIST_HOLD_SECONDS   = 0.4
FIST_FIRE_COOLDOWN  = 2.0
FIST_CURL_THRESHOLD = 0.05

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

FINGER_TIP_PIP = [
    (8,  6),
    (12, 10),
    (16, 14),
    (20, 18),
]


# ==============================
# MODE
# ==============================

MODE_MANUAL = "MANUAL"
MODE_HAND   = "HAND TRACK"
MODE_FACE   = "FACE TRACK"
MODES       = [MODE_MANUAL, MODE_HAND, MODE_FACE]

mode_index   = 0
current_mode = MODES[mode_index]


# ==============================
# SHARED STATE
# ==============================

running           = True
mouse_x, mouse_y  = 0, 0
target_cx         = None
target_cy         = None

fist_fire_requested  = False
fist_fire_lock       = threading.Lock()

# P-key capture: camera thread writes latest detections here for main loop to read
_latest_frame      = None
_latest_detections = []
_frame_lock        = threading.Lock()

capture_hud_text       = ""
capture_hud_until      = 0.0
p_was_down             = False


# ==============================
# FIST DETECTION
# ==============================

def is_fist(landmarks) -> bool:
    wrist   = landmarks[0]
    mid_mcp = landmarks[9]
    hand_h  = abs(mid_mcp.y - wrist.y)
    if hand_h < 1e-4:
        hand_h = 0.1
    for tip_idx, pip_idx in FINGER_TIP_PIP:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        if tip.y < pip.y + FIST_CURL_THRESHOLD * hand_h:
            return False
    return True


# ==============================
# OSCILLATION TRACKER
# ==============================

class OscillationTracker:
    def __init__(self):
        self._history        = []
        self._cooldown_until = 0.0

    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def record(self, sign: int):
        self._history.append(sign)
        if len(self._history) > OSCILL_HISTORY:
            self._history.pop(0)
        flips = sum(
            1 for i in range(1, len(self._history))
            if self._history[i] != self._history[i - 1]
        )
        if flips >= OSCILL_FLIP_THRESHOLD:
            self._cooldown_until = time.time() + OSCILL_COOLDOWN
            self._history.clear()

    def reset(self):
        self._history.clear()
        self._cooldown_until = 0.0


osc_x = OscillationTracker()
osc_y = OscillationTracker()


# ==============================
# TRACKING PULSE HELPER
# ==============================

def calc_pulse(error_px: int, half_range: int) -> float:
    abs_err = abs(error_px)
    if abs_err <= SLOW_ZONE:
        ratio = min((abs_err - DEAD_ZONE) / max(SLOW_ZONE - DEAD_ZONE, 1), 1.0)
        return TRACK_PULSE_MIN + ratio * (SLOW_PULSE_MAX - TRACK_PULSE_MIN)
    ratio = min(abs_err / max(half_range - DEAD_ZONE, 1), 1.0)
    return TRACK_PULSE_MIN + ratio * (TRACK_PULSE_MAX - TRACK_PULSE_MIN)


# ==============================
# MEDIAPIPE DETECTORS
# ==============================

def build_face_detector():
    base_opts = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    opts = mp_vision.FaceDetectorOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(opts)


def build_hand_detector():
    base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detection_to_box(det, frame_w, frame_h):
    bb = det.bounding_box
    x1 = max(0, int(bb.origin_x))
    y1 = max(0, int(bb.origin_y))
    x2 = min(frame_w, int(bb.origin_x + bb.width))
    y2 = min(frame_h, int(bb.origin_y + bb.height))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    area = (x2 - x1) * (y2 - y1)
    return x1, y1, x2, y2, cx, cy, area


def get_hand_center(landmarks, frame_w, frame_h):
    idxs = (0, 5, 9, 13, 17)
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return int(np.mean(xs) * frame_w), int(np.mean(ys) * frame_h)


# ==============================
# CAMERA THREAD
# ==============================

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


def camera_loop():
    global running, mouse_x, mouse_y, target_cx, target_cy, current_mode
    global fist_fire_requested, _latest_frame, _latest_detections

    fist_start_time      = None
    fist_fired_this_hold = False
    last_fist_fire_time  = 0.0

    # Recognition cache — updated every RECOG_EVERY_N frames
    recog_frame_counter = 0
    recog_cache = {}   # face index → (name, confidence)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera failed to open.")
        running = False
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

    for _ in range(CAM_WARMUP_FRAMES):
        cap.read()

    t0, got_frame = time.time(), False
    while time.time() - t0 < CAM_PROBE_SECONDS:
        ret, _ = cap.read()
        if ret:
            got_frame = True
            break

    if not got_frame:
        print("Camera opened but no frames received.")
        running = False
        cap.release()
        return

    window_name = "Turret POV (Mirrored)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    face_detector = build_face_detector()
    hand_detector = build_hand_detector()

    while running:
        for _ in range(CAM_GRAB_FLUSH):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        cx_frame = (w // 2) + AIM_OFFSET_X
        cy_frame = (h // 2) + AIM_OFFSET_Y

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        mode_now    = current_mode
        detected_cx = None
        detected_cy = None
        n_targets   = 0

        fist_hud_text  = ""
        fist_hud_color = (0, 165, 255)

        recog_frame_counter += 1
        do_recog = (recog_frame_counter % RECOG_EVERY_N == 0)

        # ------------------------------------------------------------------
        # FACE DETECTION + RECOGNITION  (MANUAL and FACE TRACK modes only)
        # ------------------------------------------------------------------
        if mode_now in (MODE_MANUAL, MODE_FACE):
            result = face_detector.detect(mp_image)

            # Always share latest detections for the P-key capture
            with _frame_lock:
                _latest_frame      = frame.copy()
                _latest_detections = result.detections if result.detections else []

            if result.detections:
                boxes = [detection_to_box(d, w, h) for d in result.detections]
                boxes.sort(key=lambda b: b[6], reverse=True)
                n_targets = len(boxes)

                # Run recognition every N frames
                if do_recog and RECOGNITION_ENABLED:
                    new_cache = {}
                    for idx, det in enumerate(result.detections):
                        crop = extract_face_crop(frame, det)
                        if crop is not None:
                            name, conf = recognise_face(crop)
                            new_cache[idx] = (name, conf)
                    recog_cache = new_cache

                # Draw non-tracked faces (red)
                for i, (x1, y1, x2, y2, cx, cy, area) in enumerate(boxes[1:], start=1):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 220), -1)
                    cached = recog_cache.get(i)
                    label_str = cached[0] if cached else "face"
                    cv2.putText(frame, label_str, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

                # Draw tracked face (green, index 0 = largest)
                x1, y1, x2, y2, cx, cy, _ = boxes[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(frame, (cx, cy), 8, (0, 220, 0), -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
                cv2.line(frame, (cx_frame, cy_frame), (cx, cy),
                         (0, 220, 0), 1, cv2.LINE_AA)

                # Name label for tracked face
                tracked_info = recog_cache.get(0)
                if tracked_info:
                    name, conf = tracked_info
                    name_label = f"{name}  {conf:.0%}" if name != "Unknown" else "Unknown"
                    name_color = (0, 255, 80) if name != "Unknown" else (60, 60, 255)
                    (lw, lh), _ = cv2.getTextSize(name_label,
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 6, y1 - 2),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, name_label, (x1 + 3, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, name_color, 2)
                else:
                    cv2.putText(frame, "TARGET", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

                # Only move turret in FACE TRACK mode
                if mode_now == MODE_FACE:
                    detected_cx, detected_cy = cx, cy

        # ------------------------------------------------------------------
        # HAND TRACKING  (+ fist-to-shoot)  — HAND TRACK mode only
        # ------------------------------------------------------------------
        elif mode_now == MODE_HAND:
            result = hand_detector.detect(mp_image)
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                hx, hy = get_hand_center(landmarks, w, h)
                n_targets = 1

                # Draw skeleton
                for a, b in HAND_CONNECTIONS:
                    ax = int(landmarks[a].x * w); ay = int(landmarks[a].y * h)
                    bx = int(landmarks[b].x * w); by = int(landmarks[b].y * h)
                    cv2.line(frame, (ax, ay), (bx, by), (80, 200, 80), 1, cv2.LINE_AA)
                # Draw landmark dots
                for lm in landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

                cv2.circle(frame, (hx, hy), 8, (0, 165, 255), -1)
                cv2.circle(frame, (hx, hy), 8, (255, 255, 255), 2)
                cv2.line(frame, (cx_frame, cy_frame), (hx, hy),
                         (0, 165, 255), 1, cv2.LINE_AA)

                detected_cx, detected_cy = hx, hy

                now = time.time()
                fist_detected     = is_fist(landmarks)
                post_fire_cooling = (now - last_fist_fire_time) < FIST_FIRE_COOLDOWN

                if fist_detected and not post_fire_cooling:
                    if fist_start_time is None:
                        fist_start_time      = now
                        fist_fired_this_hold = False

                    hold_duration = now - fist_start_time

                    if not fist_fired_this_hold:
                        bar_w  = 120; bar_h = 14
                        bar_x  = hx - bar_w // 2; bar_y = hy - 40
                        ratio  = min(hold_duration / FIST_HOLD_SECONDS, 1.0)
                        fill_w = int(bar_w * ratio)

                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                        bar_color = (0, 200, 255) if ratio < 1.0 else (0, 0, 255)
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                        cv2.putText(frame, "HOLD FIST TO FIRE",
                                    (bar_x, bar_y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                        if hold_duration >= FIST_HOLD_SECONDS:
                            fist_fired_this_hold = True
                            last_fist_fire_time  = now
                            with fist_fire_lock:
                                fist_fire_requested = True
                    else:
                        remaining = FIST_FIRE_COOLDOWN - (now - last_fist_fire_time)
                        fist_hud_text  = f"FIST FIRE! cooldown {remaining:.1f}s"
                        fist_hud_color = (0, 0, 255)
                        cv2.putText(frame, "FIRED!", (hx - 30, hy - 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif post_fire_cooling:
                    remaining = FIST_FIRE_COOLDOWN - (now - last_fist_fire_time)
                    fist_hud_text  = f"cooldown {remaining:.1f}s"
                    fist_hud_color = (100, 100, 255)
                    fist_start_time      = None
                    fist_fired_this_hold = False
                else:
                    fist_start_time      = None
                    fist_fired_this_hold = False
            else:
                fist_start_time      = None
                fist_fired_this_hold = False

        target_cx = detected_cx
        target_cy = detected_cy

        # --------------------------------------------------
        # HUD
        # --------------------------------------------------
        cv2.line(frame, (cx_frame - 20, cy_frame), (cx_frame + 20, cy_frame), (0, 0, 255), 1)
        cv2.line(frame, (cx_frame, cy_frame - 20), (cx_frame, cy_frame + 20), (0, 0, 255), 1)
        cv2.circle(frame, (cx_frame, cy_frame), 20, (0, 0, 255), 1)

        cv2.line(frame, (mouse_x - 12, mouse_y), (mouse_x + 12, mouse_y), (0, 255, 0), 1)
        cv2.line(frame, (mouse_x, mouse_y - 12), (mouse_x, mouse_y + 12), (0, 255, 0), 1)

        if mode_now == MODE_MANUAL:
            overlay_clr = (0, 255, 100)
        elif mode_now == MODE_HAND:
            overlay_clr = (0, 165, 255)
        else:
            overlay_clr = (255, 220, 100)

        tgt_label = (f"TARGET: ({detected_cx}, {detected_cy})  |  count: {n_targets}"
                     if detected_cx else f"TARGET: none  |  count: {n_targets}")

        recog_status = ("ON" if RECOGNITION_ENABLED else "OFF (run train_faces.py)")
        cv2.putText(frame, f"MODE: {mode_now}  [. cycle]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, overlay_clr, 2)
        cv2.putText(frame, tgt_label,
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, overlay_clr, 1)
        cv2.putText(frame, f"MOUSE: ({mouse_x}, {mouse_y})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"RECOG: {recog_status}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 220, 80) if RECOGNITION_ENABLED else (60, 60, 200), 1)

        if mode_now == MODE_HAND and fist_hud_text:
            cv2.putText(frame, f"FIST: {fist_hud_text}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fist_hud_color, 1)
        elif mode_now == MODE_HAND:
            cv2.putText(frame, "FIST: make a fist to fire",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)

        # Capture HUD message (shown for a few seconds after P press)
        if capture_hud_text and time.time() < capture_hud_until:
            cv2.putText(frame, capture_hud_text,
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(frame, "[P] Capture face for training",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        if mode_now != MODE_MANUAL:
            cv2.circle(frame, (cx_frame, cy_frame), DEAD_ZONE, (60, 60, 60), 1)
            cv2.circle(frame, (cx_frame, cy_frame), SLOW_ZONE, (40, 40, 40), 1)
            if detected_cx is None and mode_now == MODE_FACE:
                cv2.putText(frame, "! NO TARGET — TURRET STOPPED",
                            (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    face_detector.close()
    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()


camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()


# ==============================
# MAIN CONTROL LOOP
# ==============================

print("=" * 58)
print("  Turret | Multi-Mode Tracking + Face Recognition")
print("  .       = Cycle: MANUAL / HAND TRACK / FACE TRACK")
print("  WASD    = Move (manual mode only)")
print("  SPACE   = Fire (always)")
print("  P       = Capture face photo → people\\ folder")
print("  FIST    = Hold fist for", FIST_HOLD_SECONDS, "s to fire (HAND TRACK mode)")
print("  Q / ESC = Quit")
if RECOGNITION_ENABLED:
    print("  Recognition: ENABLED  (MANUAL + FACE TRACK modes)")
else:
    print("  Recognition: DISABLED — run train_faces.py to enable")
print("=" * 58)

current_move_cmd = None
space_was_down   = False
period_was_down  = False
p_was_down       = False
last_fire_time   = 0.0

try:
    while running:
        if is_down(VK_Q) or is_down(VK_ESC):
            break

        now = time.time()

        # ── .: cycle mode ──────────────────────────────────────────
        period_down = is_down(VK_Period)
        if period_down and not period_was_down:
            mode_index   = (MODES.index(current_mode) + 1) % len(MODES)
            current_mode = MODES[mode_index]
            stop()
            current_move_cmd = None
            set_moving_sound(False)
            osc_x.reset(); osc_y.reset()
            print(f"Mode → {current_mode}")
        period_was_down = period_down

        # ── SPACE: fire ────────────────────────────────────────────
        space_down = is_down(VK_SPACE)
        if space_down and not space_was_down and (now - last_fire_time) > FIRE_COOLDOWN:
            last_fire_time = now
            set_moving_sound(False)
            stop()
            current_move_cmd = None
            play_shoot_cycled()
            fire_async()
        space_was_down = space_down

        # ── P: capture face photo ──────────────────────────────────
        p_down = is_down(VK_P)
        if p_down and not p_was_down:
            with _frame_lock:
                frame_snap = _latest_frame.copy() if _latest_frame is not None else None
                dets_snap  = list(_latest_detections)
            if frame_snap is not None:
                msg = capture_face_photo(frame_snap, dets_snap, PEOPLE_FOLDER)
            else:
                msg = "CAPTURE: no frame yet"
            capture_hud_text  = msg
            capture_hud_until = now + 4.0
            print(f"[P] {msg}")
        p_was_down = p_down

        # ── Fist fire ──────────────────────────────────────────────
        with fist_fire_lock:
            fist_requested = fist_fire_requested
            if fist_requested:
                fist_fire_requested = False

        if fist_requested and (now - last_fire_time) > FIRE_COOLDOWN:
            last_fire_time = now
            set_moving_sound(False)
            stop()
            current_move_cmd = None
            print("FIST FIRE!")
            play_shoot_cycled()
            fire_async()

        # ── Tracking movement ──────────────────────────────────────
        if current_mode in (MODE_HAND, MODE_FACE):
            tx, ty = target_cx, target_cy

            if tx is None:
                if current_move_cmd is not None:
                    stop(); current_move_cmd = None
                set_moving_sound(False)
                time.sleep(POLL_DELAY)
                continue

            err_x = tx - (HALF_W + AIM_OFFSET_X)
            err_y = ty - (HALF_H + AIM_OFFSET_Y)

            want_h = abs(err_x) > DEAD_ZONE
            want_v = abs(err_y) > DEAD_ZONE

            if not want_h and not want_v:
                if current_move_cmd is not None:
                    stop(); current_move_cmd = None
                set_moving_sound(False)
                time.sleep(POLL_DELAY)
                continue

            h_sign = +1 if err_x > 0 else -1
            v_sign = +1 if err_y > 0 else -1
            h_cmd  = CMD_LEFT if err_x > 0 else CMD_RIGHT
            v_cmd  = CMD_DOWN if err_y > 0 else CMD_UP

            h_blocked = want_h and osc_x.in_cooldown()
            v_blocked = want_v and osc_y.in_cooldown()

            if h_blocked and v_blocked:
                if current_move_cmd is not None:
                    stop(); current_move_cmd = None
                set_moving_sound(False)
                time.sleep(OSCILL_COOLDOWN * 0.5)
                continue

            set_moving_sound(True)

            if want_h and not h_blocked and want_v and not v_blocked:
                v_pulse = calc_pulse(err_y, HALF_H)
                h_pulse = calc_pulse(err_x, HALF_W)
                osc_y.record(v_sign); osc_x.record(h_sign)
                send(v_cmd);  current_move_cmd = "DIAG"
                time.sleep(v_pulse)
                stop()
                send(h_cmd)
                time.sleep(h_pulse)
                stop()
                current_move_cmd = None
                time.sleep(TRACK_PAUSE)
            elif want_h and not h_blocked:
                pulse = calc_pulse(err_x, HALF_W)
                osc_x.record(h_sign)
                send(h_cmd);  current_move_cmd = h_cmd
                time.sleep(pulse)
                stop()
                current_move_cmd = None
                time.sleep(TRACK_PAUSE)
            elif want_v and not v_blocked:
                pulse = calc_pulse(err_y, HALF_H)
                osc_y.record(v_sign)
                send(v_cmd);  current_move_cmd = v_cmd
                time.sleep(pulse)
                stop()
                current_move_cmd = None
                time.sleep(TRACK_PAUSE)
            continue

        # ── Manual WASD ────────────────────────────────────────────
        up    = is_down(VK_W)
        down  = is_down(VK_S)
        left  = is_down(VK_A)
        right = is_down(VK_D)

        set_moving_sound(up or down or left or right)

        vert_cmd  = CMD_UP   if up   else (CMD_DOWN  if down  else None)
        horiz_cmd = CMD_LEFT if left else (CMD_RIGHT if right else None)

        if vert_cmd is None and horiz_cmd is None:
            if current_move_cmd is not None:
                stop(); current_move_cmd = None
            time.sleep(POLL_DELAY)
            continue

        if vert_cmd is not None and horiz_cmd is not None:
            current_move_cmd = "DIAG"
            if PULSE_MODE:
                send(vert_cmd);  time.sleep(DIAG_PULSE_UPDOWN);    stop()
                send(horiz_cmd); time.sleep(DIAG_PULSE_LEFTRIGHT);  stop()
                time.sleep(DIAG_OFF)
            else:
                send(vert_cmd);  time.sleep(0.02)
                send(horiz_cmd); time.sleep(0.02)
            continue

        move_cmd = vert_cmd if vert_cmd else horiz_cmd
        if PULSE_MODE:
            send(move_cmd);  current_move_cmd = move_cmd
            time.sleep(PULSE_ON)
            stop();          current_move_cmd = None
            time.sleep(PULSE_OFF)
            continue

        if current_move_cmd != move_cmd:
            send(move_cmd)
            current_move_cmd = move_cmd
        time.sleep(POLL_DELAY)

except KeyboardInterrupt:
    pass

finally:
    running = False
    try:
        set_moving_sound(False)
    except Exception:
        pass
    try:
        stop()
    except Exception:
        pass
    try:
        usb.util.release_interface(dev, iface)
    except Exception:
        pass
    camera_thread.join(timeout=1.0)
    print("Clean exit.")
