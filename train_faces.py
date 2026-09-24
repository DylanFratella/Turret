"""
train_faces.py  —  Build a face-recognition model from images in DEFAULT_FOLDER.


Folder structure expected:
    C:\\Users\\CSstudent\\turretStuff\\people\\
        danielc1.png
        danielc2.png
        dylanf1.png
        dylanf2.png
        ...   (any <name><digits>.<ext> naming works)


Output (saved next to this script):
    face_embeddings.pkl   — sklearn SVM classifier + label encoder


Requirements (install once):
    pip install tensorflow mediapipe opencv-python scikit-learn pillow
"""


import os, re, pickle, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_FOLDER   = r"C:\Users\CSstudent\turretStuff\people"
OUTPUT_PKL       = r"C:\Users\CSstudent\turretStuff\face_embeddings.pkl"
FACE_MODEL_PATH  = r"C:\Users\CSstudent\turretStuff\blaze_face_short_range.tflite"
FACE_MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
                    "face_detector/blaze_face_short_range/float16/1/"
                    "blaze_face_short_range.tflite")
IMG_SIZE         = (96, 96)   # MobileNetV2 input size
PADDING          = 0.25       # fractional face-crop padding
MIN_CONFIDENCE   = 0.5
# ───────────────────────────────────────────────────────────────────────────────




def download_if_missing(path, url, label):
    if not os.path.isfile(path):
        import urllib.request
        print(f"Downloading {label} …")
        urllib.request.urlretrieve(url, path)
        print("Done.")




def build_face_detector():
    download_if_missing(FACE_MODEL_PATH, FACE_MODEL_URL, "face detector (~1 MB)")
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_CONFIDENCE,
    )
    return mp_vision.FaceDetector.create_from_options(opts)




def augment_crop(crop_bgr):
    """Return a list of augmented versions of a face crop."""
    augmented = [crop_bgr]
   
    # Horizontal flip
    augmented.append(cv2.flip(crop_bgr, 1))
   
    # Brightness variations
    for gamma in [0.75, 1.25]:
        lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                        for i in range(256)], dtype=np.uint8)
        augmented.append(cv2.LUT(crop_bgr, lut))
   
    # Small rotations
    h, w = crop_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        augmented.append(cv2.warpAffine(crop_bgr, M, (w, h)))
   
    return augmented


def crop_face(img_bgr, detection, padding=PADDING):
    """Return a tightly-padded BGR face crop, or None if invalid."""
    h, w = img_bgr.shape[:2]
    bb = detection.bounding_box
    x1 = int(bb.origin_x);        y1 = int(bb.origin_y)
    x2 = int(bb.origin_x + bb.width)
    y2 = int(bb.origin_y + bb.height)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    x1 = max(0, x1 - px);         y1 = max(0, y1 - py)
    x2 = min(w, x2 + px);         y2 = min(h, y2 + py)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return img_bgr[y1:y2, x1:x2]




def label_from_filename(fname):
    """
    'danielc1.png' → 'danielc'
    'dylanf2.jpg'  → 'dylanf'
    Strips trailing digits + extension.
    """
    stem = os.path.splitext(fname)[0]          # remove extension
    return re.sub(r'\d+$', '', stem).lower()   # strip trailing digits




def build_embedder():
    """MobileNetV2 without the top — outputs a 1280-d embedding vector."""
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                       pooling="avg")
    base.trainable = False
    return base




def embed(embedder, crop_bgr):
    """BGR crop → normalised 1280-d numpy vector."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    x = preprocess_input(resized.astype(np.float32))
    x = np.expand_dims(x, 0)
    return embedder.predict(x, verbose=0)[0]




# ── MAIN ───────────────────────────────────────────────────────────────────────


def main():
    folder = DEFAULT_FOLDER
    if len(sys.argv) > 1:
        folder = sys.argv[1]


    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in image_exts]


    if not files:
        print(f"No images found in: {folder}")
        sys.exit(1)


    print(f"Found {len(files)} image(s) in {folder}")


    detector  = build_face_detector()
    embedder  = build_embedder()


    X, y = [], []
    skipped = 0


    for fname in sorted(files):
        label = label_from_filename(fname)
        path  = os.path.join(folder, fname)
        img   = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Cannot read {fname}, skipping.")
            skipped += 1
            continue


        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_img)


        if not result.detections:
            print(f"  [WARN] No face detected in {fname}, skipping.")
            skipped += 1
            continue


        # Use the largest detection
        best = max(result.detections,
                   key=lambda d: d.bounding_box.width * d.bounding_box.height)
        crop = crop_face(img, best)
        if crop is None:
            print(f"  [WARN] Face crop too small in {fname}, skipping.")
            skipped += 1
            continue


        for aug_crop in augment_crop(crop):
            vec = embed(embedder, aug_crop)
            X.append(vec)
            y.append(label)
        print(f"  [OK]  {fname:30s} → label='{label}'")


    detector.close()


    if len(X) < 2:
        print("\nNeed at least 2 valid face images to train. Aborting.")
        sys.exit(1)


    X = np.array(X)
    y = np.array(y)


    unique_labels = sorted(set(y))
    print(f"\nLabels found: {unique_labels}")


    le = LabelEncoder()
    y_enc = le.fit_transform(y)


    clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",   # lower C
                probability=True, class_weight="balanced")),
    ])
   
    clf.fit(X, y_enc)


    payload = {
        "classifier":    clf,
        "label_encoder": le,
        "img_size":      IMG_SIZE,
        # Store a compact copy of the embedder weights path for runtime
        # (we re-build from imagenet weights at runtime — no need to pickle Keras)
        "embed_info":    "mobilenetv2_imagenet_avg",
    }


    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(payload, f)


    print(f"\n✓  Model saved → {OUTPUT_PKL}")
    print(f"   {len(X)} faces trained, {skipped} skipped.")
    print(f"   Classes: {list(le.classes_)}")


    # Quick leave-one-out accuracy hint
    from sklearn.model_selection import cross_val_score
    if len(X) >= len(unique_labels) * 2:
        scores = cross_val_score(clf, X, y_enc, cv=3, scoring="accuracy")
        print(f"CV accuracy: {scores.mean():.0%} ± {scores.std():.0%}")
       
    # Also print per-class counts so you can see imbalance:
    from collections import Counter
    print("Samples per label:", Counter(y))




if __name__ == "__main__":
    main()





