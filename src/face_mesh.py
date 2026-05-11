"""
face_mesh.py -- MediaPipe FaceMesh landmark extraction.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.x).
On first run, automatically downloads the face_landmarker.task model (~1.8MB).

MediaPipe coordinate system:
  landmark.x / landmark.y -> normalized 0.0-1.0 (relative to image size)
  landmark.z              -> depth (unused here)
  pixel_x = int(landmark.x * image_width)
  pixel_y = int(landmark.y * image_height)

Key mouth indices out of the 468-point mesh:
  61  -> left lip corner
  291 -> right lip corner
  0   -> center of upper lip (top)
  17  -> center of lower lip (bottom)
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# -- Model download (one-time, ~1.8MB) -----------------------------------------
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join("models", "face_landmarker.task")


def _ensure_model():
    """Download the FaceLandmarker model if not already present."""
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    print(f"[FaceMesh] Downloading face landmarker model (~1.8MB) ...")
    print(f"[FaceMesh]   From : {_MODEL_URL}")
    print(f"[FaceMesh]   To   : {_MODEL_PATH}")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"[FaceMesh] Model download complete.")


# -- Mouth landmark index groups ------------------------------------------------

# Full outer lip ring (used to draw/fill mouth region)
MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Upper lip polygon (drawn in skin color)
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

# Lower lip polygon (drawn in skin color)
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# Points that move UP when mouth opens (upper lip edge)
UPPER_LIP_MOVERS = [0, 267, 269, 270, 409, 37, 39, 40, 185, 13, 312, 311, 310, 82, 81, 80]

# Points that move DOWN when mouth opens (lower lip edge)
LOWER_LIP_MOVERS = [17, 84, 181, 91, 146, 314, 405, 321, 375, 14, 87, 178, 88, 95, 317, 402]

# Jaw and chin points that drop when the mouth opens
CHIN_MOVERS = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
               377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251,
               200, 199, 175, 152]

# Static anchor points to prevent the upper face/eyes/nose from moving (by default)
# We also include the eye and eyebrow points here so they are part of the active warp mesh,
# allowing us to procedurally blink and animate them!
STATIC_ANCHORS = [4, 6, 8, 9, 33, 263, 10, 151, 109, 338,
                  # Eyes
                  159, 160, 158, 145, 144, 153, 386, 385, 387, 374, 380, 373,
                  # Eyebrows
                  70, 63, 105, 66, 107, 300, 293, 334, 296, 336]

# Lip corner indices (move horizontally for wide/narrow shapes)
LIP_CORNERS = [61, 291]


def get_landmarks(image_path: str):
    """
    Load a face image and extract all 468 MediaPipe face landmarks.
    Uses the Tasks API (mediapipe >= 0.10.x).

    Returns:
        img       : original BGR image (ndarray)
        landmarks : list of [x_px, y_px] for all 468 points
        h, w      : image height and width in pixels
    """
    _ensure_model()   # download model on first run

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Cannot load image: '{image_path}'\n"
            "Place your frontal face photo at input/face.jpg"
        )

    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -- MediaPipe Tasks API ----------------------------------------------------
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError(
            "No face detected in image.\n"
            "Tips: use a clear frontal photo with good lighting, neutral expression."
        )

    # Convert normalized coords to pixel coords (same format as before)
    landmarks = [
        [int(lm.x * w), int(lm.y * h)]
        for lm in result.face_landmarks[0]
    ]

    # Debug output
    lc = landmarks[LIP_CORNERS[0]]
    rc = landmarks[LIP_CORNERS[1]]
    uc = landmarks[0]
    dc = landmarks[17]
    print(f"[FaceMesh] 468 landmarks detected  (image: {w}x{h}px)")
    print(f"[FaceMesh]   Left corner  : {lc}")
    print(f"[FaceMesh]   Right corner : {rc}")
    print(f"[FaceMesh]   Mouth width  : {abs(rc[0] - lc[0])}px")
    print(f"[FaceMesh]   Mouth height : {abs(dc[1] - uc[1])}px  (at rest)")

    return img, landmarks, h, w


def get_mouth_bbox(landmarks: list, padding: int = 20):
    """
    Axis-aligned bounding box around the outer mouth ring.
    Returns (x1, y1, x2, y2).
    """
    pts = [landmarks[i] for i in MOUTH_OUTER]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)
