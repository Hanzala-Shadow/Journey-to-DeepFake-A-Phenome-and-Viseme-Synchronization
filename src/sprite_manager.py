import os
import cv2

import config
from src.face_mesh import get_landmarks

# Mapping of 22 Azure Visemes to the 15 Standard Oculus/MPEG-4 Viseme shapes
# Missing sprites will default to None (geometric mesh cavity deformation).
VISEME_TO_SPRITE_BUCKET = {
    0:  "sil",      # silence
    1:  "e",        # uh / schwa (eh, uh)
    2:  "aa",       # ah (wide open)
    3:  "o",        # aw (open round)
    4:  "e",        # eh (half open)
    5:  "rr",       # er
    6:  "i",        # ee (wide smile)
    7:  "u",        # oo (puckered)
    8:  "o",        # oh (round)
    9:  "aa",       # ow
    10: "o",        # oy
    11:  "aa",      # ay
    12: "kk",       # h
    13: "rr",       # r
    14: "nn",       # l
    15: "ss",       # s, z
    16: "ch",       # sh, ch, jh, zh
    17: "th",       # th, dh
    18: "ff",       # f, v
    19: "dd",       # d, t, n
    20: "kk",       # k, g, ng
    21: "pp",       # p, b, m
}

def load_sprites() -> dict:
    """
    Scans the SPRITE_DIR for the 15 standard viseme images.
    For each, it detects landmarks and stores them.
    Returns: { "bucket_name": {"img": np.ndarray, "lms": list} }
    """
    sprites = {}
    
    if not os.path.exists(config.SPRITE_DIR):
        print(f"[SpriteManager] Directory {config.SPRITE_DIR} not found. Running without sprites.")
        return sprites

    # The 15 standard Oculus shapes based on ArtStation reference
    viseme_buckets = [
        "sil", "pp", "ff", "th", "dd", "kk", "ch", "ss", 
        "nn", "rr", "aa", "e", "i", "o", "u"
    ]

    for bucket in viseme_buckets:
        path = os.path.join(config.SPRITE_DIR, f"{bucket}.jpg")
        if os.path.exists(path):
            try:
                # get_landmarks expects a file path and returns (img, lms, h, w)
                img, lms, h, w = get_landmarks(path)
            except Exception as e:
                print(f"[SpriteManager] Error loading {bucket}.jpg: {e}")
                continue
            
            if not lms:
                print(f"[SpriteManager] No face detected in sprite: {bucket}.jpg")
                continue
            
            sprites[bucket] = {"img": img, "lms": lms}
            print(f"[SpriteManager] Loaded sprite: {bucket}.jpg")
            
    if not sprites:
        print("[SpriteManager] No valid sprites found. Using default cavity rendering.")
        
    return sprites

def get_sprite_for_viseme(viseme_id: int, sprites: dict):
    """Return the (img, lms) for the given viseme, or None if no sprite available/needed."""
    bucket = VISEME_TO_SPRITE_BUCKET.get(viseme_id)
    if not bucket or bucket not in sprites:
        return None
    return sprites[bucket]
