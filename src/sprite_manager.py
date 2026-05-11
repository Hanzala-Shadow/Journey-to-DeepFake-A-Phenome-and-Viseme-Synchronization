import os
import cv2

import config
from src.face_mesh import get_landmarks

# Mapping of standard 22 Viseme IDs to our 3 basic sprite categories
# None means "use default mesh deformation without a sprite" (good for closed/silent/consonants)
VISEME_TO_SPRITE_BUCKET = {
    0:  None,       # silence
    1:  "teeth",    # ae, ah
    2:  "open",     # aa
    3:  "open",     # ao
    4:  "teeth",    # eh, ey
    5:  "teeth",    # er
    6:  "teeth",    # ih, iy
    7:  "pucker",   # uh, uw, w
    8:  "pucker",   # ow
    9:  "open",     # aw
    10: "pucker",   # oy
    11: "teeth",    # ay
    12: None,       # h
    13: "pucker",   # r
    14: "pucker",   # l
    15: "teeth",    # s, z
    16: "teeth",    # sh, ch, jh, zh
    17: "teeth",    # th, dh
    18: "teeth",    # f, v
    19: None,       # d, t, n (mostly closed)
    20: None,       # k, g, ng (mostly closed)
    21: None,       # p, b, m (closed)
}

def load_sprites() -> dict:
    """
    Scans the SPRITE_DIR for open.jpg, teeth.jpg, and pucker.jpg.
    For each, it detects landmarks and stores them.
    Returns: { "bucket_name": {"img": np.ndarray, "lms": list} }
    """
    sprites = {}
    
    if not os.path.exists(config.SPRITE_DIR):
        print(f"[SpriteManager] Directory {config.SPRITE_DIR} not found. Running without sprites.")
        return sprites

    for bucket in ["open", "teeth", "pucker"]:
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
