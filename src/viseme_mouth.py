"""
viseme_mouth.py -- Maps Azure viseme IDs to mouth geometry deformations.

This is the core of the lip sync:
  - VISEME_PARAMS defines (openness, width_factor) for each of the 22 IDs
  - compute_deformed_landmarks() moves the lip landmark points geometrically
  - smoothstep() ensures no hard jumps between shapes

Tuning guide (run debug_viewer.py to see changes live):
  openness     : 0.0 = fully closed, 1.0 = very wide open ("ah")
  width_factor : 1.0 = natural width, >1.0 = stretched wide ("ee"),
                 <1.0 = pursed/rounded ("oo")
"""

import numpy as np

from src.face_mesh import (
    UPPER_LIP_MOVERS, LOWER_LIP_MOVERS, LIP_CORNERS,
)

# -- Viseme shape table ---------------------------------------------------------
# (openness 0.0-1.0, width_factor 0.8-1.15)
VISEME_PARAMS = {
    0:  (0.00, 1.00),   # silence / rest -- closed
    1:  (0.20, 1.00),   # uh / schwa (ə, ʌ)
    2:  (0.90, 1.05),   # ah (ɑ) -- widest open
    3:  (0.70, 0.92),   # aw (ɔ) -- open + slightly rounded
    4:  (0.35, 1.00),   # eh (ɛ, ʊ)
    5:  (0.40, 0.90),   # er (ɝ)
    6:  (0.25, 1.12),   # ee (j, i, ɪ) -- wide spread lips
    7:  (0.35, 0.80),   # oo (w, u) -- very rounded/pursed
    8:  (0.55, 0.88),   # oh (o) -- rounded
    9:  (0.60, 0.95),   # ow (aʊ)
    10: (0.50, 0.90),   # oy (ɔɪ)
    11: (0.65, 1.00),   # eye (aɪ)
    12: (0.30, 1.00),   # h -- breathy, slightly open
    13: (0.30, 0.90),   # r (ɹ) -- slight rounding
    14: (0.25, 1.00),   # l
    15: (0.10, 1.00),   # s/z -- near-closed, teeth close
    16: (0.15, 0.85),   # sh/ch (ʃ, tʃ) -- narrow + rounded
    17: (0.20, 1.00),   # th (ð) -- slight opening
    18: (0.10, 1.00),   # f/v -- near-closed
    19: (0.15, 1.00),   # d/t/n -- tongue on palate
    20: (0.30, 1.00),   # k/g -- back of mouth
    21: (0.00, 1.00),   # p/b/m -- lips pressed together
}

# How much to scale the mouth height at openness=1.0
# (e.g. 2.5 means the mouth opens to 3.5x its rest height)
OPEN_SCALE = 2.5

# Upper lip moves 40% of target height upward; lower lip 60% downward
# (lower jaw naturally drops more than upper lip rises)
UPPER_SPLIT = 0.40
LOWER_SPLIT = 0.60


# -- Math helpers ---------------------------------------------------------------

def smoothstep(t: float) -> float:
    """
    Cubic ease-in/out. Eliminates velocity spikes at viseme boundaries.
      t=0.0 -> 0.0   (zero velocity)
      t=0.5 -> 0.5   (max velocity at midpoint)
      t=1.0 -> 1.0   (zero velocity)
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# -- Core deformation -----------------------------------------------------------

def compute_deformed_landmarks(
    base_landmarks: list,
    viseme_id_a: int,
    viseme_id_b: int,
    t_blend: float,
    frame_idx: int = 0
) -> list:
    """
    Compute pixel positions of all 468 landmarks after blending two viseme shapes.

    Args:
        base_landmarks  : list of [x_px, y_px] -- rest-pose positions (468 items)
    Given two visemes and a blend factor (0.0 -> 1.0), calculate the new coordinates
    for all 468 landmarks. Also applies procedural eye blinking and head bobbing.
    """
    import copy
    import math
    lms = copy.deepcopy(base_landmarks)

    open_a, width_a = VISEME_PARAMS[viseme_id_a]
    open_b, width_b = VISEME_PARAMS[viseme_id_b]
    t         = smoothstep(t_blend)
    openness  = lerp(open_a, open_b, t)
    width_fac = lerp(width_a, width_b, t)

    # -- Compute rest-pose mouth geometry --------------------------------------
    upper_ys     = [base_landmarks[i][1] for i in UPPER_LIP_MOVERS]
    lower_ys     = [base_landmarks[i][1] for i in LOWER_LIP_MOVERS]
    upper_center = float(np.mean(upper_ys))
    lower_center = float(np.mean(lower_ys))
    base_height  = abs(lower_center - upper_center)
    mouth_mid_y  = (upper_center + lower_center) / 2.0

    corner_l_x   = float(base_landmarks[LIP_CORNERS[0]][0])
    corner_r_x   = float(base_landmarks[LIP_CORNERS[1]][0])
    mouth_mid_x  = (corner_l_x + corner_r_x) / 2.0

    # -- Apply vertical deformation --------------------------------------------
    target_height = base_height * (1.0 + openness * OPEN_SCALE)
    lower_shift = (target_height * LOWER_SPLIT) - (base_height * LOWER_SPLIT)
    upper_shift = (target_height * UPPER_SPLIT) - (base_height * UPPER_SPLIT)

    for idx in UPPER_LIP_MOVERS:
        lms[idx][1] -= int(upper_shift)

    for idx in LOWER_LIP_MOVERS:
        lms[idx][1] += int(lower_shift)

    # Shift chin/jaw points down by a similar amount (fade out slightly)
    from src.face_mesh import CHIN_MOVERS
    for idx in CHIN_MOVERS:
        lms[idx][1] += int(lower_shift * 0.85)

    # -- Apply horizontal deformation (width) ----------------------------------
    for idx in LIP_CORNERS:
        orig_x = float(base_landmarks[idx][0])
        offset = (orig_x - mouth_mid_x) * (width_fac - 1.0)
        lms[idx][0] = int(orig_x + offset)

    # -- Procedural Animation (Breathing & Blinking) ---------------------------
    import random
    time_sec = frame_idx / 30.0

    # 1. Natural Head Movement (pseudo-random drift)
    # Combine prime-frequency sine waves for natural, non-repeating movement
    yaw_drift   = math.sin(time_sec * 0.73) + 0.5 * math.sin(time_sec * 1.37)
    pitch_drift = math.sin(time_sec * 0.59) + 0.5 * math.sin(time_sec * 1.13)
    
    # Apply uniformly to avoid tearing the piecewise affine mesh
    shift_x = int(yaw_drift * 2.5)
    shift_y = int(pitch_drift * 2.5)
    for pt in lms:
        pt[0] += shift_x
        pt[1] += shift_y

    # 2. Natural Eye Blinking
    # Trigger a blink roughly once every 4 seconds, deterministically based on time
    window_idx = int(time_sec / 4.0)
    random.seed(window_idx * 12345)
    blink_time = (window_idx * 4.0) + random.uniform(0.5, 3.5)
    
    # A blink lasts ~0.2 seconds (0.1s close, 0.1s open)
    diff = abs(time_sec - blink_time)
    blink_amt = 0.0
    if diff < 0.1:
        # Smooth cosine curve for the blink
        blink_amt = math.cos((diff / 0.1) * (math.pi / 2))
        blink_amt = blink_amt * blink_amt # Ease in/out

    if blink_amt > 0.01:
        LEFT_EYE_TOP  = [159, 160, 158]
        LEFT_EYE_BOT  = [145, 144, 153]
        RIGHT_EYE_TOP = [386, 385, 387]
        RIGHT_EYE_BOT = [374, 380, 373]
        LEFT_BROW     = [70, 63, 105, 66, 107]
        RIGHT_BROW    = [300, 293, 334, 296, 336]

        # In a real blink, the upper lid moves down a lot (80%), lower lid moves up a bit (20%)
        # And the eyebrows dip slightly.
        
        # Left Eye
        for top, bot in zip(LEFT_EYE_TOP, LEFT_EYE_BOT):
            close_y = lms[bot][1] * 0.2 + lms[top][1] * 0.8
            lms[top][1] = int(lms[top][1] * (1.0 - blink_amt) + close_y * blink_amt)
            lms[bot][1] = int(lms[bot][1] * (1.0 - blink_amt) + close_y * blink_amt)
        for brow in LEFT_BROW:
            lms[brow][1] += int(3.0 * blink_amt)

        # Right Eye
        for top, bot in zip(RIGHT_EYE_TOP, RIGHT_EYE_BOT):
            close_y = lms[bot][1] * 0.2 + lms[top][1] * 0.8
            lms[top][1] = int(lms[top][1] * (1.0 - blink_amt) + close_y * blink_amt)
            lms[bot][1] = int(lms[bot][1] * (1.0 - blink_amt) + close_y * blink_amt)
        for brow in RIGHT_BROW:
            lms[brow][1] += int(3.0 * blink_amt)

    return lms


# -- New simple API (used by assembler + debug_viewer) -------------------------

def get_blended_shape(
    viseme_id_a: int,
    viseme_id_b: int,
    t_blend: float,
) -> tuple:
    """
    Return (openness, width_factor) blended between two viseme shapes.

    Args:
        viseme_id_a : current viseme  (0-21)
        viseme_id_b : next viseme to blend toward (0-21)
        t_blend     : progress 0.0 -> 1.0 (smoothstepped inside)

    Returns:
        (openness, width_factor) both as floats
    """
    open_a, width_a = VISEME_PARAMS[viseme_id_a]
    open_b, width_b = VISEME_PARAMS[viseme_id_b]
    t = smoothstep(t_blend)
    return lerp(open_a, open_b, t), lerp(width_a, width_b, t)
