

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from src.face_mesh import (
    UPPER_LIP_MOVERS, LOWER_LIP_MOVERS, LIP_CORNERS,
    CHIN_MOVERS, STATIC_ANCHORS, MOUTH_OUTER, UPPER_LIP, LOWER_LIP
)
import config

# Global cache for the Delaunay triangulation and grid so we don't rebuild them every frame
_CACHE = {}


def _get_warp_maps(base_lms, deformed_lms, h_img, w_img):
    """
    Generate dense displacement maps (map_x, map_y) for cv2.remap.
    Uses SciPy's LinearNDInterpolator over a Delaunay triangulation of the face points.
    """
    global _CACHE

    # Extract only the points we care about for the warp (lips, chin, and anchors)
    active_indices = list(set(
        UPPER_LIP_MOVERS + LOWER_LIP_MOVERS + LIP_CORNERS + CHIN_MOVERS + STATIC_ANCHORS
    ))

    # Add the 4 corners of the image as static anchors so the background doesn't warp
    corners = np.array([
        [0, 0], [w_img - 1, 0], [0, h_img - 1], [w_img - 1, h_img - 1],
        [w_img // 2, 0], [w_img // 2, h_img - 1], [0, h_img // 2], [w_img - 1, h_img // 2]
    ])

    base_pts = np.array([base_lms[i] for i in active_indices], dtype=np.float32)
    def_pts  = np.array([deformed_lms[i] for i in active_indices], dtype=np.float32)

    base_pts = np.vstack([base_pts, corners])
    def_pts  = np.vstack([def_pts, corners])

    # We map DESTINATION pixels back to SOURCE pixels to pull the colors properly
    if "tri" not in _CACHE:
        # Triangulate the DEFORMED points (since we are doing inverse mapping)
        _CACHE["tri"] = Delaunay(base_pts)  # We actually want to triangulate the base points for stability
        
        # Create full image grid
        grid_x, grid_y = np.meshgrid(np.arange(w_img), np.arange(h_img))
        _CACHE["grid_x"] = grid_x
        _CACHE["grid_y"] = grid_y

    # We need to map from deformed space -> base space to fill the new image
    # But for small deformations, mapping base -> deformed and interpolating is fine
    # Actually, proper remap requires inverse mapping: given pixel (x,y) in output, where did it come from?
    # So we triangulate the DEFORMED points and interpolate the BASE coordinates.
    tri_def = Delaunay(def_pts)
    
    interp_x = LinearNDInterpolator(tri_def, base_pts[:, 0])
    interp_y = LinearNDInterpolator(tri_def, base_pts[:, 1])

    map_x = interp_x(_CACHE["grid_x"], _CACHE["grid_y"]).astype(np.float32)
    map_y = interp_y(_CACHE["grid_x"], _CACHE["grid_y"]).astype(np.float32)

    # Fill NaNs (pixels outside the convex hull of our points) with identity mapping
    nan_mask = np.isnan(map_x)
    map_x[nan_mask] = _CACHE["grid_x"][nan_mask]
    map_y[nan_mask] = _CACHE["grid_y"][nan_mask]

    return map_x, map_y


def _draw_single_sprite(canvas: np.ndarray, deformed_lms: list, sprite_data: dict) -> np.ndarray:
    """Warp a single sprite and alpha-blend its inner mouth onto the canvas."""
    h, w = canvas.shape[:2]
    sprite_img = sprite_data["img"]
    sprite_lms = sprite_data["lms"]
    
    active_indices = list(set(UPPER_LIP_MOVERS + LOWER_LIP_MOVERS + LIP_CORNERS + CHIN_MOVERS + STATIC_ANCHORS))
    s_pts = np.array([sprite_lms[i] for i in active_indices], dtype=np.float32)
    d_pts = np.array([deformed_lms[i] for i in active_indices], dtype=np.float32)
    
    corners = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]
    ])
    s_pts = np.vstack([s_pts, corners])
    d_pts = np.vstack([d_pts, corners])
    
    tri_def = Delaunay(d_pts)
    interp_x = LinearNDInterpolator(tri_def, s_pts[:, 0])
    interp_y = LinearNDInterpolator(tri_def, s_pts[:, 1])
    
    if "grid_x" not in _CACHE:
        _CACHE["grid_x"], _CACHE["grid_y"] = np.meshgrid(np.arange(w), np.arange(h))
        
    map_x = interp_x(_CACHE["grid_x"], _CACHE["grid_y"]).astype(np.float32)
    map_y = interp_y(_CACHE["grid_x"], _CACHE["grid_y"]).astype(np.float32)
    
    nan_mask = np.isnan(map_x)
    map_x[nan_mask] = _CACHE["grid_x"][nan_mask]
    map_y[nan_mask] = _CACHE["grid_y"][nan_mask]
    
    warped_sprite = cv2.remap(sprite_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    inner_pts = np.array([[deformed_lms[i][0], deformed_lms[i][1]] for i in inner_lip_indices], dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [inner_pts], 255)
    
    upper_ys = [deformed_lms[i][1] for i in UPPER_LIP_MOVERS]
    lower_ys = [deformed_lms[i][1] for i in LOWER_LIP_MOVERS]
    mouth_h = abs(np.mean(lower_ys) - np.mean(upper_ys))
    
    if mouth_h > 3:
        fk = max(3, int(mouth_h // 4) * 2 + 1)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (fk, fk), 0)[:, :, np.newaxis]
        result = warped_sprite.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha)
        return result.astype(np.uint8)
    return canvas

def _draw_fallback_cavity(canvas: np.ndarray, deformed_lms: list) -> np.ndarray:
    """Draw the procedural dark throat cavity."""
    h, w = canvas.shape[:2]
    lc_x = deformed_lms[LIP_CORNERS[0]][0]
    rc_x = deformed_lms[LIP_CORNERS[1]][0]
    lc_y = deformed_lms[LIP_CORNERS[0]][1]
    rc_y = deformed_lms[LIP_CORNERS[1]][1]
    cx = (lc_x + rc_x) // 2
    cy = (lc_y + rc_y) // 2

    upper_ys = [deformed_lms[i][1] for i in UPPER_LIP_MOVERS]
    lower_ys = [deformed_lms[i][1] for i in LOWER_LIP_MOVERS]
    upper_y = int(np.mean(upper_ys))
    lower_y = int(np.mean(lower_ys))
    
    mouth_w = abs(rc_x - lc_x)
    mouth_h = abs(lower_y - upper_y)

    if mouth_h < 3:
        return canvas

    cavity = canvas.copy()
    
    half_w = int(mouth_w * 0.42)
    half_h = mouth_h // 2
    mid_y = (upper_y + lower_y) // 2

    cv2.ellipse(cavity, (cx, mid_y), (half_w, half_h), 0, 0, 360, (20, 10, 15), -1)

    if mouth_h > 8:
        teeth_h = max(2, int(half_h * 0.35))
        teeth_w = int(half_w * 0.7)
        cv2.ellipse(cavity, (cx, upper_y + teeth_h), (teeth_w, teeth_h), 0, 0, 360, (180, 178, 170), -1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, mid_y), (half_w, half_h), 0, 0, 360, 200, -1)
    
    fk = max(5, (half_h // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (fk, fk), 0)
    alpha = cv2.GaussianBlur(blurred, (fk, fk), 0)[:, :, np.newaxis]

    result = cavity.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha)
    return result.astype(np.uint8)


def _apply_hologram(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tinted = np.zeros_like(frame)
    tinted[:, :, 0] = (gray * 0.90).astype(np.uint8)
    tinted[:, :, 1] = (gray * 1.00).astype(np.uint8)
    tinted[:, :, 2] = (gray * 0.25).astype(np.uint8)
    s = config.HOLOGRAM_TINT_STRENGTH
    frame = cv2.addWeighted(frame, 1.0 - s, tinted, s, 0)

    scan = np.ones(frame.shape, dtype=np.float32)
    scan[::4, :] = config.SCANLINE_DARKEN
    frame = np.clip(frame.astype(np.float32) * scan, 0, 255).astype(np.uint8)

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray2, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    glow = np.zeros_like(frame)
    glow[:, :, 0] = edges
    glow[:, :, 1] = edges
    frame = cv2.addWeighted(frame, 1.0, glow, config.EDGE_GLOW_STRENGTH, 0)

    flicker = 1.0 + config.FLICKER_AMPLITUDE * np.sin(frame_idx * 0.37)
    frame = np.clip(frame.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
    return frame


def render_frame(base_img: np.ndarray, base_landmarks: list,
                 deformed_landmarks: list, sprite_a: dict = None, sprite_b: dict = None, t_blend: float = 0.0,
                 apply_hologram: bool = True, frame_idx: int = 0) -> np.ndarray:
    """
    Render one video frame using Dense Mesh Warping and Sprite Blending.
    """
    h, w = base_img.shape[:2]

    # 1. Generate the displacement maps to stretch the skin
    map_x, map_y = _get_warp_maps(base_landmarks, deformed_landmarks, h, w)

    # 2. Warp the image (this physically moves the pixels of the lips and chin)
    warped_face = cv2.remap(base_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 3. Add the dark cavity or realistic sprite teeth behind the newly stretched lips
    canvas_a = _draw_single_sprite(warped_face, deformed_landmarks, sprite_a) if sprite_a else _draw_fallback_cavity(warped_face, deformed_landmarks)
    canvas_b = _draw_single_sprite(warped_face, deformed_landmarks, sprite_b) if sprite_b else _draw_fallback_cavity(warped_face, deformed_landmarks)
    
    # Smoothly crossfade between the two inner mouth textures
    canvas = cv2.addWeighted(canvas_b, t_blend, canvas_a, 1.0 - t_blend, 0)

    # 4. Apply Hologram VFX
    if apply_hologram:
        canvas = _apply_hologram(canvas, frame_idx)
        
    return canvas
