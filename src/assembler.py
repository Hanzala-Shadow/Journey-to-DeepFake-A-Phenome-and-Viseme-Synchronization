"""
assembler.py -- Full render pipeline.

1. Load face image -> detect landmarks (once, cached for all frames)
2. Load visemes.json -> build timeline
3. For each frame: look up active viseme, blend, deform, render
4. Write frames to temp_video.mp4 (silent)
5. Merge audio.mp3 + temp_video.mp4 -> result.mp4 (MoviePy)
"""

import json
import os

import cv2
import numpy as np
from moviepy.editor import AudioFileClip, VideoFileClip

import config
from src.face_mesh import get_landmarks
from src.renderer import render_frame
from src.viseme_mouth import compute_deformed_landmarks
from src.sprite_manager import load_sprites, get_sprite_for_viseme



def load_visemes(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def get_viseme_state(time_ms: float, events: list):
    """
    Given a point in time, find:
      - The current viseme (ID of the most recent event at or before time_ms)
      - The next viseme (ID of the first event after time_ms)
      - t_blend: how far we are from current -> next  (0.0 = just arrived, 1.0 = next)

    This drives the smoothstep interpolation in viseme_mouth.py.
    """
    prev_id, prev_t = 0, 0
    next_id, next_t = 0, (events[-1]["time_ms"] + 300) if events else 300

    for i, ev in enumerate(events):
        if ev["time_ms"] <= time_ms:
            prev_id = ev["viseme_id"]
            prev_t  = ev["time_ms"]
            if i + 1 < len(events):
                next_id = events[i + 1]["viseme_id"]
                next_t  = events[i + 1]["time_ms"]
            else:
                next_id = 0
                next_t  = prev_t + 300   # hold last shape for 300ms then close
        else:
            break

    duration = max(1, next_t - prev_t)
    t_blend  = min(1.0, (time_ms - prev_t) / duration)
    return prev_id, next_id, t_blend


def get_audio_duration(audio_path: str) -> float:
    clip = AudioFileClip(audio_path)
    dur  = clip.duration
    clip.close()
    return dur


# -- Main render pipeline -------------------------------------------------------

def run(
    face_path: str,
    viseme_path: str,
    audio_path: str,
    output_path: str,
    fps: int = 30,
    hologram: bool = True,
):
    # -- Step 1: landmarks (once) -----------------------------------------------
    print("[Assembler] Loading face + detecting landmarks...")
    base_img, base_lms, h, w = get_landmarks(face_path)

    # Resize to target resolution
    tw, th = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    if (w, h) != (tw, th):
        scale_x = tw / w
        scale_y = th / h
        base_img = cv2.resize(base_img, (tw, th))
        base_lms = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in base_lms]
        w, h = tw, th
        print(f"[Assembler]   Resized -> {tw}x{th}")

    # -- Step 2: viseme timeline ------------------------------------------------
    print("[Assembler] Loading viseme timeline...")
    events = load_visemes(viseme_path)
    print(f"[Assembler]   {len(events)} events  |  span: {events[0]['time_ms']} - {events[-1]['time_ms']}ms")

    sprites = load_sprites()

    # -- Step 3: frame count from audio ----------------------------------------
    print("[Assembler] Reading audio duration...")
    audio_dur    = get_audio_duration(audio_path)
    total_frames = int(audio_dur * fps) + fps   # +1s tail for silence fade
    print(f"[Assembler]   Audio: {audio_dur:.2f}s -> {total_frames} frames @ {fps}fps")

    # -- Step 4: render frame loop ----------------------------------------------
    os.makedirs(os.path.dirname(config.OUTPUT_TEMP_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(config.OUTPUT_TEMP_VIDEO, fourcc, fps, (w, h))

    print("[Assembler] Rendering frames...")
    for idx in range(total_frames):
        time_ms = (idx / fps) * 1000.0

        vis_a, vis_b, t = get_viseme_state(time_ms, events)
        deformed        = compute_deformed_landmarks(base_lms, vis_a, vis_b, t, frame_idx=idx)
        sprite_a        = get_sprite_for_viseme(vis_a, sprites)
        sprite_b        = get_sprite_for_viseme(vis_b, sprites)
        
        # Smooth the blend curve for texture crossfading (same as the mesh deformation)
        t_smooth = t * t * (3 - 2 * t)
        
        frame = render_frame(base_img, base_lms, deformed, 
                             sprite_a=sprite_a, sprite_b=sprite_b, t_blend=t_smooth,
                             apply_hologram=hologram, frame_idx=idx)
        writer.write(frame)

        # Progress every second of video
        if idx % fps == 0:
            pct = (idx / total_frames) * 100
            filled = int(pct / 5)
            bar = ('=' * filled) + ('-' * (20 - filled))
            print(f"  [{bar}] {pct:5.1f}%  t={time_ms/1000:.2f}s  "
                  f"vis={vis_a}->{vis_b}  blend={t:.2f}")

    writer.release()
    print(f"[Assembler] OK Silent video -> {config.OUTPUT_TEMP_VIDEO}")

    # -- Step 5: merge audio ----------------------------------------------------
    print("[Assembler] Merging audio with MoviePy...")
    video_clip = VideoFileClip(config.OUTPUT_TEMP_VIDEO)
    audio_clip = AudioFileClip(audio_path)

    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final = video_clip.set_audio(audio_clip)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,   # suppress verbose moviepy logs
    )
    video_clip.close()
    audio_clip.close()
    print(f"[Assembler] OK Final video -> {output_path}")
    print("[Assembler] Done! Open output/result.mp4 to review.")
