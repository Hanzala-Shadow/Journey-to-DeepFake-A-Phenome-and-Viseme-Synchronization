# Face Synthesis Engine

A fully offline, CPU-based lip-sync animation and facial rendering engine. This system takes a single static face photo and a text script, generates audio locally, forced-aligns the audio to phonemes, and uses dense mesh piecewise affine warping to stretch and animate the face in sync with the speech. 

It features advanced Sprite-Based Texture Blending to seamlessly integrate real photorealistic mouth interiors (teeth/tongue), along with procedural 3D head drift and natural blinking.

## How to Use

Creating a new video requires setting up your input files and running two simple commands.

### 1. Setup Input Files

All user-provided files go into the `input/` directory.

*   **Script:** Open `input/script.txt` and type the text you want the face to say.
*   **Base Face:** Place a frontal, well-lit portrait photo named `face.jpg` inside the `input/` folder. The face must have a closed, relaxed mouth.
*   **Mouth Sprites (Optional but Recommended):** To achieve photorealistic lip-syncing, the engine blends real mouth textures inside the opening lips using a standard 15-viseme Oculus/MPEG-4 mapping. Place the following 15 photos of the **same person** in the `input/sprites/` folder:
    *   `sil.jpg`, `pp.jpg`, `ff.jpg`, `th.jpg`, `dd.jpg`, `kk.jpg`, `ch.jpg`, `ss.jpg`, `nn.jpg`, `rr.jpg`, `aa.jpg`, `e.jpg`, `i.jpg`, `o.jpg`, `u.jpg`
    *   *Note: If any of these files are missing, the engine gracefully falls back to rendering a procedural geometric cavity for that specific viseme.*

### 2. Generate Audio & Timing (TTS)

Run the following command to generate the voice audio and calculate the exact millisecond timings for lip movements. You only need to run this when your script changes.

```bash
python main.py --tts
```

### 3. Render the Video

Once the audio is generated, run the render command to warp the face, blend the sprites, and assemble the video:

**Normal Mode (Photorealistic RGB):**
```bash
python main.py --render
```

**Hologram Mode (Sci-Fi VFX):**
If you want to apply the legacy cyan tint, scanlines, and edge-glow VFX, use the hologram flag:
```bash
python main.py --render --hologram
```

Your final video will be saved as **`output/result.mp4`**.

---

## Technical Architecture

This project is built from scratch to avoid heavy PyTorch/GPU dependencies or paid APIs. It runs entirely locally on the CPU. The pipeline is split into three main technical domains:

### 1. Text-To-Speech & Phoneme Extraction (`src/local_tts.py`)
*   **Audio Generation:** Supports multiple TTS backends, configurable in `config.py`. Options include:
    *   `edge`: Microsoft Edge Neural TTS (High-quality cloud voices)
    *   `gtts`: Google TTS (Standard cloud voices)
    *   `pyttsx3`: Windows SAPI (Offline local voices)
    *   `azure`: Azure Cognitive Services (Requires API key)
*   **Phonetic Mapping:** Uses the NLTK `cmudict` (Carnegie Mellon University Pronouncing Dictionary) to translate English text into raw ARPAbet phonemes.
*   **Viseme Mapping:** Maps the generated ARPAbet phonemes to a standard set of 22 visual phonemes (visemes).

### 2. Forced Alignment (`src/align_whisper.py`)
*   The system uses **faster-whisper** (OpenAI's Whisper model optimized for CPU via CTranslate2).
*   Whisper analyzes the generated audio and returns exact millisecond timestamps for every spoken word.
*   The phonemes for each word are proportionally distributed inside that exact time window. Silence is explicitly held during pauses, ensuring the mouth snaps shut.

### 3. Face Warping & Rendering (`src/face_mesh.py` & `src/renderer.py`)
*   **Landmark Detection:** Uses Google's MediaPipe Vision Tasks API (`face_landmarker`) to locate 468 3D landmarks on the static face photo and all sprite photos.
*   **Procedural Deformation (`src/viseme_mouth.py`):** Calculates how the 468 points should move based on the current viseme.
*   **Procedural Life:** Injects pseudo-random overlapping sine waves to create natural 3D head drift (parallax yaw/pitch) and calculates mathematically accurate anatomical blinking.
*   **Delaunay Triangulation & Remapping:** Uses `scipy.spatial.Delaunay` and `LinearNDInterpolator` to map the rest-pose pixels to the deformed mesh coordinates. `cv2.remap` stretches the actual skin pixels.
*   **Sprite Blending (`src/sprite_manager.py`):** Maps the 22 Azure visemes to the 15 standard Oculus/MPEG-4 sprite buckets. The system warps the target sprite to match the shape of the opening mouth and alpha-blends it inside the inner-lip polygon, smoothly cross-fading textures during sound transitions.

## File Structure

*   `main.py`: CLI entry point (`--tts`, `--render`, `--hologram`).
*   `config.py`: Central configuration for TTS modes and file paths.
*   `src/assembler.py`: The video loop. Reads the audio and viseme timeline, calculates the blend state, and writes to MP4.
*   `src/viseme_mouth.py`: Procedural animation logic for mouth shapes, head drift, and blinking.
*   `src/sprite_manager.py`: Handles loading and parsing the 15-viseme mouth textures.
*   `src/face_mesh.py`: Landmark index definitions.
