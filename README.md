# Face Synthesis Engine

A fully offline, CPU-based lip-sync animation and facial rendering engine. This system takes a single static face photo and a text script, generates audio locally, forced-aligns the audio to phonemes, and uses dense mesh piecewise affine warping to stretch and animate the face in sync with the speech. 

It features advanced Sprite-Based Texture Blending to seamlessly integrate real photorealistic mouth interiors (teeth/tongue), along with procedural 3D head drift and natural blinking.

## How to Use

Creating a new video requires setting up your input files and running two simple commands.

### 1. Setup Input Files

All user-provided files go into the `input/` directory.

*   **Script:** Open `input/script.txt` and type the text you want the face to say.
*   **Base Face:** Place a frontal, well-lit portrait photo named `face.jpg` inside the `input/` folder. The face must have a closed, relaxed mouth.
*   **Mouth Sprites (Optional but Recommended):** To achieve photorealistic lip-syncing, the engine blends real mouth textures inside the opening lips. Place three photos of the **same person** in the `input/sprites/` folder:
    *   `open.jpg`: Mouth wide open (like shouting "Ah").
    *   `teeth.jpg`: Wide smile showing teeth (like saying "Cheese").
    *   `pucker.jpg`: Lips pursed or rounded (like saying "Oo").
    *   *Note: If these files are missing, the engine gracefully falls back to rendering a procedural dark throat cavity.*

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
*   **Audio Generation:** Uses `pyttsx3` (Windows SAPI) to generate synthetic voice offline.
*   **Phonetic Mapping:** Uses the NLTK `cmudict` (Carnegie Mellon University Pronouncing Dictionary) to translate English text into raw ARPAbet phonemes.
*   **Viseme Mapping:** Maps the generated ARPAbet phonemes to a standard set of 22 visual phonemes (visemes) based on the Azure Cognitive Services standard.

### 2. Forced Alignment (`src/align_whisper.py`)
*   The system uses **faster-whisper** (OpenAI's Whisper model optimized for CPU via CTranslate2).
*   Whisper analyzes the generated `audio.wav` and returns exact millisecond timestamps for every spoken word.
*   The phonemes for each word are proportionally distributed inside that exact time window. Silence is explicitly held during pauses, ensuring the mouth snaps shut.

### 3. Face Warping & Rendering (`src/face_mesh.py` & `src/renderer.py`)
Instead of drawing primitive shapes, this project uses **Dense Mesh Image Warping** (Piecewise Affine Warping) and **Sprite-Based Texture Blending**.
*   **Landmark Detection:** Uses Google's MediaPipe Vision Tasks API (`face_landmarker`) to locate 468 3D landmarks on the static face photo and all sprite photos.
*   **Procedural Deformation (`src/viseme_mouth.py`):** Calculates how the 468 points should move based on the current viseme. The upper lip moves up, the lower lip moves down, and the entire jaw and chin drop naturally.
*   **Procedural Life:** Injects pseudo-random overlapping sine waves to create natural 3D head drift (parallax yaw/pitch) and calculates mathematically accurate anatomical blinking (including eyebrow dips).
*   **Delaunay Triangulation & Remapping:** Uses `scipy.spatial.Delaunay` and `LinearNDInterpolator` to map the rest-pose pixels to the deformed mesh coordinates. `cv2.remap` stretches the actual skin pixels.
*   **Sprite Blending (`src/sprite_manager.py`):** When the mouth opens, the system takes the target texture (e.g. `teeth.jpg`), warps it to match the exact shape of the opening mouth, and alpha-blends it inside the inner-lip polygon. When transitioning between sounds, the system perfectly cross-fades two different warped sprites simultaneously for infinite smoothness.

## File Structure

*   `main.py`: CLI entry point (`--tts`, `--render`, `--hologram`).
*   `config.py`: Central configuration for TTS modes and file paths.
*   `src/assembler.py`: The video loop. Reads the audio and viseme timeline, calculates the blend state for every frame (30 FPS), and writes to MP4.
*   `src/viseme_mouth.py`: The procedural animation logic for mouth shapes, head drift, and blinking.
*   `src/sprite_manager.py`: Handles loading and parsing mouth textures.
*   `src/face_mesh.py`: Landmark index definitions (which points belong to the lips, chin, eyes, etc).
