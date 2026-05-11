# Local Face Hologram Synthesis

A fully offline, CPU-based lip-sync animation and hologram rendering engine. This system takes a single static face photo and a text script, generates audio locally, forced-aligns the audio to phonemes, and uses dense mesh piecewise affine warping to stretch and animate the face in sync with the speech.

##  How to Use (Change Picture & Script)

Creating a new hologram video is incredibly simple and requires just two commands.

### 1. Update your Input Files
*   **The Script:** Open `input/script.txt` and type the words you want the hologram to say.
*   **The Face:** Place a frontal, well-lit portrait photo named `face.jpg` inside the `input/` folder. (Replace the existing one). *Note: The face should ideally have a closed, relaxed mouth.*

### 2. Generate Audio & Timing (TTS)
Run the following command to generate the voice audio and calculate the exact millisecond timings for lip movements:
```bash
python main.py --tts
```

### 3. Render the Video
Once the audio is generated, run the render command to warp the face and assemble the video:
```bash
python main.py --render
```
Your final video will be saved as **`output/result.mp4`**!

---

##  Technical Architecture

This project is built from scratch to avoid external dependencies or paid APIs (like Azure or HeyGen). It runs entirely locally. The pipeline is split into three main technical domains:

### 1. Text-To-Speech & Phoneme Extraction (`src/local_tts.py`)
*   **Audio Generation:** Uses `pyttsx3` (Windows SAPI) to generate synthetic voice entirely offline. 
*   **Phonetic Mapping:** Uses the NLTK `cmudict` (Carnegie Mellon University Pronouncing Dictionary) to translate English text into raw ARPAbet phonemes (e.g., "HELLO" -> `HH AH L OW`).
*   **Viseme Mapping:** Maps the generated ARPAbet phonemes to a standard set of 22 visual phonemes (visemes) based on the Azure Cognitive Services standard.

### 2. Forced Alignment (`src/align_whisper.py`)
Standard local TTS does not return timestamps for when words are spoken. To achieve perfect lip sync:
*   The system uses **faster-whisper** (OpenAI's Whisper model optimized for CPU via CTranslate2).
*   Whisper analyzes the generated `audio.wav` and returns exact millisecond `[start, end]` timestamps for every spoken word.
*   The phonemes for each word are proportionally distributed inside that exact time window. Silence is explicitly held during pauses, ensuring the mouth snaps shut when no audio is playing.

### 3. Face Warping & Rendering (`src/face_mesh.py` & `src/renderer.py`)
Instead of drawing primitive shapes or using heavy neural networks, this project uses **Dense Mesh Image Warping** (Piecewise Affine Warping) for ultra-fast, highly realistic animation.
*   **Landmark Detection:** Uses Google's modern MediaPipe Vision Tasks API (`face_landmarker`) to locate 468 3D landmarks on the static face photo.
*   **Procedural Deformation (`src/viseme_mouth.py`):** Calculates how the 468 points should move based on the current viseme. The upper lip moves up, the lower lip moves down, and the *entire jaw and chin drop naturally*. 
*   **Procedural Life:** Injects slow sine waves into the mesh to simulate breathing (head bobbing) and eyelid blinking.
*   **Delaunay Triangulation & Remapping:** Uses `scipy.spatial.Delaunay` and `LinearNDInterpolator` to map the rest-pose pixels to the deformed mesh coordinates. `cv2.remap` stretches the actual skin and photo pixels natively.
*   **Cavity Blending:** A dynamic, feathered dark ellipse (with simulated teeth) is drawn *behind* the stretched lips to simulate the inside of the throat.
*   **Hologram VFX:** Post-processing layers add scanlines, cyan tinting, edge-glow (Canny edge detection), and slight brightness flickering.

##  File Structure

*   `main.py`: CLI entry point (`--tts`, `--render`).
*   `config.py`: Central configuration for TTS modes, hologram VFX parameters, and file paths.
*   `debug_viewer.py`: An interactive GUI allowing you to press number keys to force viseme shapes on the face live.
*   `src/assembler.py`: The video loop. Reads the audio and viseme timeline, calculates the blend state for every frame (30 FPS), and writes to MP4.
*   `src/viseme_mouth.py`: The definitions of how wide/open the mouth should be for all 22 visemes.
*   `src/face_mesh.py`: Landmark index definitions (which points belong to the lips, chin, eyes, etc).
