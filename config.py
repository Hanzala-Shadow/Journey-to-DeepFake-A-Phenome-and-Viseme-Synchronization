import os

# -- TTS Mode -------------------------------------------------------------------
# Supported TTS Engines:
#   "pyttsx3" -> Windows SAPI (Local, default)
#   "gtts"    -> Google TTS (Cloud)
#   "edge"    -> Microsoft Edge Neural TTS (Cloud)
#   "azure"   -> Azure Cognitive Services (Cloud, requires API key)
TTS_MODE = "edge"

# Use faster-whisper for exact word-level timing (highly recommended for local TTS)
USE_WHISPER_ALIGN = True

# -- Azure TTS (only used when TTS_MODE = "azure") -----------------------------
AZURE_SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "YOUR_KEY_HERE")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AZURE_VOICE_NAME    = "en-US-GuyNeural"   # Male voice

# -- File Paths -----------------------------------------------------------------
INPUT_FACE    = "input/face.jpg"
INPUT_SCRIPT  = "input/script.txt"
SPRITE_DIR    = "input/sprites/"

OUTPUT_AUDIO      = "output/audio.mp3"
OUTPUT_VISEMES    = "output/visemes.json"
OUTPUT_TEMP_VIDEO = "output/temp_video.mp4"
OUTPUT_VIDEO      = "output/result.mp4"

# Ensure directories exist since they are gitignored
os.makedirs("input/sprites", exist_ok=True)
os.makedirs("output", exist_ok=True)

# -- Video Settings -------------------------------------------------------------
VIDEO_FPS    = 30
VIDEO_WIDTH  = 1280
VIDEO_HEIGHT = 720

# -- Hologram Effect ------------------------------------------------------------
HOLOGRAM_TINT_STRENGTH = 0.60   # 0.0 = no tint, 1.0 = full cyan
SCANLINE_DARKEN        = 0.65   # brightness of every 4th row (0-1)
EDGE_GLOW_STRENGTH     = 0.40   # opacity of edge glow layer
FLICKER_AMPLITUDE      = 0.04   # subtle brightness oscillation amplitude
