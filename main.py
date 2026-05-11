"""
main.py -- Entry point for the face hologram lip sync prototype.

Usage:
  python main.py --tts          # Step 1 (one-time): Azure TTS -> audio.mp3 + visemes.json
  python main.py --render       # Step 2 (iterative): render output.mp4 from saved files
  python main.py --render --no-hologram  # Render without cyan effect (debug)

Workflow:
  1. Add your API key to config.py  (AZURE_SPEECH_KEY)
  2. Put your face photo at input/face.jpg
  3. Edit input/script.txt with the text to speak
  4. Run:  python main.py --tts
  5. Run:  python main.py --render
  6. Open: output/result.mp4
"""

import argparse
import os
import sys


def run_tts():
    import config
    from src.local_tts import synthesize as local_synthesize
    from src.tts_azure import synthesize as azure_synthesize

    if not os.path.exists(config.INPUT_SCRIPT):
        print(f"[ERROR] Script file not found: {config.INPUT_SCRIPT}")
        sys.exit(1)

    with open(config.INPUT_SCRIPT, "r", encoding="utf-8") as f:
        script_text = f.read().strip()

    if not script_text:
        print("[ERROR] input/script.txt is empty -- add your text first.")
        sys.exit(1)

    mode = config.TTS_MODE
    print(f"[Main] TTS Mode: {mode}")

    if mode == "azure":
        azure_synthesize(
            script_text=script_text,
            key=config.AZURE_SPEECH_KEY,
            region=config.AZURE_SPEECH_REGION,
            voice=config.AZURE_VOICE_NAME,
            audio_out=config.OUTPUT_AUDIO,
            viseme_out=config.OUTPUT_VISEMES,
        )
    elif mode in ("pyttsx3", "gtts"):
        actual_audio, _ = local_synthesize(
            script_text=script_text,
            audio_out=config.OUTPUT_AUDIO,   # may become .wav for pyttsx3
            viseme_out=config.OUTPUT_VISEMES,
            backend=mode,
        )
        # Update OUTPUT_AUDIO in memory so --render finds the right file
        config.OUTPUT_AUDIO = actual_audio
    else:
        print(f"[ERROR] Unknown TTS_MODE '{mode}' in config.py")
        print("        Valid options: 'pyttsx3', 'gtts', 'azure'")
        sys.exit(1)



def run_render(hologram: bool):
    import config
    from src.assembler import run

    # pyttsx3 saves as .wav -- auto-detect the actual audio file
    audio_path = config.OUTPUT_AUDIO
    wav_fallback = os.path.splitext(config.OUTPUT_AUDIO)[0] + ".wav"
    if not os.path.exists(audio_path) and os.path.exists(wav_fallback):
        audio_path = wav_fallback
        print(f"[Main] Using WAV audio: {wav_fallback}")

    missing = []
    for path, label in [
        (config.INPUT_FACE,     "input/face.jpg      <- your face photo"),
        (config.OUTPUT_VISEMES, "output/visemes.json <- run --tts first"),
        (audio_path,            "output/audio.*      <- run --tts first"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  MISSING: {path}   ({label})")

    if missing:
        print("[ERROR] Required files not found:")
        for m in missing:
            print(m)
        sys.exit(1)

    run(
        face_path=config.INPUT_FACE,
        viseme_path=config.OUTPUT_VISEMES,
        audio_path=audio_path,
        output_path=config.OUTPUT_VIDEO,
        fps=config.VIDEO_FPS,
        hologram=hologram,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Hologram Lip Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="[One-time] Call Azure TTS -> saves audio.mp3 + visemes.json",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="[Main loop] Render lip-synced hologram video",
    )
    parser.add_argument(
        "--no-hologram", action="store_true",
        help="Skip cyan tint/scanlines (raw mouth animation only -- useful for tuning)",
    )

    args = parser.parse_args()

    if not args.tts and not args.render:
        parser.print_help()
        sys.exit(0)

    if args.tts:
        run_tts()

    if args.render:
        run_render(hologram=not args.no_hologram)
