"""
tts_azure.py -- One-time Azure TTS call.
Produces: output/audio.mp3 + output/visemes.json

Key insight: Azure fires a viseme_received event for EVERY phoneme change.
  evt.audio_offset  -> time in 100-nanosecond ticks  (divide by 10,000 -> ms)
  evt.viseme_id     -> integer 0-21 (the mouth shape to show)
"""

import json
import os
import sys

import azure.cognitiveservices.speech as speechsdk


def synthesize(script_text, key, region, voice, audio_out, viseme_out):
    """
    Call Azure TTS and save audio + viseme timeline to disk.

    Returns:
        list of {time_ms, viseme_id} dicts, sorted by time.
    """
    if key == "YOUR_KEY_HERE":
        print("[ERROR] Azure key not set.")
        print("        Edit config.py  ->  AZURE_SPEECH_KEY = 'your-actual-key'")
        print("        OR run:  set AZURE_SPEECH_KEY=your-actual-key  (then re-run)")
        sys.exit(1)

    os.makedirs(os.path.dirname(audio_out), exist_ok=True)

    # -- Speech config ----------------------------------------------------------
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_out)
    synthesizer  = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    # -- Collect viseme events --------------------------------------------------
    viseme_events = []

    def on_viseme(evt):
        viseme_events.append({
            "time_ms":   evt.audio_offset // 10_000,  # 100ns ticks -> ms
            "viseme_id": evt.viseme_id,
        })

    synthesizer.viseme_received.connect(on_viseme)

    # -- Synthesize -------------------------------------------------------------
    print(f"[TTS] Voice       : {voice}")
    print(f"[TTS] Region      : {region}")
    print(f"[TTS] Script text : {script_text[:80]}{'...' if len(script_text) > 80 else ''}")
    print("[TTS] Calling Azure TTS...")

    result = synthesizer.speak_text_async(script_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"[TTS] OK Audio saved  -> {audio_out}")
        print(f"[TTS] OK {len(viseme_events)} viseme events captured")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"[TTS] ✗ Synthesis canceled: {details.reason}")
        if details.reason == speechsdk.CancellationReason.Error:
            print(f"[TTS]   Error code    : {details.error_code}")
            print(f"[TTS]   Error details : {details.error_details}")
            print("[TTS]   -> Check your key and region in config.py")
        sys.exit(1)

    # -- Save visemes -----------------------------------------------------------
    with open(viseme_out, "w") as f:
        json.dump(viseme_events, f, indent=2)
    print(f"[TTS] OK Visemes saved -> {viseme_out}")

    # -- Debug: print first 10 events -------------------------------------------
    print("\n[TTS] First 10 viseme events (verify timing looks right):")
    print(f"  {'time_ms':>8}   viseme_id   meaning")
    print(f"  {'-'*8}   ---------   -------")
    labels = {
        0:"silence", 1:"uh", 2:"ah", 3:"aw", 4:"eh", 5:"er",
        6:"ee", 7:"oo", 8:"oh", 9:"ow", 10:"oy", 11:"eye",
        12:"h", 13:"r", 14:"l", 15:"s/z", 16:"sh", 17:"th",
        18:"f/v", 19:"d/t/n", 20:"k/g", 21:"p/b/m",
    }
    for ev in viseme_events[:10]:
        vid = ev["viseme_id"]
        print(f"  {ev['time_ms']:>8}ms      {vid:>2}         {labels.get(vid,'?')}")

    return viseme_events
