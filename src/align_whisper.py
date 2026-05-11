

import re

# Relative duration weights per phoneme class
_VOWELS  = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
_LONG_C  = {"L","M","N","NG","R","SH","ZH","TH","DH"}
_SHORT_C = {"B","D","G","JH","K","P","T"}

def _weight(phone: str) -> float:
    if phone in _VOWELS:  return 1.5
    if phone in _LONG_C:  return 1.1
    if phone in _SHORT_C: return 0.6
    return 1.0

# ARPAbet -> Azure-compatible viseme ID (same table as local_tts.py)
ARPABET_TO_VISEME = {
    "AA":2,"AE":1,"AH":1,"AO":3,"AW":9,"AY":11,"B":21,"CH":16,
    "D":19,"DH":17,"EH":4,"ER":5,"EY":4,"F":18,"G":20,"HH":12,
    "IH":6,"IY":6,"JH":16,"K":20,"L":14,"M":21,"N":19,"NG":20,
    "OW":8,"OY":10,"P":21,"R":13,"S":15,"SH":16,"T":19,"TH":17,
    "UH":7,"UW":7,"V":18,"W":7,"Y":6,"Z":15,"ZH":16,
}


def _load_cmudict():
    import nltk
    try:
        return nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download("cmudict", quiet=True)
        return nltk.corpus.cmudict.dict()


def _word_to_phonemes(word: str, cmudict: dict) -> list:
    """Return ARPAbet phonemes for a word (stress digits stripped)."""
    key = word.lower().strip("'")
    if key in cmudict:
        return [re.sub(r"\d", "", p) for p in cmudict[key][0]]
    # Letter-by-letter fallback
    return ["AH" if c in "aeiou" else "T" for c in key if c.isalpha()]


def _phonemes_in_window(phonemes: list, t_start_ms: float,
                         t_end_ms: float, viseme_events: list):
    """Append viseme events for phonemes distributed within [t_start_ms, t_end_ms]."""
    if not phonemes:
        return
    weights     = [_weight(p) for p in phonemes]
    total_w     = sum(weights) or 1.0
    duration_ms = max(1, t_end_ms - t_start_ms)
    t = t_start_ms
    for i, phone in enumerate(phonemes):
        viseme_events.append({
            "time_ms":   int(t),
            "viseme_id": ARPABET_TO_VISEME.get(phone, 0),
        })
        t += (weights[i] / total_w) * duration_ms


def align(audio_path: str, script_text: str) -> list:
    """
    Use faster-whisper to get word timestamps, then map to viseme events.

    Args:
        audio_path  : path to the generated audio file (wav or mp3)
        script_text : the original text (used for CMUdict lookup)

    Returns:
        list of {time_ms, viseme_id} dicts
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[Align] faster-whisper not installed.")
        print("[Align] Run:  pip install faster-whisper")
        return []

    print("[Align] Loading Whisper tiny model (downloads ~75MB on first run)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print(f"[Align] Transcribing: {audio_path}")
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
    )

    # Collect word-level timestamps
    word_times = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                clean = re.sub(r"[^a-zA-Z']", "", w.word).lower()
                if clean:
                    word_times.append({
                        "word":     clean,
                        "start_ms": w.start * 1000.0,
                        "end_ms":   w.end   * 1000.0,
                    })

    if not word_times:
        print("[Align] WARNING: Whisper found no words. Falling back to estimation.")
        return []

    print(f"[Align] {len(word_times)} words found with timestamps:")
    for wt in word_times[:8]:
        print(f"  '{wt['word']}'  {wt['start_ms']:.0f}ms - {wt['end_ms']:.0f}ms")
    if len(word_times) > 8:
        print(f"  ... and {len(word_times)-8} more")

    # Build viseme timeline
    cmudict = _load_cmudict()
    events  = [{"time_ms": 0, "viseme_id": 0}]   # leading silence

    for i, wt in enumerate(word_times):
        # Prevent slow interpolation across long silences:
        # Hold silence until right before the next word starts
        gap_start = 0 if i == 0 else word_times[i-1]["end_ms"]
        gap = wt["start_ms"] - gap_start
        if gap > 150:
            events.append({"time_ms": int(wt["start_ms"] - 80), "viseme_id": 0})

        phonemes = _word_to_phonemes(wt["word"], cmudict)
        _phonemes_in_window(phonemes, wt["start_ms"], wt["end_ms"], events)
        # Brief silence after each word
        events.append({"time_ms": int(wt["end_ms"]), "viseme_id": 0})

    # Sort by time (Whisper words should already be sorted, but be safe)
    events.sort(key=lambda e: e["time_ms"])

    print(f"[Align] {len(events)} viseme events aligned to audio.")
    return events
