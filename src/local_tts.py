

import json
import os
import re
import sys


# ARPAbet uses stress markers (0/1/2) -- strip them before lookup
ARPABET_TO_VISEME = {
    "AA": 2,   # ah (ɑ)        -- open
    "AE": 1,   # ae (æ)        -- short uh
    "AH": 1,   # uh (ʌ, ə)    -- schwa
    "AO": 3,   # aw (ɔ)        -- rounded open
    "AW": 9,   # ow (aʊ)
    "AY": 11,  # eye (aɪ)
    "B":  21,  # b             -- lips pressed
    "CH": 16,  # ch (tʃ)       -- sh/ch
    "D":  19,  # d             -- d/t/n
    "DH": 17,  # th voiced (ð)
    "EH": 4,   # eh (ɛ)
    "ER": 5,   # er (ɝ)
    "EY": 4,   # ey (eɪ)       -- close to eh
    "F":  18,  # f             -- lip-teeth
    "G":  20,  # g             -- back
    "HH": 12,  # h             -- breathy
    "IH": 6,   # ih (ɪ)        -- ee
    "IY": 6,   # iy (i)        -- ee
    "JH": 16,  # j (dʒ)        -- sh/ch
    "K":  20,  # k             -- back
    "L":  14,  # l
    "M":  21,  # m             -- lips pressed
    "N":  19,  # n             -- d/t/n
    "NG": 20,  # ng (ŋ)        -- back
    "OW": 8,   # ow (o)        -- oh
    "OY": 10,  # oy (ɔɪ)
    "P":  21,  # p             -- lips pressed
    "R":  13,  # r (ɹ)
    "S":  15,  # s             -- s/z
    "SH": 16,  # sh (ʃ)
    "T":  19,  # t             -- d/t/n
    "TH": 17,  # th unvoiced (θ)
    "UH": 7,   # uh (ʊ)        -- oo (short)
    "UW": 7,   # uw (u)        -- oo
    "V":  18,  # v             -- lip-teeth
    "W":  7,   # w             -- oo (rounded)
    "Y":  6,   # y (j)         -- ee
    "Z":  15,  # z             -- s/z
    "ZH": 16,  # zh (ʒ)        -- sh/ch
}

SILENCE_VISEME = 0
INTER_WORD_SILENCE_MS = 60   # gap between words (ms)
LEADING_SILENCE_MS    = 80   # silence at start before first word


# -- Step 1: Text -> phoneme sequence -------------------------------------------

def _text_to_phonemes(text: str):
    """
    Convert text to a list of (word, [ARPAbet phonemes]) tuples.
    Uses nltk CMUdict. Words not found get a simple vowel/consonant fallback.

    Returns:
        List of ARPAbet phoneme strings (stress digits stripped), e.g.:
        ["HH", "AH", "L", "OW"]   for "hello"
    """
    try:
        import nltk
        try:
            cmudict = nltk.corpus.cmudict.dict()
        except LookupError:
            print("[LocalTTS] Downloading nltk cmudict corpus (one-time, ~6MB)...")
            nltk.download("cmudict", quiet=True)
            cmudict = nltk.corpus.cmudict.dict()
    except ImportError:
        print("[LocalTTS] nltk not found -- run:  pip install nltk")
        sys.exit(1)

    # Clean text: lowercase, split on whitespace, strip punctuation per word
    words = re.findall(r"[a-zA-Z']+", text.lower())
    all_phonemes = []

    unknown = []
    for word in words:
        if word in cmudict:
            # CMUdict may have multiple pronunciations -- use first
            phones = cmudict[word][0]
            # Strip stress digits (e.g. "AH0" -> "AH", "EY1" -> "EY")
            phones = [re.sub(r"\d", "", p) for p in phones]
            all_phonemes.extend(phones)
        else:
            # Fallback: treat each letter as a rough approximation
            unknown.append(word)
            for ch in word:
                # Map vowels -> AH, consonants -> a neutral consonant
                if ch in "aeiou":
                    all_phonemes.append("AH")
                else:
                    all_phonemes.append("T")

    if unknown:
        print(f"[LocalTTS] WARNING - Words not in CMUdict (fallback used): {unknown}")

    return all_phonemes


# -- Step 2: Build viseme timeline from phonemes + audio duration ---------------

_VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
_LONG_C = {"L","M","N","NG","R","SH","ZH","TH","DH"}
_SHORT_C = {"B","D","G","JH","K","P","T"}

def _phoneme_weight(phone: str) -> float:
    if phone in _VOWELS:  return 1.5
    if phone in _LONG_C:  return 1.1
    if phone in _SHORT_C: return 0.6
    return 1.0

def _estimate_viseme_events(phonemes: list, audio_duration_s: float) -> list:
    """Distribute phonemes weighted by class (vowels longer, stops shorter)."""
    if not phonemes:
        return [{"time_ms": 0, "viseme_id": SILENCE_VISEME}]

    total_ms       = audio_duration_s * 1000.0
    n_words        = max(1, len(phonemes) // 4)
    silence_budget = LEADING_SILENCE_MS + n_words * INTER_WORD_SILENCE_MS
    weights        = [_phoneme_weight(p) for p in phonemes]
    ms_per_unit    = (total_ms - silence_budget) / max(1, sum(weights))

    events = [{"time_ms": 0, "viseme_id": SILENCE_VISEME}]
    t = float(LEADING_SILENCE_MS)

    for i, phone in enumerate(phonemes):
        viseme_id = ARPABET_TO_VISEME.get(phone, SILENCE_VISEME)
        events.append({"time_ms": int(t), "viseme_id": viseme_id})
        t += ms_per_unit * weights[i]
        if (i + 1) % 4 == 0 and i < len(phonemes) - 1:
            events.append({"time_ms": int(t), "viseme_id": SILENCE_VISEME})
            t += INTER_WORD_SILENCE_MS

    events.append({"time_ms": int(total_ms - 50), "viseme_id": SILENCE_VISEME})
    return events


# -- Audio backend: pyttsx3 ----------------------------------------------------

def _synthesize_pyttsx3(text: str, audio_out: str, voice_index: int = 0):
    """
    Generate speech using Windows SAPI (pyttsx3). Saves as WAV.
    Returns the output path (may be .wav instead of .mp3).
    """
    try:
        import pyttsx3
    except ImportError:
        print("[LocalTTS] pyttsx3 not found -- run:  pip install pyttsx3")
        sys.exit(1)

    # Force .wav extension (pyttsx3 cannot save as mp3 natively)
    wav_out = os.path.splitext(audio_out)[0] + ".wav"
    os.makedirs(os.path.dirname(wav_out) or ".", exist_ok=True)

    engine = pyttsx3.init()

    # List available voices, pick a male one
    voices = engine.getProperty("voices")
    male_voices = [v for v in voices if "david" in v.name.lower()
                   or "mark" in v.name.lower()
                   or "male" in v.name.lower()
                   or "guy" in v.name.lower()]

    if male_voices:
        engine.setProperty("voice", male_voices[0].id)
        print(f"[LocalTTS] Voice: {male_voices[0].name}")
    elif voices:
        engine.setProperty("voice", voices[0].id)
        print(f"[LocalTTS] Voice (fallback): {voices[0].name}")

    engine.setProperty("rate",   165)   # words per minute (natural pace)
    engine.setProperty("volume", 1.0)

    engine.save_to_file(text, wav_out)
    engine.runAndWait()

    print(f"[LocalTTS] Audio saved -> {wav_out}")
    return wav_out


# -- Audio backend: gTTS --------------------------------------------------------

def _synthesize_gtts(text: str, audio_out: str):
    """
    Generate speech using Google TTS.
    Saves directly as MP3.
    """
    try:
        from gtts import gTTS
    except ImportError:
        print("[LocalTTS] gTTS not found -- run:  pip install gtts")
        sys.exit(1)

    os.makedirs(os.path.dirname(audio_out) or ".", exist_ok=True)
    tts = gTTS(text=text, lang="en", tld="com", slow=False)
    tts.save(audio_out)
    print(f"[LocalTTS] Audio saved -> {audio_out}")
    return audio_out


# -- Audio backend: Edge TTS ----------------------------------------------------

def _synthesize_edge(text: str, audio_out: str):
    """
    Generate speech using Microsoft Edge Neural TTS.
    Saves directly as MP3.
    """
    import subprocess
    import sys
    os.makedirs(os.path.dirname(audio_out) or ".", exist_ok=True)
    
    # Check if edge-tts is installed
    try:
        # Use ChristopherNeural (excellent male voice) or GuyNeural
        voice = "en-US-ChristopherNeural" 
        cmd = [sys.executable, "-m", "edge_tts", "--text", text, "--voice", voice, "--write-media", audio_out]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[LocalTTS] Audio saved -> {audio_out}")
    except FileNotFoundError:
        print("[LocalTTS] edge-tts not found -- run:  pip install edge-tts")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[LocalTTS] edge-tts failed: {e.stderr.decode()}")
        sys.exit(1)
        
    return audio_out


# -- Public API -----------------------------------------------------------------

def synthesize(script_text: str, audio_out: str, viseme_out: str,
               backend: str = "pyttsx3"):
    """
    Generates audio + visemes.json.

    Args:
        script_text : text to speak
        audio_out   : path for audio file (mp3 or wav depending on backend)
        viseme_out  : path for visemes.json
        backend     : "pyttsx3" (local), "gtts" (Google), or "edge" (Edge Neural)

    Returns:
        (actual_audio_path, viseme_events list)
    """
    print(f"[LocalTTS] Backend  : {backend}")
    print(f"[LocalTTS] Text     : {script_text[:80]}{'...' if len(script_text)>80 else ''}")

    # -- Generate audio --------------------------------------------------------
    if backend == "pyttsx3":
        actual_audio = _synthesize_pyttsx3(script_text, audio_out)
    elif backend == "gtts":
        actual_audio = _synthesize_gtts(script_text, audio_out)
    elif backend == "edge":
        actual_audio = _synthesize_edge(script_text, audio_out)
    else:
        print(f"[LocalTTS] Unknown backend '{backend}'. Use 'pyttsx3', 'gtts', or 'edge'.")
        sys.exit(1)

    # -- Get audio duration ----------------------------------------------------
    try:
        from moviepy.editor import AudioFileClip
        clip     = AudioFileClip(actual_audio)
        duration = clip.duration
        clip.close()
        print(f"[LocalTTS] Audio duration: {duration:.2f}s")
    except Exception as e:
        print(f"[LocalTTS] Warning: could not read audio duration ({e}). Using estimate.")
        duration = len(script_text.split()) * 0.4   # ~0.4s per word fallback

    # -- Get phonemes ----------------------------------------------------------
    print("[LocalTTS] Extracting phonemes from text (CMUdict)...")
    phonemes = _text_to_phonemes(script_text)
    print(f"[LocalTTS] {len(phonemes)} phonemes found")

    # -- Build viseme timeline -------------------------------------------------
    import config
    if getattr(config, "USE_WHISPER_ALIGN", False):
        from src.align_whisper import align
        events = align(actual_audio, script_text)
        if not events:
            print("[LocalTTS] Whisper alignment failed/empty. Falling back to estimation.")
            events = _estimate_viseme_events(phonemes, duration)
    else:
        events = _estimate_viseme_events(phonemes, duration)

    print(f"[LocalTTS] {len(events)} viseme events generated")

    os.makedirs(os.path.dirname(viseme_out) or ".", exist_ok=True)
    with open(viseme_out, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[LocalTTS] Visemes saved -> {viseme_out}")

    # -- Debug: first 10 events ------------------------------------------------
    labels = {
        0:"silence", 1:"uh", 2:"ah", 3:"aw", 4:"eh", 5:"er",
        6:"ee", 7:"oo", 8:"oh", 9:"ow", 10:"oy", 11:"eye",
        12:"h", 13:"r", 14:"l", 15:"s/z", 16:"sh", 17:"th",
        18:"f/v", 19:"d/t/n", 20:"k/g", 21:"p/b/m",
    }
    print("\n[LocalTTS] First 10 viseme events:")
    print(f"  {'time_ms':>8}   viseme   meaning")
    print(f"  {'-'*8}   ------   -------")
    for ev in events[:10]:
        vid = ev["viseme_id"]
        print(f"  {ev['time_ms']:>8}ms    {vid:>2}     {labels.get(vid,'?')}")

    return actual_audio, events
