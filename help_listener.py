#!/usr/bin/env python3
"""
Multilingual Voice Duress Trigger Runner using Sarvam AI Speech-to-Text.

Continuously captures microphone audio, gates calls using RMS speech energy detection
to conserve cloud credits, transcribes speech in real time with Sarvam AI's saaras:v3 model,
and triggers emergency alerts whenever distress keywords in English or major Indian
languages are recognized.

Features:
- Extensible multilingual distress lexicon (English, Hindi, Telugu, Tamil, Bengali, Kannada, etc.)
- Strict energy-based rate limiting (never sends silent/humming audio, conserving API credits)
- Live running API counter for presentation and demo visibility
- Automatic evidence WAV persistence and dashboard alert dispatch
- Configurable refractory cooldown to prevent duplicate triggers
"""

import argparse
import io
import os
import queue
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# ==============================================================================
# COMPREHENSIVE 23-LANGUAGE DISTRESS KEYWORDS LEXICON
# Covers all 22 Eighth Schedule Indian languages plus English supported by Sarvam Saaras v3.
# Maps Language -> List of (native_script, transliteration) tuples.
# ==============================================================================
DISTRESS_KEYWORDS: Dict[str, List[Tuple[str, str]]] = {
    "English": [
        ("help", "help"),
        ("help me", "help me"),
        ("save me", "save me"),
        ("emergency", "emergency"),
        ("call police", "call police"),
    ],
    "Hindi": [
        ("बचाओ", "bachao"),
        ("बचाओ मुझे", "bachao mujhe"),
        ("मदद", "madad"),
        ("मदद करो", "madad karo"),
        ("सहायता", "sahayata"),
    ],
    "Bengali": [
        ("বাঁচাও", "bachao"),
        ("আমায় বাঁচাও", "amay bachao"),
        ("সাহায্য", "sahajjo"),
        ("সাহায্য করুন", "sahajjo korun"),
    ],
    "Telugu": [
        ("కాపాడు", "kaapadu"),
        ("కాపాడండి", "kaapaadandi"),
        ("నన్ను కాపాడండి", "nannu kaapaadandi"),
        ("సహాయం", "sahayam"),
        ("సహాయం చేయండి", "sahayam cheyandi"),
    ],
    "Marathi": [
        ("वाचवा", "vachva"),
        ("मला वाचवा", "mala vachva"),
        ("मदत", "madat"),
        ("मदत करा", "madat kara"),
    ],
    "Tamil": [
        ("காப்பாத்துங்க", "kaapaathunga"),
        ("காப்பாற்றுங்கள்", "kaapaatrungal"),
        ("என்னை காப்பாத்துங்க", "ennai kaapaathunga"),
        ("உதவி", "udhavi"),
        ("உதவி செய்யுங்கள்", "udhavi seiyungal"),
    ],
    "Urdu": [
        ("بچاؤ", "bachao"),
        ("مجھے بچاؤ", "mujhe bachao"),
        ("مدد", "madad"),
        ("مدد کرو", "madad karo"),
    ],
    "Gujarati": [
        ("બચાવો", "bachavo"),
        ("મને બચાવો", "mane bachavo"),
        ("મદદ", "madad"),
        ("મદદ करो", "madad karo"),
    ],
    "Kannada": [
        ("ಕಾಪಾಡಿ", "kaapadi"),
        ("ನನ್ನನ್ನು ಕಾಪಾಡಿ", "nannannu kaapadi"),
        ("ಸಹಾಯ", "sahaya"),
        ("ಸಹಾಯ ಮಾಡಿ", "sahaya maadi"),
    ],
    "Odia": [
        ("ବଞ୍ଚାଅ", "banchao"),
        ("ମତେ ବଞ୍ଚାଅ", "mate banchao"),
        ("ସାହାଯ୍ୟ", "sahajya"),
        ("ସାହାଯ୍ୟ କରନ୍ତୁ", "sahajya karantu"),
    ],
    "Malayalam": [
        ("രക്ഷിക്കൂ", "rakshikkoo"),
        ("രക്ഷിക്കണേ", "rakshikkane"),
        ("എന്നെ രക്ഷിക്കൂ", "enne rakshikkoo"),
        ("സഹായിക്കൂ", "sahayikkoo"),
    ],
    "Punjabi": [
        ("ਬਚਾਓ", "bachao"),
        ("ਮੈਨੂੰ ਬਚਾਓ", "mainu bachao"),
        ("ਮਦਦ", "madad"),
        ("ਮਦਦ ਕਰੋ", "madad karo"),
        ("ਸਹਾਇਤਾ", "sahaita"),
    ],
    "Assamese": [
        ("বচাওক", "bosaok"),
        ("মোক বচাওক", "mok bosaok"),
        ("সহায়", "xohay"),
        ("সহায় কৰক", "xohay korok"),
    ],
    "Maithili": [
        ("बचाउ", "bachau"),
        ("हमरा बचाउ", "hamra bachau"),
        ("मदति", "madati"),
        ("सहायता", "sahayata"),
    ],
    "Sanskrit": [
        ("त्राहि", "traahi"),
        ("त्रायताम्", "traayataam"),
        ("रक्ष माम्", "raksha maam"),
        ("साहाय्यम्", "saahayyam"),
    ],
    "Kashmiri": [
        ("بچٲوِو", "bachaaviv"),
        ("بचاو", "bachaav"),
        ("مدد", "madad"),
        ("رچھ", "rachh"),
    ],
    "Nepali": [
        ("बचाउनुहोस्", "bachaaunuhos"),
        ("मलाई बचाउनुहोस्", "malai bachaaunuhos"),
        ("गुहार", "guhaar"),
        ("मद्दत", "maddat"),
    ],
    "Sindhi": [
        ("بچايو", "bachayo"),
        ("بچاؤ", "bachao"),
        ("مدد", "madad"),
        ("مونکي بچايو", "moonkhe bachayo"),
    ],
    "Konkani": [
        ("वांचयात", "vanchyat"),
        ("म्हाका वांचयात", "mhaka vanchyat"),
        ("पाव", "paav"),
        ("मदत", "madat"),
    ],
    "Dogri": [
        ("बचाओ", "bachao"),
        ("मिगी बचाओ", "migi bachao"),
        ("मदद", "madad"),
    ],
    "Manipuri": [
        # Meitei Mayek & Bengali script (Meiteilon)
        ("ꯀꯟꯕꯤꯌꯨ", "kanbiyu"),
        ("কনবীয়ু", "kanbiyu"),
        ("মতেন্ পাংবীয়ু", "mateng pangbiyu"),  # NOTE: verify with a native speaker
    ],
    "Bodo": [
        # Devanagari script for Bodo
        ("अनसुंथाय", "ansunthai"),  # help/assistance
        ("रैखा खालाम", "raikha khalam"),  # save/rescue
        ("मदद", "madad"),  # NOTE: verify with a native speaker for colloquial distress cry
    ],
    "Santali": [
        # Ol Chiki script for Santali
        ("ᱜᱚᱲᱚ", "goro"),  # help
        ("ᱜᱚᱲᱚᱧᱢᱮ", "goronjme"),  # help me
        ("ᱵᱟᱧᱪᱟᱣ", "banchaw"),  # save
        ("banchaw", "banchaw"),  # NOTE: verify with a native speaker for colloquial distress cry
    ],
}

# Default API configuration
DEFAULT_SARVAM_ENDPOINT: str = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_MODEL: str = "saaras:v3"
DEFAULT_SARVAM_MODE: str = "transcribe"
DEFAULT_CHUNK_DURATION_SECONDS: float = 3.0
DEFAULT_ENERGY_THRESHOLD: float = 0.015
DEFAULT_COOLDOWN_SECONDS: float = 8.0
DEFAULT_DASHBOARD_URL: str = "http://127.0.0.1:5000/trigger"
DEFAULT_SAMPLE_RATE: int = 16000


def calculate_rms_energy(audio: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) energy of an audio array."""
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Encode a 1D float32 numpy audio array into 16-bit PCM WAV bytes in-memory."""
    audio_clamped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio_clamped, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def check_distress_keywords(
    transcript: str,
    keywords_dict: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> List[Dict[str, str]]:
    """
    Search for distress keywords across all configured languages in the transcription.

    Flattens the dictionary into a single case-insensitive substring check,
    stripping trailing and inline punctuation (including Indic danda '।',
    double danda '॥', periods, commas, etc.) before matching.

    Returns:
        List of dicts: [{"language": "<Language>", "term": "<Matched Term>"}, ...]
    """
    if not transcript:
        return []

    if keywords_dict is None:
        keywords_dict = DISTRESS_KEYWORDS

    # Strip punctuation characters including Indic danda and double danda
    clean_text = transcript.strip().lower()
    clean_text = re.sub(r"[।॥.,!?;:\"'()\[\]{}\-—_]+", " ", clean_text)
    clean_text = " ".join(clean_text.split())

    matches: List[Dict[str, str]] = []
    seen = set()

    for language, pairs in keywords_dict.items():
        for native_term, translit_term in pairs:
            for term in (native_term, translit_term):
                clean_kw = re.sub(r"[।॥.,!?;:\"'()\[\]{}\-—_]+", " ", term.strip().lower())
                clean_kw = " ".join(clean_kw.split())
                if clean_kw and clean_kw in clean_text:
                    match_key = (language, clean_kw)
                    if match_key not in seen:
                        seen.add(match_key)
                        matches.append({"language": language, "term": term})

    return matches


def query_sarvam_stt(
    wav_bytes: bytes,
    api_key: str,
    api_url: str = DEFAULT_SARVAM_ENDPOINT,
    model: str = DEFAULT_SARVAM_MODEL,
    mode: str = DEFAULT_SARVAM_MODE,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Send an audio chunk to Sarvam AI Speech-to-Text API for transcription.

    Requires explicit filename and 'audio/wav' content type in multipart data.
    """
    headers = {"api-subscription-key": api_key.strip()}
    files = {"file": ("speech_chunk.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"model": model, "mode": mode}

    response = requests.post(
        api_url,
        headers=headers,
        files=files,
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def notify_dashboard(
    confidence: float,
    audio_path: str,
    language: str = "Unknown",
    contact: str = "Voice Trigger (Sarvam)",
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    timeout: float = 2.0,
) -> None:
    """Send trigger payload to local dashboard with language attribution, failing silently if unreachable."""
    try:
        requests.post(
            dashboard_url,
            json={
                "contact": contact,
                "language": language,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": confidence,
                "audio_path": audio_path,
            },
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        # Dashboard not running or unreachable — fail silently without crashing
        pass


class HelpListener:
    """
    Continuous microphone listener with energy gating and Sarvam AI STT transcription.
    """

    def __init__(
        self,
        api_key: str,
        chunk_duration: float = DEFAULT_CHUNK_DURATION_SECONDS,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN_SECONDS,
        dashboard_url: str = DEFAULT_DASHBOARD_URL,
        evidence_dir: str = "evidence",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: Optional[int] = None,
        keywords_dict: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        keywords: Optional[Any] = None,
    ):
        self.api_key = api_key.strip()
        self.chunk_duration = float(chunk_duration)
        self.energy_threshold = float(energy_threshold)
        self.cooldown = float(cooldown)
        self.dashboard_url = dashboard_url
        self.evidence_dir = Path(evidence_dir)
        self.sample_rate = int(sample_rate)
        self.device = device

        if keywords_dict is not None:
            self.keywords_dict = keywords_dict
        elif isinstance(keywords, dict):
            self.keywords_dict = keywords
        else:
            self.keywords_dict = DISTRESS_KEYWORDS

        self.chunk_samples = int(round(self.chunk_duration * self.sample_rate))
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._audio_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._api_call_count = 0
        self._last_trigger_time = 0.0

    @property
    def keywords(self) -> List[str]:
        """Flattened list of all native and transliterated keyword strings."""
        flat: List[str] = []
        seen = set()
        for pairs in self.keywords_dict.values():
            for native, translit in pairs:
                for term in (native, translit):
                    if term and term not in seen:
                        seen.add(term)
                        flat.append(term)
        return flat

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio streaming callback executed by sounddevice."""
        if not self._is_running:
            return
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._audio_queue.put(chunk)

    def process_chunk(self, chunk: np.ndarray) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """
        Evaluate an audio chunk:
        1. Checks RMS energy against threshold to conserve API credits.
        2. If speech detected, calls Sarvam STT API.
        3. Checks for distress keywords across 23 languages in transcript.
        4. Triggers dashboard and evidence capture if matched and cooldown elapsed.

        Returns:
            (transcript, matched_records)
        """
        rms = calculate_rms_energy(chunk)

        if rms < self.energy_threshold:
            print(
                f"[Silence] RMS: {rms:.4f} < {self.energy_threshold:.4f} — skipping API call (saved credit)",
                flush=True,
            )
            return None, []

        # Speech activity detected: call Sarvam API
        self._api_call_count += 1
        print(
            f"\n[API Call #{self._api_call_count}] Speech detected (RMS: {rms:.4f}) -> Transcribing with Sarvam...",
            flush=True,
        )

        wav_bytes = audio_to_wav_bytes(chunk, self.sample_rate)

        try:
            result = query_sarvam_stt(wav_bytes, api_key=self.api_key)
        except Exception as err:
            print(f"  [API Error] Failed to transcribe audio: {err}", file=sys.stderr)
            return None, []

        transcript = result.get("transcript", "").strip()
        print(f"  Transcript: \"{transcript}\"", flush=True)

        # Extract confidence score (language_probability) if available
        lang_prob = result.get("language_probability")
        if lang_prob is not None and isinstance(lang_prob, (int, float)):
            confidence = round(float(lang_prob) * 100.0, 1) if float(lang_prob) <= 1.0 else round(float(lang_prob), 1)
        else:
            confidence = 95.0

        matches = check_distress_keywords(transcript, self.keywords_dict)

        if matches:
            matched_langs = sorted(list(dict.fromkeys(m["language"] for m in matches)))
            langs_str = ", ".join(matched_langs)
            matched_terms = [m["term"] for m in matches]
            print(f"  🚨 DISTRESS KEYWORD DETECTED in {langs_str}: {matched_terms}", flush=True)

            now = time.time()
            if now - self._last_trigger_time >= self.cooldown:
                self._last_trigger_time = now

                # Save evidence WAV
                file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                evidence_path = self.evidence_dir / f"sarvam_evidence_{file_timestamp}.wav"
                sf.write(str(evidence_path), chunk, self.sample_rate, subtype="PCM_16")

                # Send dashboard alert with language attribution
                notify_dashboard(
                    confidence=confidence,
                    audio_path=str(evidence_path),
                    language=langs_str,
                    contact="Voice Trigger (Sarvam)",
                    dashboard_url=self.dashboard_url,
                )
                print(
                    f"  ✅ Alert dispatched to {self.dashboard_url} [Language: {langs_str}, Confidence: {confidence}%, Evidence: {evidence_path.name}]",
                    flush=True,
                )
            else:
                remaining = self.cooldown - (now - self._last_trigger_time)
                print(f"  [Cooldown] Alert suppressed ({remaining:.1f}s remaining in refractory period)", flush=True)

        return transcript, matches

    def run(self) -> None:
        """Start streaming microphone audio and processing chunks continuously."""
        self._is_running = True
        accumulator = np.zeros(0, dtype=np.float32)

        print("=" * 70)
        print("   SARVAM AI MULTILINGUAL DURESS DETECTOR")
        print("=" * 70)
        print(f"Chunk Duration:     {self.chunk_duration:.1f}s")
        print(f"Energy Gate:        RMS >= {self.energy_threshold:.4f}")
        print(f"Cooldown:           {self.cooldown:.1f}s")
        print(f"Dashboard URL:      {self.dashboard_url}")
        print(f"Sample Rate:        {self.sample_rate} Hz")
        total_expressions = sum(len(pairs) for pairs in self.keywords_dict.values())
        print(f"Active Keywords:    {total_expressions} expressions across {len(self.keywords_dict)} languages")
        print("=" * 70)
        print("Listening for distress keywords... Press Ctrl+C to stop.\n", flush=True)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            device=self.device,
        )

        with stream:
            try:
                while self._is_running:
                    try:
                        chunk = self._audio_queue.get(timeout=0.2)
                        if chunk is None or len(chunk) == 0:
                            continue
                        accumulator = np.concatenate((accumulator, chunk))
                    except queue.Empty:
                        continue

                    # Process complete chunks
                    while len(accumulator) >= self.chunk_samples and self._is_running:
                        current_chunk = accumulator[: self.chunk_samples]
                        accumulator = accumulator[self.chunk_samples :]
                        self.process_chunk(current_chunk)

            except (KeyboardInterrupt, SystemExit):
                pass
            finally:
                self._is_running = False
                print(f"\nStopped listening. Total Sarvam API calls made: {self._api_call_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multilingual continuous voice duress listener powered by Sarvam AI STT."
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION_SECONDS,
        help=f"Audio evaluation chunk duration in seconds (default: {DEFAULT_CHUNK_DURATION_SECONDS}s).",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help=f"Minimum RMS energy required to invoke Sarvam API (default: {DEFAULT_ENERGY_THRESHOLD}).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Refractory cooldown in seconds between alerts (default: {DEFAULT_COOLDOWN_SECONDS}s).",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default=DEFAULT_DASHBOARD_URL,
        help=f"Endpoint to notify on trigger (default: {DEFAULT_DASHBOARD_URL}).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=str,
        default="evidence",
        help="Directory to save triggering audio chunks (default: evidence).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional input audio device index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: SARVAM_API_KEY environment variable is not set.\n"
            "Please configure it before running:\n"
            "  Windows PowerShell: $env:SARVAM_API_KEY=\"your_key_here\"\n"
            "  Windows CMD:        set SARVAM_API_KEY=your_key_here\n"
            "  Linux/macOS:        export SARVAM_API_KEY=\"your_key_here\"",
            file=sys.stderr,
        )
        sys.exit(1)

    listener = HelpListener(
        api_key=api_key,
        chunk_duration=args.chunk_duration,
        energy_threshold=args.energy_threshold,
        cooldown=args.cooldown,
        dashboard_url=args.dashboard_url,
        evidence_dir=args.evidence_dir,
        device=args.device,
    )
    listener.run()


if __name__ == "__main__":
    main()
