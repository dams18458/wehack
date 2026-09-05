import os
import requests

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()

def transcribe_and_check(wav_path):
    with open(wav_path, "rb") as f:
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"model": "saaras:v3", "mode": "transcribe"},
        )
    result = response.json()
    print("Status:", response.status_code)
    print("Full response:", result)
    transcript = result.get("transcript", "")
    print("Transcript:", transcript)

    distress_words = ["help", "bachao", "madad", "save me", "help me"]
    found = [w for w in distress_words if w.lower() in transcript.lower()]
    if found:
        print(f"DISTRESS KEYWORD DETECTED: {found}")
    else:
        print("No distress keyword found.")
    return transcript, found

if __name__ == "__main__":
    import sys
    transcribe_and_check(sys.argv[1])
