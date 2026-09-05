#!/usr/bin/env python3
"""
CLI Helper to send a test trigger event to the local Silent Alert Dashboard.
"""

import argparse
import json
import urllib.request
from datetime import datetime


def send_trigger(endpoint: str, contact: str, confidence: float, audio_path: str):
    payload = {
        "contact": contact,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": confidence,
        "audio_path": audio_path,
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print("Response:", res_data)
            print(f"[SUCCESS] Sent alert to {endpoint} (Contact: {contact}, Confidence: {confidence}%)")
    except Exception as e:
        print(f"[ERROR] Failed to send trigger: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send test trigger to Silent Alert Dashboard.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5000/trigger", help="Target trigger URL.")
    parser.add_argument("--contact", type=str, default="Sarah Chen (Emergency Contact)", help="Mock contact name.")
    parser.add_argument("--confidence", type=float, default=89.2, help="Confidence score (0-100).")
    parser.add_argument("--audio", type=str, default="evidence/duress_evidence_sample.wav", help="Audio path reference.")
    args = parser.parse_args()

    send_trigger(args.url, args.contact, args.confidence, args.audio)