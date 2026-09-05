#!/usr/bin/env python3
"""
Live Microphone Vocal Distress Classifier Demo.

Continuously captures microphone audio in a loop, analyzing rolling 2-second
windows, and printing the estimated vocal distress confidence score (0-100%)
to the console every 2 seconds.

Designed for live hackathon demos to demonstrate score transitions when speaking
calmly vs. projecting stress / vocal tension.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import sounddevice as sd

from distress_detector.classifier import DEFAULT_MODEL_PATH, DistressClassifier


def render_confidence_meter(score: float, width: int = 25) -> str:
    """Render an ASCII progress bar for the confidence score."""
    filled_len = int(round(width * (score / 100.0)))
    bar = "=" * filled_len + "-" * (width - filled_len)

    if score >= 70.0:
        badge = "ACUTE DISTRESS / PANIC"
    elif score >= 50.0:
        badge = "ELEVATED EMOTION / STRESS"
    elif score >= 30.0:
        badge = "MILD TENSION"
    else:
        badge = "CALM / NEUTRAL SPEECH"

    return f"[{bar}] {score:5.1f}%  | {badge}"


def generate_synthetic_demo_audio(sr: int, is_distressed: bool, dur: float = 2.0) -> np.ndarray:
    """Generate synthetic voice-like audio for dry-run testing."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    if is_distressed:
        # Higher F0 (~280 Hz), high jitter/tremor, breath turbulence
        f0 = 280.0 + 35.0 * np.sin(2 * np.pi * 6.0 * t)  # 6 Hz vocal tremor
        y = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * (2 * f0) * t)
        breath = np.random.normal(0, 0.08, len(t))
        return (y + breath).astype(np.float32)
    else:
        # Calm low F0 (~120 Hz), smooth harmonic, quiet
        f0 = 120.0
        y = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * (2 * f0) * t)
        return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Live microphone vocal distress detection demo.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model artifact.")
    parser.add_argument("--interval", type=float, default=2.0, help="Rolling window & update interval in seconds (default: 2.0s).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000).")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate live audio with synthetic calm vs. stressed speech for automated testing.")
    parser.add_argument("--max-iterations", type=int, default=0, help="Exit after N updates (default: 0 = loop until Ctrl+C).")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        print(f"Error: Model not found at '{model_path}'. Run 'python train.py' first.", file=sys.stderr)
        sys.exit(1)

    print("Loading vocal distress model...")
    classifier = DistressClassifier(model_path=model_path)

    print("=" * 75)
    print("      LIVE VOCAL DISTRESS DETECTION MONITOR")
    print("=" * 75)
    print(f"Model:           {model_path.name}")
    print(f"Window Interval: Every {args.interval:.1f} seconds")
    print(f"Sample Rate:     {args.sample_rate} Hz")
    if args.dry_run:
        print("Mode:            SIMULATED AUDIO (Dry-Run Mode)")
    else:
        print("Mode:            LIVE MICROPHONE STREAM (Speak calmly, then stressed!)")
    print("Press Ctrl+C to terminate.")
    print("=" * 75)
    print(f"{'Timestamp':<10} | {'Confidence Meter':<35} | Status")
    print("-" * 75)

    num_samples = int(round(args.interval * args.sample_rate))
    iteration = 0

    try:
        while True:
            iteration += 1
            now_str = datetime.now().strftime("%H:%M:%S")

            if args.dry_run:
                # Alternate between calm and stressed every 2 iterations
                simulate_distress = (iteration % 4 >= 2)
                audio_chunk = generate_synthetic_demo_audio(
                    sr=args.sample_rate,
                    is_distressed=simulate_distress,
                    dur=args.interval,
                )
                time.sleep(0.5)  # Fast pacing for automated verification
            else:
                # Live mic capture
                audio_chunk = sd.rec(
                    num_samples,
                    samplerate=args.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=args.device,
                )
                sd.wait()
                audio_chunk = audio_chunk.squeeze()

            # Predict rolling distress score
            score = classifier.predict_array(audio_chunk, sr=args.sample_rate)
            meter_str = render_confidence_meter(score)
            print(f"{now_str:<10} | {meter_str}")
            sys.stdout.flush()

            if args.max_iterations > 0 and iteration >= args.max_iterations:
                break

    except KeyboardInterrupt:
        print("\n\nMonitor terminated by user.")


if __name__ == "__main__":
    main()