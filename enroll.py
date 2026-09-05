#!/usr/bin/env python3
"""
Duress-Trigger Enrollment Script.

Interactive CLI tool to capture 3-5 short audio samples of a user's chosen
trigger phrase or sound (e.g., 'code red override', or two sharp exhales),
extract MFCC features and RMS energy envelopes, compute an averaged template,
and persist it to 'enrolled_phrase.npy'.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import sounddevice as sd

from duress_detector.features import (
    DEFAULT_SAMPLE_RATE,
    create_enrolled_template,
    save_enrolled_template,
    trim_silence,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll a custom duress phrase or non-lexical sound."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        choices=[3, 4, 5],
        help="Number of enrollment samples to capture (3 to 5, default: 3).",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=2.5,
        help="Recording duration per sample in seconds (default: 2.5s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="enrolled_phrase.npy",
        help="Output path for the enrolled template (default: enrolled_phrase.npy).",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Microphone sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index.",
    )
    return parser.parse_args()


def record_sample(duration: float, sample_rate: int, device=None) -> np.ndarray:
    """Record a single audio clip from the microphone."""
    num_frames = int(round(duration * sample_rate))
    audio = sd.rec(
        num_frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.squeeze()


def main():
    args = parse_args()

    print("=" * 65)
    print("      DURESS-TRIGGER ENROLLMENT WIZARD")
    print("=" * 65)
    print("This script will record audio samples of your chosen trigger phrase")
    print("or non-lexical sound (e.g. 'activate safe mode' or two sharp exhales).")
    print(f"Target: {args.num_samples} samples | Duration per sample: {args.duration:.1f}s")
    print(f"Template will be saved to: {args.output}")
    print("=" * 65)

    samples = []
    for i in range(1, args.num_samples + 1):
        print(f"\n[Sample {i}/{args.num_samples}]")
        input("Press [Enter] when ready to speak/perform your trigger...")

        print("Get ready...", end="", flush=True)
        time.sleep(0.4)
        print(" 3...", end="", flush=True)
        time.sleep(0.4)
        print(" 2...", end="", flush=True)
        time.sleep(0.4)
        print(" 1...", end="", flush=True)
        time.sleep(0.4)
        print(" [RECORDING - Speak Now!]")

        raw_audio = record_sample(args.duration, args.sample_rate, device=args.device)
        print("Recording captured.")

        max_amp = float(np.max(np.abs(raw_audio))) if len(raw_audio) > 0 else 0.0
        if max_amp < 0.02:
            print("  Warning: Signal was very quiet. Please ensure your microphone is active and speak clearly.")

        trimmed, (start_idx, end_idx) = trim_silence(raw_audio, top_db=25.0)
        trimmed_dur = len(trimmed) / args.sample_rate
        print(f"  Processed duration after silence trim: {trimmed_dur:.2f}s (peak amplitude: {max_amp:.3f})")

        samples.append(trimmed if len(trimmed) > int(args.sample_rate * 0.2) else raw_audio)

    print("\nComputing time-aligned template and feature vectors...")
    try:
        template = create_enrolled_template(
            samples=samples,
            sr=args.sample_rate,
        )
    except Exception as e:
        print(f"Error creating template: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = save_enrolled_template(template, filepath=args.output)

    print("\n" + "=" * 65)
    print("ENROLLMENT SUCCESSFUL!")
    print("=" * 65)
    print(f"Template File:       {output_path}")
    print(f"Samples Enrolled:    {template['num_samples_enrolled']}")
    print(f"Mean Phrase Length:  {template['duration']:.2f} seconds")
    print(f"MFCC Template Shape: {template['mfcc'].shape} (frames, coefficients)")
    print(f"Energy Envelope:     {template['energy_envelope'].shape} frames")
    print("\nNext steps:")
    print("  1. Test offline against pre-recorded clips:  py test_trigger.py --folder <dir>")
    print("  2. Run silent live background detection:    py listen.py")
    print("=" * 65)


if __name__ == "__main__":
    main()