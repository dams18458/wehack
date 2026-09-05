#!/usr/bin/env python3
"""
Offline Trigger Evaluation Tool.

Evaluates the duress detection pipeline against pre-recorded .wav files
without requiring live microphone input.

Supports benchmark categorization of:
  - True Positives: recordings containing the actual trigger phrase/sound.
  - False Positives: recordings containing non-trigger speech, similar phrases, or background noise.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import librosa
import numpy as np
import soundfile as sf

from duress_detector.detector import (
    DEFAULT_HOP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    process_audio_windows,
)
from duress_detector.features import load_enrolled_template
from duress_detector.matcher import (
    DTW_DISTANCE_THRESHOLD,
    ENERGY_SIMILARITY_THRESHOLD,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run duress detection pipeline offline against a folder of .wav files."
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Directory containing test .wav audio files.",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="enrolled_phrase.npy",
        help="Path to enrolled template .npy (default: enrolled_phrase.npy).",
    )
    parser.add_argument(
        "--dtw-threshold",
        type=float,
        default=DTW_DISTANCE_THRESHOLD,
        help=f"Max allowed normalized DTW distance (default: {DTW_DISTANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=ENERGY_SIMILARITY_THRESHOLD,
        help=f"Min required energy correlation (default: {ENERGY_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Sliding window size in seconds (default: {DEFAULT_WINDOW_SECONDS}s).",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=0.25,
        help="Hop size in seconds between evaluation windows (default: 0.25s for high test resolution).",
    )
    return parser.parse_args()


def infer_expected_label(path: Path) -> Optional[bool]:
    """
    Infer ground truth label from file or directory name conventions:
    Returns True for positive/trigger files, False for negative/noise files,
    or None if ambiguous.
    """
    stem_lower = path.stem.lower()
    parent_lower = path.parent.name.lower()

    pos_keywords = ["trigger", "positive", "target", "true", "pos"]
    neg_keywords = ["negative", "noise", "distractor", "false", "neg", "other", "speech"]

    for kw in pos_keywords:
        if kw in stem_lower or kw == parent_lower:
            return True

    for kw in neg_keywords:
        if kw in stem_lower or kw == parent_lower:
            return False

    return None


def evaluate_wav_file(
    file_path: Path,
    template: dict,
    window_sec: float,
    hop_sec: float,
    dtw_thresh: float,
    energy_thresh: float,
) -> Tuple[bool, float, float, float, int]:
    """
    Process a single .wav file through the window evaluation engine.
    
    Returns:
        (triggered, min_dtw, max_energy, duration_sec, num_windows)
    """
    audio_data, file_sr = sf.read(str(file_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)
    audio_data = audio_data.astype(np.float32)

    target_sr = int(template.get("sample_rate", 16000))
    if file_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=file_sr, target_sr=target_sr)

    duration_sec = len(audio_data) / target_sr

    window_results = process_audio_windows(
        audio=audio_data,
        sr=target_sr,
        template=template,
        window_seconds=window_sec,
        hop_seconds=hop_sec,
        dtw_threshold=dtw_thresh,
        energy_threshold=energy_thresh,
    )

    if not window_results:
        return False, float("inf"), 0.0, duration_sec, 0

    min_dtw = min(res.dtw_distance for _, res in window_results)
    max_energy = max(res.energy_similarity for _, res in window_results)
    triggered = any(res.is_match for _, res in window_results)

    return triggered, min_dtw, max_energy, duration_sec, len(window_results)


def main():
    args = parse_args()

    test_folder = Path(args.folder)
    if not test_folder.is_dir():
        print(f"Error: Specified folder '{test_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)

    template_path = Path(args.template)
    if not template_path.is_file():
        print(f"Error: Template file '{template_path}' not found.", file=sys.stderr)
        sys.exit(1)

    template = load_enrolled_template(template_path)

    wav_files = sorted(list(test_folder.glob("*.wav")) + list(test_folder.glob("**/*.wav")))
    # Deduplicate paths
    wav_files = sorted(list(set(wav_files)))

    if not wav_files:
        print(f"No .wav files found in '{test_folder}'.", file=sys.stderr)
        sys.exit(1)

    print("=" * 85)
    print(f"   DURESS TRIGGER OFFLINE EVALUATION BENCHMARK")
    print("=" * 85)
    print(f"Test Directory:      {test_folder.resolve()}")
    print(f"Enrolled Template:   {template_path.name} (enrolled duration: {template['duration']:.2f}s)")
    print(f"DTW Distance Limit:  <= {args.dtw_threshold:.1f} (lower is stricter)")
    print(f"Energy Corr Limit:   >= {args.energy_threshold:.2f} (higher is stricter)")
    print(f"Window: {args.window:.2f}s | Hop: {args.hop:.2f}s | Total Test Files: {len(wav_files)}")
    print("=" * 85)

    header = f"{'Filename':<32} {'Dur':<6} {'Min DTW':<10} {'Max Env':<10} {'Triggered':<11} {'Status'}"
    print(header)
    print("-" * 85)

    tp, fp, tn, fn = 0, 0, 0, 0
    unlabeled = 0

    for wav_file in wav_files:
        rel_name = wav_file.name
        if len(rel_name) > 30:
            rel_name = rel_name[:27] + "..."

        triggered, min_dtw, max_energy, dur, _ = evaluate_wav_file(
            file_path=wav_file,
            template=template,
            window_sec=args.window,
            hop_sec=args.hop,
            dtw_thresh=args.dtw_threshold,
            energy_thresh=args.energy_threshold,
        )

        expected = infer_expected_label(wav_file)
        if expected is True:
            if triggered:
                tp += 1
                status = "[PASS] TRUE POSITIVE"
            else:
                fn += 1
                status = "[FAIL] FALSE NEGATIVE (Missed)"
        elif expected is False:
            if triggered:
                fp += 1
                status = "[FAIL] FALSE POSITIVE"
            else:
                tn += 1
                status = "[PASS] TRUE NEGATIVE"
        else:
            unlabeled += 1
            status = "[ALERT]" if triggered else "[SILENT]"

        trig_str = "YES" if triggered else "NO"
        print(f"{rel_name:<32} {dur:<6.2f} {min_dtw:<10.2f} {max_energy:<10.2f} {trig_str:<11} {status}")

    print("-" * 85)
    labeled_total = tp + fp + tn + fn
    if labeled_total > 0:
        accuracy = (tp + tn) / labeled_total * 100.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"Evaluation Summary:")
        print(f"  True Positives (Correct triggers):      {tp}")
        print(f"  True Negatives (Correctly ignored):     {tn}")
        print(f"  False Positives (Spurious triggers):    {fp}")
        print(f"  False Negatives (Missed triggers):      {fn}")
        print(f"  Accuracy:  {accuracy:.1f}%")
        print(f"  Precision: {precision:.1f}%")
        print(f"  Recall:    {recall:.1f}%")
        print(f"  F1-Score:  {f1:.1f}%")
    else:
        print(f"Processed {unlabeled} files (no ground-truth naming detected).")

    print("=" * 85)
    if fp > 0 or fn > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()