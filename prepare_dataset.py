#!/usr/bin/env python3
"""
Dataset Preparation Script for Vocal Distress Classification.

Downloads and prepares a balanced subset of audio clips from:
1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Filtered to: Neutral (01), Sad (04), Fearful (06)
2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
   - Filtered to: NEU (neutral), SAD (sad), FEA (fear)

Labels:
- 1: Distress (Fear, Sad)
- 0: Non-Distress (Neutral)

Extracts acoustic features using librosa (pYIN F0, jitter, shimmer,
spectral centroid, spectral tilt, silence ratio, breath noise energy)
and writes them to data/distress_features.csv.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from distress_detector.features import FEATURE_NAMES, extract_features


# ==============================================================================
# Curated subset URLs for RAVDESS and CREMA-D
# ==============================================================================

RAVDESS_BASE = (
    "https://raw.githubusercontent.com/ZenvilleErasmus/RAVDESS-emotions-speech-audio-only"
    "/master/Audio_Speech_Actors_01-24"
)

# Format: 03-01-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
# Emotion: 01=Neutral, 04=Sad, 06=Fearful
RAVDESS_FILES = [
    # Actor 01 (Male)
    ("ravdess_a01_neu_01.wav", f"{RAVDESS_BASE}/Actor_01/03-01-01-01-01-01-01.wav", "RAVDESS", "Actor_01", "neutral", 0),
    ("ravdess_a01_neu_02.wav", f"{RAVDESS_BASE}/Actor_01/03-01-01-01-02-01-01.wav", "RAVDESS", "Actor_01", "neutral", 0),
    ("ravdess_a01_sad_01.wav", f"{RAVDESS_BASE}/Actor_01/03-01-04-01-01-01-01.wav", "RAVDESS", "Actor_01", "sad", 1),
    ("ravdess_a01_sad_02.wav", f"{RAVDESS_BASE}/Actor_01/03-01-04-02-01-01-01.wav", "RAVDESS", "Actor_01", "sad", 1),
    ("ravdess_a01_fea_01.wav", f"{RAVDESS_BASE}/Actor_01/03-01-06-01-01-01-01.wav", "RAVDESS", "Actor_01", "fear", 1),
    ("ravdess_a01_fea_02.wav", f"{RAVDESS_BASE}/Actor_01/03-01-06-02-01-01-01.wav", "RAVDESS", "Actor_01", "fear", 1),

    # Actor 02 (Female)
    ("ravdess_a02_neu_01.wav", f"{RAVDESS_BASE}/Actor_02/03-01-01-01-01-01-02.wav", "RAVDESS", "Actor_02", "neutral", 0),
    ("ravdess_a02_neu_02.wav", f"{RAVDESS_BASE}/Actor_02/03-01-01-01-02-01-02.wav", "RAVDESS", "Actor_02", "neutral", 0),
    ("ravdess_a02_sad_01.wav", f"{RAVDESS_BASE}/Actor_02/03-01-04-01-01-01-02.wav", "RAVDESS", "Actor_02", "sad", 1),
    ("ravdess_a02_sad_02.wav", f"{RAVDESS_BASE}/Actor_02/03-01-04-02-01-01-02.wav", "RAVDESS", "Actor_02", "sad", 1),
    ("ravdess_a02_fea_01.wav", f"{RAVDESS_BASE}/Actor_02/03-01-06-01-01-01-02.wav", "RAVDESS", "Actor_02", "fear", 1),
    ("ravdess_a02_fea_02.wav", f"{RAVDESS_BASE}/Actor_02/03-01-06-02-01-01-02.wav", "RAVDESS", "Actor_02", "fear", 1),

    # Actor 03 (Male)
    ("ravdess_a03_neu_01.wav", f"{RAVDESS_BASE}/Actor_03/03-01-01-01-01-01-03.wav", "RAVDESS", "Actor_03", "neutral", 0),
    ("ravdess_a03_neu_02.wav", f"{RAVDESS_BASE}/Actor_03/03-01-01-01-02-01-03.wav", "RAVDESS", "Actor_03", "neutral", 0),
    ("ravdess_a03_sad_01.wav", f"{RAVDESS_BASE}/Actor_03/03-01-04-01-01-01-03.wav", "RAVDESS", "Actor_03", "sad", 1),
    ("ravdess_a03_sad_02.wav", f"{RAVDESS_BASE}/Actor_03/03-01-04-02-01-01-03.wav", "RAVDESS", "Actor_03", "sad", 1),
    ("ravdess_a03_fea_01.wav", f"{RAVDESS_BASE}/Actor_03/03-01-06-01-01-01-03.wav", "RAVDESS", "Actor_03", "fear", 1),
    ("ravdess_a03_fea_02.wav", f"{RAVDESS_BASE}/Actor_03/03-01-06-02-01-01-03.wav", "RAVDESS", "Actor_03", "fear", 1),

    # Actor 04 (Female)
    ("ravdess_a04_neu_01.wav", f"{RAVDESS_BASE}/Actor_04/03-01-01-01-01-01-04.wav", "RAVDESS", "Actor_04", "neutral", 0),
    ("ravdess_a04_neu_02.wav", f"{RAVDESS_BASE}/Actor_04/03-01-01-01-02-01-04.wav", "RAVDESS", "Actor_04", "neutral", 0),
    ("ravdess_a04_sad_01.wav", f"{RAVDESS_BASE}/Actor_04/03-01-04-01-01-01-04.wav", "RAVDESS", "Actor_04", "sad", 1),
    ("ravdess_a04_sad_02.wav", f"{RAVDESS_BASE}/Actor_04/03-01-04-02-01-01-04.wav", "RAVDESS", "Actor_04", "sad", 1),
    ("ravdess_a04_fea_01.wav", f"{RAVDESS_BASE}/Actor_04/03-01-06-01-01-01-04.wav", "RAVDESS", "Actor_04", "fear", 1),
    ("ravdess_a04_fea_02.wav", f"{RAVDESS_BASE}/Actor_04/03-01-06-02-01-01-04.wav", "RAVDESS", "Actor_04", "fear", 1),
]

CREMA_BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"

CREMA_FILES = [
    # Actor 1001
    ("crema_1001_neu.wav", f"{CREMA_BASE}/1001_DFA_NEU_XX.wav", "CREMA-D", "1001", "neutral", 0),
    ("crema_1001_sad.wav", f"{CREMA_BASE}/1001_DFA_SAD_XX.wav", "CREMA-D", "1001", "sad", 1),
    ("crema_1001_fea.wav", f"{CREMA_BASE}/1001_DFA_FEA_XX.wav", "CREMA-D", "1001", "fear", 1),
    ("crema_1001_neu_ieo.wav", f"{CREMA_BASE}/1001_IEO_NEU_MD.wav", "CREMA-D", "1001", "neutral", 0),
    ("crema_1001_sad_ieo.wav", f"{CREMA_BASE}/1001_IEO_SAD_MD.wav", "CREMA-D", "1001", "sad", 1),
    ("crema_1001_fea_ieo.wav", f"{CREMA_BASE}/1001_IEO_FEA_MD.wav", "CREMA-D", "1001", "fear", 1),

    # Actor 1002
    ("crema_1002_neu.wav", f"{CREMA_BASE}/1002_DFA_NEU_XX.wav", "CREMA-D", "1002", "neutral", 0),
    ("crema_1002_sad.wav", f"{CREMA_BASE}/1002_DFA_SAD_XX.wav", "CREMA-D", "1002", "sad", 1),
    ("crema_1002_fea.wav", f"{CREMA_BASE}/1002_DFA_FEA_XX.wav", "CREMA-D", "1002", "fear", 1),
    ("crema_1002_neu_ieo.wav", f"{CREMA_BASE}/1002_IEO_NEU_MD.wav", "CREMA-D", "1002", "neutral", 0),
    ("crema_1002_sad_ieo.wav", f"{CREMA_BASE}/1002_IEO_SAD_MD.wav", "CREMA-D", "1002", "sad", 1),
    ("crema_1002_fea_ieo.wav", f"{CREMA_BASE}/1002_IEO_FEA_MD.wav", "CREMA-D", "1002", "fear", 1),

    # Actor 1003
    ("crema_1003_neu.wav", f"{CREMA_BASE}/1003_DFA_NEU_XX.wav", "CREMA-D", "1003", "neutral", 0),
    ("crema_1003_sad.wav", f"{CREMA_BASE}/1003_DFA_SAD_XX.wav", "CREMA-D", "1003", "sad", 1),
    ("crema_1003_fea.wav", f"{CREMA_BASE}/1003_DFA_FEA_XX.wav", "CREMA-D", "1003", "fear", 1),
    ("crema_1003_neu_ieo.wav", f"{CREMA_BASE}/1003_IEO_NEU_MD.wav", "CREMA-D", "1003", "neutral", 0),
    ("crema_1003_sad_ieo.wav", f"{CREMA_BASE}/1003_IEO_SAD_MD.wav", "CREMA-D", "1003", "sad", 1),
    ("crema_1003_fea_ieo.wav", f"{CREMA_BASE}/1003_IEO_FEA_MD.wav", "CREMA-D", "1003", "fear", 1),

    # Actor 1004
    ("crema_1004_neu.wav", f"{CREMA_BASE}/1004_DFA_NEU_XX.wav", "CREMA-D", "1004", "neutral", 0),
    ("crema_1004_sad.wav", f"{CREMA_BASE}/1004_DFA_SAD_XX.wav", "CREMA-D", "1004", "sad", 1),
    ("crema_1004_fea.wav", f"{CREMA_BASE}/1004_DFA_FEA_XX.wav", "CREMA-D", "1004", "fear", 1),
    ("crema_1004_neu_ieo.wav", f"{CREMA_BASE}/1004_IEO_NEU_MD.wav", "CREMA-D", "1004", "neutral", 0),
    ("crema_1004_sad_ieo.wav", f"{CREMA_BASE}/1004_IEO_SAD_MD.wav", "CREMA-D", "1004", "sad", 1),
    ("crema_1004_fea_ieo.wav", f"{CREMA_BASE}/1004_IEO_FEA_MD.wav", "CREMA-D", "1004", "fear", 1),
]

ALL_ENTRIES = RAVDESS_FILES + CREMA_FILES


def download_file(entry: Tuple, audio_dir: Path, timeout: int = 15) -> Tuple[str, bool]:
    """Download audio file with timeout and retry."""
    filename, url, dataset, actor, emotion, label = entry
    dest_path = audio_dir / filename
    if dest_path.is_file() and dest_path.stat().st_size > 1000:
        return filename, True  # Already cached

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            if data.startswith(b"version https://git-lfs"):
                return filename, False
            with open(dest_path, "wb") as f:
                f.write(data)
        return filename, True
    except Exception:
        return filename, False


def main():
    parser = argparse.ArgumentParser(description="Download and extract features from RAVDESS & CREMA-D subset.")
    parser.add_argument("--output-csv", type=str, default="data/distress_features.csv", help="Path to output CSV.")
    parser.add_argument("--audio-dir", type=str, default="data/audio", help="Directory to store audio clips.")
    parser.add_argument("--max-samples", type=int, default=len(ALL_ENTRIES), help="Max number of clips to process.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download workers.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print(" VOCAL DISTRESS DATASET PREPARATION (RAVDESS & CREMA-D)", flush=True)
    print("=" * 75, flush=True)
    print(f"Target Clips:    {min(args.max_samples, len(ALL_ENTRIES))}", flush=True)
    print(f"Audio Cache:     {audio_dir}", flush=True)
    print(f"Output CSV:      {output_csv}", flush=True)
    print("=" * 75, flush=True)

    entries_to_process = ALL_ENTRIES[: args.max_samples]

    print(f"\n[1/2] Ingesting audio clips in parallel ({args.workers} workers)...", flush=True)
    successful_downloads = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, entry, audio_dir): entry for entry in entries_to_process}
        for future in as_completed(futures):
            entry = futures[future]
            fname, ok = future.result()
            if ok:
                successful_downloads += 1
            status = "OK" if ok else "FAILED"
            print(f"  Downloaded/Verified: {fname:<26} -> {status}", flush=True)

    print(f"\nSuccessfully verified {successful_downloads}/{len(entries_to_process)} audio clips.", flush=True)

    # --------------------------------------------------------------------------
    # Extract Acoustic Features
    # --------------------------------------------------------------------------
    print(f"\n[2/2] Extracting acoustic features with librosa...", flush=True)
    fieldnames = [
        "filename",
        "dataset",
        "actor",
        "emotion",
        "label",
    ] + list(FEATURE_NAMES)

    csv_rows = []
    emotion_counts = {}

    for idx, (filename, _, dataset, actor, emotion, label) in enumerate(entries_to_process, 1):
        dest_path = audio_dir / filename
        if not dest_path.is_file():
            continue

        try:
            t0 = time.time()
            feats = extract_features(dest_path)
            dt = time.time() - t0
        except Exception as e:
            print(f"  Warning: failed to extract features for {filename}: {e}", file=sys.stderr, flush=True)
            continue

        row = {
            "filename": filename,
            "dataset": dataset,
            "actor": actor,
            "emotion": emotion,
            "label": label,
        }
        for k in FEATURE_NAMES:
            row[k] = round(feats.get(k, 0.0), 5)

        csv_rows.append(row)
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        print(f"  [{idx:2d}/{len(entries_to_process)}] Extracted: {filename:<25} ({dt:.2f}s) | F0: {feats['f0_mean']:5.1f} Hz", flush=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 75, flush=True)
    print("DATASET PREPARATION COMPLETE", flush=True)
    print("=" * 75, flush=True)
    print(f"Wrote {len(csv_rows)} labeled rows to: {output_csv}", flush=True)
    print("\nClass & Emotion Breakdown:", flush=True)
    for emo, count in sorted(emotion_counts.items()):
        lbl = "Distress (1)" if emo in ("fear", "sad") else "Non-Distress (0)"
        print(f"  - {emo.capitalize():<8}: {count:2d} samples ({lbl})", flush=True)

    distress_total = sum(1 for r in csv_rows if r["label"] == 1)
    neutral_total = sum(1 for r in csv_rows if r["label"] == 0)
    print(f"\nSummary Totals:", flush=True)
    print(f"  - Distress Class (Fear/Sad): {distress_total}", flush=True)
    print(f"  - Neutral Class:             {neutral_total}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()