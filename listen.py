#!/usr/bin/env python3
"""
Live Duress Detector CLI Runner.

Runs the live duress detection pipeline in complete background silence.
Captures audio via sounddevice, maintains a 30-second circular buffer,
evaluates DTW distance and energy envelope similarity, and fires trigger_action()
upon detection.

ZERO console output or audio feedback during normal listening.
Prints "ALERT TRIGGERED" and writes evidence only when a trigger occurs.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import requests

import duress_detector.detector as detector_module
from duress_detector.detector import (
    DEFAULT_BUFFER_CAPACITY_SECONDS,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_HOP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    LiveDetectionPipeline,
)
from duress_detector.features import load_enrolled_template
from duress_detector.matcher import (
    DTW_DISTANCE_THRESHOLD,
    ENERGY_SIMILARITY_THRESHOLD,
)
from duress_detector.trigger import trigger_action as base_trigger_action


def notify_dashboard(confidence: float, audio_path: str, contact: str = "Trusted Contact"):
    """Send the trigger event to the local dashboard for display."""
    try:
        requests.post(
            "http://127.0.0.1:5000/trigger",
            json={
                "contact": contact,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": confidence,
                "audio_path": audio_path,
            },
            timeout=2,
        )
    except requests.exceptions.RequestException:
        # Dashboard not running or unreachable — fail silently, 
        # don't let this crash the core detection pipeline
        pass


# Default DTW threshold used for confidence score normalization
_configured_dtw_threshold: float = DTW_DISTANCE_THRESHOLD


def trigger_action(
    buffer,
    evidence_dir="evidence",
    log_path="alerts.log",
    metadata=None,
):
    """
    Execute alert procedures upon duress detection and notify local dashboard.

    Preserves existing logging and evidence-saving behavior:
    1. Flushes rolling 30s audio buffer to evidence/<file>.wav
    2. Appends timestamped ALERT TRIGGERED entry to alerts.log
    3. Prints "ALERT TRIGGERED" to console
    4. Dispatches alert payload to http://127.0.0.1:5000/trigger
    """
    saved_file = base_trigger_action(
        buffer=buffer,
        evidence_dir=evidence_dir,
        log_path=log_path,
        metadata=metadata,
    )
    dtw_distance = float(metadata.get("dtw_dist", metadata.get("dtw_distance", 0.0))) if metadata else 0.0
    dtw_threshold = float(metadata.get("dtw_threshold", _configured_dtw_threshold)) if metadata else _configured_dtw_threshold
    confidence = max(0.0, min(1.0, 1 - (dtw_distance / dtw_threshold))) if dtw_threshold > 0 else 1.0
    confidence = round(confidence * 100.0, 1)
    notify_dashboard(confidence=confidence, audio_path=str(saved_file))
    return saved_file


# Hook into detector pipeline trigger-handling code path
detector_module.trigger_action = trigger_action



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live duress trigger detector in silent background mode."
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="enrolled_phrase.npy",
        help="Path to enrolled template .npy file (default: enrolled_phrase.npy).",
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
        help=f"Min required energy envelope correlation (default: {ENERGY_SIMILARITY_THRESHOLD}).",
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
        default=DEFAULT_HOP_SECONDS,
        help=f"Hop size between windows in seconds (default: {DEFAULT_HOP_SECONDS}s).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=str,
        default="evidence",
        help="Directory to dump 30-second evidence audio on trigger (default: evidence).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="alerts.log",
        help="Log file for trigger events (default: alerts.log).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Trigger cooldown in seconds (default: {DEFAULT_COOLDOWN_SECONDS}s).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional input audio device index.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print real-time DTW distance and energy correlation for each analysis window.",
    )
    return parser.parse_args()


def main():
    global _configured_dtw_threshold
    args = parse_args()
    _configured_dtw_threshold = args.dtw_threshold

    if args.verbose:
        orig_evaluate_match = detector_module.evaluate_match

        def verbose_evaluate_match(*m_args, **m_kwargs):
            res = orig_evaluate_match(*m_args, **m_kwargs)
            print(f"[DTW: {res.dtw_distance:.1f}, Energy: {res.energy_similarity:.2f}]", flush=True)
            return res

        detector_module.evaluate_match = verbose_evaluate_match

    template_path = Path(args.template)
    if not template_path.is_file():
        print(
            f"Error: Enrolled template not found at '{template_path}'.\n"
            "Please run 'python enroll.py' first to enroll your trigger phrase.",
            file=sys.stderr,
        )
        sys.exit(1)

    template = load_enrolled_template(template_path)

    # Initialize detection pipeline
    pipeline = LiveDetectionPipeline(
        template=template,
        window_seconds=args.window,
        hop_seconds=args.hop,
        buffer_capacity_seconds=DEFAULT_BUFFER_CAPACITY_SECONDS,
        dtw_threshold=args.dtw_threshold,
        energy_threshold=args.energy_threshold,
        cooldown_seconds=args.cooldown,
        evidence_dir=args.evidence_dir,
        log_path=args.log_path,
        device=args.device,
    )

    # Enter silent listening loop
    # In accordance with the silence requirement, no messages are printed during normal operation.
    try:
        pipeline.run_forever()
    except (KeyboardInterrupt, SystemExit):
        pipeline.stop()


if __name__ == "__main__":
    main()