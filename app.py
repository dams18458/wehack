#!/usr/bin/env python3
"""
Silent Alert Dispatch Dashboard - Flask Backend.

Provides a local web dashboard to monitor and demo covert duress triggers.
- POST /trigger: Accepts trigger event JSON and records it in-memory.
- GET /api/alerts: Returns the event log and active state for 2-second polling.
- GET /: Renders the calm, minimal, pitch-ready monitor interface.
- POST /reset: Resets state to idle for rehearsal runs.
"""

import argparse
from datetime import datetime
from typing import Any, Dict, List
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Thread-safe in-memory store for trigger events (newest first)
ALERTS_HISTORY: List[Dict[str, Any]] = []


@app.route("/")
def index():
    """Render the dashboard UI."""
    return render_template("index.html")


@app.route("/trigger", methods=["POST"])
def trigger():
    """
    Accept an incoming duress trigger notification payload.
    Expected JSON schema:
      {
        "contact": str,
        "timestamp": str,
        "confidence": float,
        "audio_path": str
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON payload"}), 400

    contact = data.get("contact", "Emergency Proxy")
    timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0

    audio_path = data.get("audio_path", "")

    event = {
        "id": len(ALERTS_HISTORY) + 1,
        "contact": str(contact),
        "timestamp": str(timestamp),
        "confidence": confidence,
        "audio_path": str(audio_path),
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Prepend newest event to front of in-memory list
    ALERTS_HISTORY.insert(0, event)

    return jsonify({
        "status": "success",
        "message": "Alert recorded successfully",
        "alert": event,
    }), 201


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Return in-memory trigger log and active monitoring status."""
    status_label = "alert_sent" if ALERTS_HISTORY else "idle"
    return jsonify({
        "status": status_label,
        "count": len(ALERTS_HISTORY),
        "latest": ALERTS_HISTORY[0] if ALERTS_HISTORY else None,
        "alerts": ALERTS_HISTORY,
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Clear alert history to return the dashboard to idle for rehearsal."""
    ALERTS_HISTORY.clear()
    return jsonify({
        "status": "success",
        "message": "Alert log reset. Dashboard returned to idle state.",
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Run Silent Alert Dashboard.")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port to listen on (default: 5000).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("      SILENT ALERT DASHBOARD // LOCAL MONITOR")
    print("=" * 65)
    print(f"Dashboard URL: http://{args.host}:{args.port}/")
    print(f"Trigger API:   POST http://{args.host}:{args.port}/trigger")
    print("=" * 65)
    app.run(host=args.host, port=args.port, debug=False)