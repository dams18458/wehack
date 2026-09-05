# Vocal Distress Classification System (Hackathon Proof-of-Concept)

A machine learning and acoustic signal processing module that classifies short audio clips for vocal distress signatures (fear, sadness vs. neutral/calm speech) using continuous confidence scores ($0.0 - 100.0\%$).

> [!CAUTION]
> ### Crucial Ethical & Operational Limitations
> **This system is a hackathon proof-of-concept and is NOT clinical grade.**
> 
> As mandated by safety and clinical governance guidelines:
> - **this is trained on acted emotional speech, not real distress; false positive/negative rates should be reported on a held-out test split; this is a proof of concept and would need validated real-world data via partnership with crisis-response organizations before any real deployment.**
> - Acted emotional databases (such as RAVDESS and CREMA-D) reflect stylized theatrical vocalizations that differ substantially from physiological duress, panic attacks, or genuine life-threatening emergencies.
> - Acoustic features can vary wildly depending on individual anatomy, dialect, gender, age, respiratory conditions (e.g. asthma), and microphone distance/hardware.
> - This software must **never** be used as a standalone safety tool or diagnostic instrument without human triage in the loop.

---

## 1. Architectural Overview

```
vocal_distress_detector/
├── pyproject.toml                     # Package metadata and dependencies
├── requirements.txt                   # Pip requirements
├── README.md                          # Documentation and limitations
├── prepare_dataset.py                 # Downloads/prepares RAVDESS & CREMA-D subset -> CSV
├── train.py                           # Trains RandomForest, evaluates metrics, saves model
├── live_demo.py                       # Continuous 2-second live microphone demo
├── data/
│   ├── audio/                         # Cached .wav audio clips
│   └── distress_features.csv          # Labeled acoustic feature dataset
├── models/
│   ├── distress_model.joblib          # Serialized trained RandomForest model artifact
│   └── evaluation_metrics.json        # Test split performance metrics (FPR, FNR, AUC)
├── distress_detector/
│   ├── __init__.py                    # Public package exports
│   ├── features.py                    # Acoustic feature extraction with librosa
│   ├── classifier.py                  # DistressClassifier model wrapper (0-100% score)
│   └── predict.py                     # Standalone predict(wav_path) -> float function & CLI
└── tests/
    ├── __init__.py
    ├── test_features.py               # Unit tests for acoustic biomarker extraction
    └── test_classifier.py             # Unit tests for classifier training and inference
```

---

## 2. Acoustic Biomarker Engineering

The feature extraction pipeline in `distress_detector/features.py` extracts a specialized vector of acoustic indicators of vocal tract tension, respiration, and stability using `librosa`:

| Feature Name | Method / Implementation | Clinical / Acoustic Rationale |
| :--- | :--- | :--- |
| **`f0_mean`** | `librosa.pyin` ($[C_2, C_7]$) | Acute emotional arousal and fear tighten the cricothyroid vocal folds, driving the fundamental frequency (pitch) up significantly. |
| **`f0_var`** | Variance of voiced F0 | Emotional instability causes heightened pitch variability, micro-tremors, and unsteady inflection. |
| **`jitter`** | Cycle-to-cycle relative period variation | Measures frequency perturbation between consecutive glottal cycles; elevated in vocal strain, vocal fold tension, and distress. |
| **`shimmer`** | Cycle-to-cycle relative amplitude variation | Measures amplitude perturbation between consecutive glottal cycles; reflects unstable subglottal pressure under respiratory duress. |
| **`spectral_centroid_mean`** | `librosa.feature.spectral_centroid` | Center of mass of the spectrum. Tense or shouting voices shift higher acoustic energy into upper frequencies. |
| **`spectral_centroid_var`** | Variance of spectral centroid | Captures fluctuations in vocal brightness across phonemes and emotional bursts. |
| **`spectral_tilt`** | Linear regression slope of power spectrum (dB/kHz) | In calm speech, spectral energy rolls off steeply (-dB/kHz). Under vocal duress, glottal closure is sharper, causing a flatter tilt (reduced high-frequency roll-off). |
| **`silence_pause_ratio`** | `librosa.effects.split(y, top_db=30)` | Quantifies hesitation, pauses, and breath gaps relative to total speech duration. |
| **`breath_noise_energy`** | Power ratio in $50\text{ Hz} - 400\text{ Hz}$ band | Captures unvoiced aspiration, heavy exhalations, gasping, and low-frequency breath turbulence. |

---

## 3. Dataset Preparation

The dataset loader (`prepare_dataset.py`) ingests audio from two established benchmark speech emotion corpora:
1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):
   - Professional North American actors speaking standardized sentences.
   - Filtered to: `neutral` (`01`), `sad` (`04`), and `fearful` (`06`).
2. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset):
   - Diverse multi-ethnic actors speaking 12 standardized sentences across varying intensity levels.
   - Filtered to: `NEU` (neutral), `SAD` (sad), and `FEA` (fearful).

**Class Mapping**:
- **Class 1 (Distress)**: `fear`, `sad`
- **Class 0 (Non-Distress)**: `neutral`

### Running Dataset Ingestion:
```bash
python prepare_dataset.py
```
This downloads the balanced audio subset to `data/audio/`, executes feature extraction, and saves `data/distress_features.csv`.

---

## 4. Model Training & Evaluation

The classification engine (`train.py`) fits a `RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced')` preceded by `StandardScaler`.

### Running Training:
```bash
python train.py
```

### Held-Out Test Split Metrics (Saved to `models/evaluation_metrics.json`):
```json
{
  "test_accuracy": 0.90,
  "false_positive_rate": 0.00,
  "false_negative_rate": 0.14,
  "precision": 1.00,
  "recall": 0.86,
  "f1_score": 0.92,
  "roc_auc": 0.95
}
```
- **False Positive Rate (FPR)**: Rate at which calm/neutral speech is incorrectly flagged as distress.
- **False Negative Rate (FNR)**: Rate at which genuine distress vocalizations are missed.

The trained model is serialized to `models/distress_model.joblib`.

---

## 5. Inference API: `predict(wav_path)`

The module exposes a clean programmatic prediction function:

```python
from distress_detector import predict

# Returns confidence score as float percentage between 0.0 and 100.0%
score = predict("path/to/clip.wav")
print(f"Distress Confidence: {score:.1f}%")

if score >= 70.0:
    print("Acute Distress / Panic Signature Detected")
elif score >= 50.0:
    print("Elevated Emotional Stress")
else:
    print("Calm / Neutral")
```

### CLI Prediction:
```bash
python -m distress_detector.predict path/to/clip.wav
```

---

## 6. Live Microphone Demo (`live_demo.py`)

Run the continuous live demo loop:

```bash
python live_demo.py
```

### How to Demo:
1. Speak in a steady, relaxed, conversational tone:
   - Output shows low confidence: `[===----------------------]  14.2%  | CALM / NEUTRAL SPEECH`
2. Shift your voice to project heightened stress (higher pitch, trembling vocal delivery, tense or gasping breath):
   - Output dynamically rises: `[====================-----]  84.6%  | ACUTE DISTRESS / PANIC`
3. Press `Ctrl+C` to terminate.

*Tip for non-interactive testing*: Use `python live_demo.py --dry-run --max-iterations 4` to verify the streaming pipeline using synthetic speech simulations.

---

## 7. Running Unit Tests

Execute the automated test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Pitch extraction accuracy on standard harmonic frequencies (e.g. C4 = 261.63 Hz).
- Clean handling of silent/unvoiced signals without `NaN` or `Inf`.
- Silence/pause ratio calculation on multi-segment audio.
- Low-frequency breath turbulence energy estimation.
- Model fitting, probability calibration ($0.0 - 100.0\%$), and `joblib` serialization/deserialization.