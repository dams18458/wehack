# Voice for the Voiceless

**Translating silence, not just language.**

Voice AI has spent years solving accent bias, dialect coverage, and language translation — 
all assuming the speaker is free to speak. This project addresses the case nobody's building 
for: the person who *can't* speak freely, safely, or at all.

## The problem

- A domestic abuse survivor who can't speak freely with someone in the next room
- A stroke or laryngectomy patient who has lost speech but still has intent
- Someone in a family or workplace hierarchy where saying "I'm not okay" isn't safe
- An emergency call that gets cut short — a whispered word, a breath before the line dies

The real accent bias isn't regional. It's that voice technology only listens to people who 
are allowed to speak.

## Architecture

Three independent detection layers, converging on one shared alert dashboard:
                ┌─────────────────┐
                │  Microphone      │
                └────────┬─────────┘
          ┌──────────────┼──────────────┐
          ▼               ▼               ▼
 ┌────────────────┐┌────────────────┐┌──────────────────┐
 │ Trigger layer  ││ Distress layer ││ Multilingual      │
 │ MFCC + DTW      ││ Vocal biomarkers││ fallback (Sarvam) │
 └────────┬────────┘└────────┬────────┘└─────────┬─────────┘
          └──────────────┬───┴───────────────────┘
                          ▼
                ┌──────────────────┐
                │ Alert dashboard   │
                │ (Flask REST API)  │
                └────────┬──────────┘
                          ▼
                ┌──────────────────┐
                │ Trusted contact   │
                │     alerted       │
                └──────────────────┘
              
Each layer is architecturally independent — failure or absence of one does not affect 
the others. All three send events to the same dashboard endpoint.

### 1. Trigger layer — `duress_detection/`
A user pre-enrolls a chosen phrase or non-lexical sound. The system extracts MFCC 
(Mel-Frequency Cepstral Coefficient) features and matches live audio against the enrolled 
template using Dynamic Time Warping (DTW) distance and energy-envelope correlation as a 
dual-gate check. Runs fully on-device — no network dependency.

**Stack:** Python, librosa, FastDTW, sounddevice

**Run it:**
```bash
cd duress_detection
py enroll.py -n 5 -d 2.5        # enroll your phrase
py listen.py                     # start silent live detection
```

### 2. Distress classifier layer — `duress_detection/` (distress detector module)
Extracts acoustic biomarkers (pitch variance, jitter/shimmer, spectral tilt, silence ratio, 
breath-band energy) and scores distress confidence (0–100%) using a RandomForestClassifier 
trained on the RAVDESS and CREMA-D acted-emotion speech corpora. Does not require 
pre-enrollment.

**Stack:** Python, librosa, scikit-learn

### 3. Multilingual fallback layer — `duress_detection/help_listener.py`
For unregistered users or first-contact emergencies. Captures speech, transcribes it via 
Sarvam AI's Saaras v3 speech-to-text model, and checks the transcript against a distress 
keyword list covering multiple Indian languages in both native script and transliteration.

**Stack:** Python, Sarvam AI Speech-to-Text API (Saaras v3)

**Run it:**
```bash
cd duress_detection
$env:SARVAM_API_KEY = "your-key-here"
py help_listener.py
```

### Alert dashboard — `silent_alert_dashboard/`
A local Flask app that receives events from all three layers and displays a calm, 
non-alarming "Alert Sent" screen with the triggering contact, timestamp, confidence score, 
and evidence reference.

**Stack:** Python, Flask

**Run it:**
```bash
cd silent_alert_dashboard
py app.py
# open http://127.0.0.1:5000/
```

## Ethics and consent

- Nothing listens without explicit opt-in enrollment by the user themselves.
- The trigger layer runs fully on-device — no ambient audio is transcribed, stored, or 
  transmitted; it only ever compares against the user's own pre-enrolled pattern, similar 
  in principle to how a smoke detector reacts only to smoke.
- A short grace period separates detection from dispatch, allowing an accidental match to 
  be stood down before a contact is notified.

## Known limitations (proof-of-concept stage)

- The distress classifier is trained on **acted** emotional speech, not real distress — 
  real-world deployment would need validated data via partnership with crisis-response 
  organizations.
- Detection range is limited by standard microphone hardware (roughly under a meter for 
  whispers) — a production version would run on a wearable held close to the body.
- Acoustic matching (trigger layer) has natural variance between a user's own utterances; 
  a neural keyword-spotting model would improve robustness over the current DTW approach.

## Roadmap

- Wearable-paired activation for range-independent detection
- Partnership with crisis-response organizations for validated real-world training data
- Extending the distress layer toward aphasia/laryngectomy intent reconstruction
- Neural keyword-spotting model to replace DTW-based acoustic matching

## Track

Built for the Sarvam AI Challenge — reframing "Voice for the Voiceless," "Listening 
Machines," "Beyond Translation," "Vocal Fingerprints," and "Accent Bias" as a single, 
unified problem: articulation under constraint.
