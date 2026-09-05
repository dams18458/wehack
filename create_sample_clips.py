"""
Generate demo synthetic audio clips and enrolled template for immediate offline testing.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from duress_detector.features import (
    DEFAULT_SAMPLE_RATE,
    create_enrolled_template,
    save_enrolled_template,
)

sr = DEFAULT_SAMPLE_RATE
clips_dir = Path("sample_clips")
clips_dir.mkdir(parents=True, exist_ok=True)

def synth_sound(pitch1=440, pitch2=660, dur=1.5, noise=0.005):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Double burst (simulating two-syllable phrase or two sharp exhales)
    env1 = np.exp(-((t - 0.4)**2) / 0.01)
    env2 = np.exp(-((t - 1.0)**2) / 0.01)
    w1 = np.sin(2 * np.pi * pitch1 * t) * env1
    w2 = np.sin(2 * np.pi * pitch2 * t) * env2
    n = np.random.normal(0, noise, len(t))
    return (0.5 * (w1 + w2) + n).astype(np.float32)

# 1. Generate enrollment samples and save demo template
s1 = synth_sound(dur=1.4)
s2 = synth_sound(dur=1.5, pitch1=445, pitch2=655)
s3 = synth_sound(dur=1.55, pitch1=435, pitch2=665)
template = create_enrolled_template([s1, s2, s3], sr=sr)
save_enrolled_template(template, "sample_clips/demo_enrolled_phrase.npy")

# 2. Generate positive test clips
sf.write("sample_clips/trigger_burst_clip1.wav", synth_sound(dur=1.48), sr)
sf.write("sample_clips/trigger_burst_clip2.wav", synth_sound(dur=1.52, pitch1=442, pitch2=658), sr)

# 3. Generate negative test clips
noise = np.random.normal(0, 0.05, int(sr * 2.0)).astype(np.float32)
sf.write("sample_clips/negative_room_noise.wav", noise, sr)

t_mono = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
beep = (0.4 * np.sin(2 * np.pi * 1800 * t_mono)).astype(np.float32)
sf.write("sample_clips/negative_high_beep.wav", beep, sr)

print("Demo clips created in sample_clips/")