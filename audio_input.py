"""
audio_input.py
===============
Day 3, part 1: turn a song into the "stimulus" signal that will drive the
connectome's sensory (JO) neurons.

Extracts, per audio frame:
  - amplitude envelope (overall loudness -> how "hard" the sound hits)
  - onset/beat strength (percussive hits -> sharp stimulus pulses)
  - a few frequency bands (since JO neurons are tuned to vibration frequency,
    different bands can drive different JO subtypes for a richer effect)

Real usage:
    from audio_input import load_audio_features
    feats = load_audio_features("my_song.mp3")

For now (no song downloaded yet), `make_test_tone()` below generates a
synthetic click track so the rest of the pipeline (and demo-day silence
if the real file is ever missing) still has something to run against.
"""

import numpy as np
import librosa

FRAME_RATE = 30  # target "frames per second" for the visualization (~video fps)


def make_test_tone(duration_s=6.0, bpm=120, sr=22050, out_path=None):
    """
    Synthetic stand-in for a real song: a click every beat, with a rising
    tone on top, so amplitude AND frequency both vary over time -- enough
    to sanity-check the whole pipeline before a real song is available.
    """
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)

    # background tone: frequency slowly rises (200Hz -> 800Hz)
    freq = 200 + 600 * (t / duration_s)
    tone = 0.15 * np.sin(2 * np.pi * freq * t)

    # percussive click on every beat
    beat_period = 60.0 / bpm
    clicks = np.zeros_like(t)
    click_times = np.arange(0, duration_s, beat_period)
    for ct in click_times:
        idx = int(ct * sr)
        width = int(0.03 * sr)
        env = np.exp(-np.linspace(0, 8, width))
        end = min(idx + width, len(clicks))
        clicks[idx:end] += env[: end - idx]

    y = tone + clicks
    y = y / np.max(np.abs(y))

    if out_path:
        import soundfile as sf
        sf.write(out_path, y, sr)

    return y, sr


def load_audio_features(path=None, sr=22050):
    """
    Returns a dict of per-frame arrays, all the same length (n_frames):
      times       : timestamp (s) of each frame
      amplitude   : overall loudness envelope, normalized 0-1
      onset       : percussive onset/beat strength, normalized 0-1
      low_band    : energy in low frequencies (0-1)
      mid_band    : energy in mid frequencies (0-1)
      high_band   : energy in high frequencies (0-1)
      sr          : sample rate used
    """
    if path is None:
        y, sr = make_test_tone(sr=sr)
    else:
        y, sr = librosa.load(path, sr=sr, mono=True)

    hop_length = int(sr / FRAME_RATE)

    # overall amplitude envelope (RMS energy per frame)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # percussive onset strength (beat/hit detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 3-band frequency energy (low/mid/high), useful to drive different
    # JO subtypes differently since real JO neurons are frequency-tuned
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    low_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 2000)
    high_mask = freqs >= 2000

    low_band = S[low_mask].mean(axis=0)
    mid_band = S[mid_mask].mean(axis=0)
    high_band = S[high_mask].mean(axis=0)

    # align all features to the same (shortest) frame count
    n = min(len(rms), len(onset_env), len(low_band), len(mid_band), len(high_band))

    def norm(x):
        x = x[:n]
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    times = np.arange(n) * hop_length / sr

    return {
        "times": times,
        "amplitude": norm(rms),
        "onset": norm(onset_env),
        "low_band": norm(low_band),
        "mid_band": norm(mid_band),
        "high_band": norm(high_band),
        "sr": sr,
        "hop_length": hop_length,
    }


if __name__ == "__main__":
    feats = load_audio_features(None)  # synthetic test tone for now
    print(f"Frames: {len(feats['times'])}  |  Duration: {feats['times'][-1]:.2f}s")
    print(f"Amplitude range: {feats['amplitude'].min():.2f}-{feats['amplitude'].max():.2f}")
    print(f"Onset range: {feats['onset'].min():.2f}-{feats['onset'].max():.2f}")
