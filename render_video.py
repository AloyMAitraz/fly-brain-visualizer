"""
render_video.py
================
Day 5: turn the single-frame renderer into an actual animated video,
synced with the song.

Approach:
  1. Render one PNG per frame across a chosen [start_s, end_s] window
     (reusing render_frame() from render_activity_frame.py -- nothing
     new to debug there, just calling it in a loop).
  2. Stitch the PNGs into a silent .mp4 with ffmpeg at the same fps the
     activity was simulated at (so motion timing is correct).
  3. Mux in the actual audio clip (trimmed to the same window) so the
     final video has real sound synced to the real visuals.

Needs ffmpeg installed and on PATH (same requirement noted back when we
first loaded the song -- if that worked, this will too).
"""

import os
import shutil
import subprocess
import numpy as np

from connectome import get_circuit
from audio_input import load_audio_features, FRAME_RATE
from activity_model import simulate, get_window
from render_dots import compute_neuron_points, compute_brain_outline, render_dot_frame

# ---- CONFIG ----
USE_MOCK_CIRCUIT = True    # False = fetch the real circuit (needs NEUPRINT_TOKEN)
SONG_PATH = None            # e.g. "my_song.mp3"; None = built-in synthetic click track
START_S = 0.0           # window to render (seconds)
END_S = 6.0            # the synthetic test tone is 6s long
OUT_NAME = "brain_clip"
# -----------------


def render_video(nodes, skeletons, sim_window, out_dir, song_path=None):
    frames_dir = os.path.join(out_dir, "frames_tmp")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    body_ids = sim_window["body_ids"]
    times = sim_window["times"]
    activity = sim_window["activity"]
    n_frames = len(times)

    # dot-mode setup: one representative point per neuron + a brain
    # outline computed once (not per-frame -- the shape doesn't change,
    # only which points light up does)
    points_df = compute_neuron_points(skeletons)
    hull = compute_brain_outline(skeletons)
    hull_points = hull.points if hull is not None else None

    print(f"Rendering {n_frames} frames (dot mode)...")
    for i in range(n_frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        render_dot_frame(
            nodes, points_df, body_ids, activity[i], frame_path,
            title=f"t={times[i]:.2f}s", hull=hull, hull_points=hull_points,
        )
        if i % 30 == 0:
            print(f"  {i}/{n_frames}")

    silent_video = os.path.join(out_dir, f"{OUT_NAME}_silent.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FRAME_RATE),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        silent_video,
    ], check=True)
    print(f"Silent video: {silent_video}")

    final_video = os.path.join(out_dir, f"{OUT_NAME}.mp4")
    if song_path and os.path.exists(song_path):
        subprocess.run([
            "ffmpeg", "-y",
            "-i", silent_video,
            "-ss", str(times[0]), "-t", str(times[-1] - times[0]),
            "-i", song_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            final_video,
        ], check=True)
        print(f"Final video with audio: {final_video}")
    else:
        shutil.copy(silent_video, final_video)
        print(f"Final video (no audio available): {final_video}")

    shutil.rmtree(frames_dir)
    return final_video


if __name__ == "__main__":
    nodes, conn, skeletons, meshes = get_circuit(use_mock=USE_MOCK_CIRCUIT)
    feats = load_audio_features(SONG_PATH)
    sim = simulate(nodes, conn, feats)
    window = get_window(sim, start_s=START_S, end_s=END_S)

    os.makedirs("out", exist_ok=True)
    render_video(nodes, skeletons, window, "out", song_path=SONG_PATH)
