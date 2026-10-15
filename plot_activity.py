"""
plot_activity.py
=================
Runs the activity simulation and saves a graph showing mean activity per
layer (JO / AMMC / WED) over time -- lets you SEE the brain "pulse" with
the beat, as a line chart, before we build the actual 3D animated version.

Run this any time you want to check activity behaves sensibly, e.g. after
changing DECAY/SPREAD_WEIGHT in activity_model.py, or after switching to
a real song.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from connectome import get_circuit
from audio_input import load_audio_features
from activity_model import simulate

# ---- CONFIG: swap these two lines to use your real song later ----
USE_MOCK_CIRCUIT = True
SONG_PATH = None   # e.g. "my_song.mp3"
# --------------------------------------------------------------------

nodes, conn, skeletons, meshes = get_circuit(use_mock=USE_MOCK_CIRCUIT)
feats = load_audio_features(SONG_PATH)
sim = simulate(nodes, conn, feats)

roi_by_body = nodes.set_index("bodyId")["roi"].to_dict()
body_ids = sim["body_ids"]
activity = sim["activity"]

fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="#0b0b12")
ax.set_facecolor("#0b0b12")
colors = {"JO": "#4FC3F7", "AMMC": "#FFB74D", "WED": "#EF5350"}

for roi, color in colors.items():
    cols = [i for i, b in enumerate(body_ids) if roi_by_body[b] == roi]
    mean_activity = activity[:, cols].mean(axis=1)
    ax.plot(sim["times"], mean_activity, color=color, label=f"{roi} (mean)", linewidth=1.2)

ax.legend(facecolor="#0b0b12", labelcolor="white")
title = "test tone" if SONG_PATH is None else SONG_PATH
ax.set_title(f"Mean activity per layer over time ({title})", color="white")
ax.set_xlabel("time (s)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

plt.tight_layout()
os.makedirs("out", exist_ok=True)
out_path = os.path.join("out", "activity_plot.png")
fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
print(f"Saved {out_path}")
