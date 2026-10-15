"""
render_activity_frame.py
=========================
Day 4: render the 3D brain at ONE specific moment in time, with each
neuron colored/brightened by how active it is right then.

This is the missing link between activity_model.py (numbers) and
render_static.py (shapes) -- once this single frame looks right,
animating it across many frames (Day 5) is just calling this in a loop.

Color mapping:
  low activity  -> dim, desaturated color (resting)
  high activity -> bright, saturated color (firing)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from connectome import get_circuit
from audio_input import load_audio_features
from activity_model import simulate, get_window

# base hue per layer (same palette as render_static.py) -- activity
# controls BRIGHTNESS of this color, not which color is used
ROI_BASE_COLOR = {
    "JO": np.array([0x4F, 0xC3, 0xF7]) / 255,
    "AMMC": np.array([0xFF, 0xB7, 0x4D]) / 255,
    "WED": np.array([0xEF, 0x53, 0x50]) / 255,
}
MIN_BRIGHTNESS = 0.15  # even "resting" neurons stay faintly visible


def activity_to_color(base_color, activity_level):
    """Blend from a dim version of base_color up to full-bright base_color."""
    brightness = MIN_BRIGHTNESS + (1 - MIN_BRIGHTNESS) * activity_level
    return base_color * brightness


def render_frame(nodes, skeletons, body_ids, activity_row, out_path, title=""):
    """
    activity_row: 1D array, one activity value per neuron (matches body_ids order)
    """
    activity_by_body = dict(zip(body_ids, activity_row))
    roi_by_body = nodes.set_index("bodyId")["roi"].to_dict()

    fig = plt.figure(figsize=(10, 7), facecolor="#0b0b12")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b0b12")

    for body_id, grp in skeletons.groupby("bodyId"):
        roi = roi_by_body.get(body_id, "?")
        base_color = ROI_BASE_COLOR.get(roi, np.array([0.5, 0.5, 0.5]))
        level = activity_by_body.get(body_id, 0.0)
        color = activity_to_color(base_color, level)

        grp = grp.sort_values("node_id")
        id_to_xyz = {row.node_id: (row.x, row.y, row.z) for row in grp.itertuples()}
        segs = []
        for row in grp.itertuples():
            if row.parent_id in id_to_xyz:
                segs.append([id_to_xyz[row.parent_id], (row.x, row.y, row.z)])
        if segs:
            # brighter neurons get thicker lines too, so firing really pops
            linewidth = 1.0 + 3.0 * level
            lc = Line3DCollection(segs, colors=[color], linewidths=linewidth, alpha=0.95)
            ax.add_collection3d(lc)

        pts = grp[["x", "y", "z"]].to_numpy()
        size = 4 + 40 * level
        ax.scatter(*pts.T, color=[color], s=size, alpha=min(1.0, 0.5 + level))

    ax.set_title(title, color="white", fontsize=13)
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-60)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    # ---- CONFIG ----
    USE_MOCK_CIRCUIT = True     # False = fetch the real circuit (needs NEUPRINT_TOKEN)
    SONG_PATH = None            # e.g. "my_song.mp3"
    TIMESTAMP_S = 0.6           # the moment to render, in seconds
    # -----------------

    nodes, conn, skeletons, meshes = get_circuit(use_mock=USE_MOCK_CIRCUIT)
    feats = load_audio_features(SONG_PATH)
    sim = simulate(nodes, conn, feats)

    # find the frame closest to TIMESTAMP_S
    times = sim["times"]
    frame_idx = int(np.argmin(np.abs(times - TIMESTAMP_S)))
    actual_time = times[frame_idx]
    activity_row = sim["activity"][frame_idx]

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", f"frame_t{actual_time:.2f}s.png")
    render_frame(
        nodes, skeletons, sim["body_ids"], activity_row, out_path,
        title=f"Brain activity at t={actual_time:.2f}s"
    )
    print(f"Saved {out_path}")
    print(f"Mean activity this frame: {activity_row.mean():.2f} (max: {activity_row.max():.2f})")
