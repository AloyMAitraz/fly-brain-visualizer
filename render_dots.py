"""
render_dots.py
===============
Fast alternative to render_activity_frame.py's full-skeleton rendering.
Instead of drawing every branch of every neuron (929k+ line segments for
the real circuit -- ~40s/frame, far too slow for a full video), each
neuron is reduced to a single point (the centroid of its skeleton) that
lights up with activity. Matplotlib handles a few hundred points easily,
so this should render in a fraction of a second per frame instead.

Also adds a translucent brain-outline shell computed directly from the
skeleton point cloud via a convex hull -- no dependency on the ROI mesh
API, which returned HTTP 400 for this dataset/endpoint earlier.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import numpy as np

from connectome import get_circuit
from audio_input import load_audio_features
from activity_model import simulate, get_window

ROI_BASE_COLOR = {
    "JO": np.array([0x4F, 0xC3, 0xF7]) / 255,
    "AMMC": np.array([0xFF, 0xB7, 0x4D]) / 255,
    "WED": np.array([0xEF, 0x53, 0x50]) / 255,
}
MIN_BRIGHTNESS = 0.15


def compute_neuron_points(skeletons):
    """
    One representative 3D point per neuron -- the centroid of all its
    skeleton points. Simple and always available (works identically for
    mock or real data) without needing a separate soma-location fetch.
    """
    return skeletons.groupby("bodyId")[["x", "y", "z"]].mean().reset_index()


def compute_brain_outline(skeletons, subsample=4000, seed=0):
    """
    Translucent outer shell of the whole point cloud via convex hull --
    gives visual "this is the brain's shape" context without needing the
    ROI mesh API (which isn't available for this dataset via navis).
    Subsamples first since ConvexHull on 900k+ points is unnecessarily
    slow and the hull only depends on the outermost points anyway.
    """
    pts = skeletons[["x", "y", "z"]].to_numpy()
    if len(pts) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), subsample, replace=False)
        pts = pts[idx]
    if len(pts) < 4:
        return None
    try:
        return ConvexHull(pts)
    except Exception:
        return None


def render_dot_frame(nodes, points_df, body_ids, activity_row, out_path,
                      title="", hull=None, hull_points=None):
    activity_by_body = dict(zip(body_ids, activity_row))
    roi_by_body = nodes.set_index("bodyId")["roi"].to_dict()
    points_by_body = points_df.set_index("bodyId")

    fig = plt.figure(figsize=(10, 7), facecolor="#0b0b12")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b0b12")

    # translucent brain-shape outline, drawn first (behind the dots)
    if hull is not None and hull_points is not None:
        faces = [hull_points[simplex] for simplex in hull.simplices]
        shell = Poly3DCollection(faces, alpha=0.06, facecolor="#8888aa", edgecolor="none")
        ax.add_collection3d(shell)

    xs, ys, zs, colors, sizes = [], [], [], [], []
    for body_id in points_df["bodyId"]:
        if body_id not in points_by_body.index:
            continue
        roi = roi_by_body.get(body_id, "?")
        base_color = ROI_BASE_COLOR.get(roi, np.array([0.5, 0.5, 0.5]))
        level = activity_by_body.get(body_id, 0.0)
        brightness = MIN_BRIGHTNESS + (1 - MIN_BRIGHTNESS) * level

        p = points_by_body.loc[body_id]
        xs.append(p["x"]); ys.append(p["y"]); zs.append(p["z"])
        colors.append(base_color * brightness)
        sizes.append(15 + 120 * level)

    ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.9, edgecolors="none")

    ax.set_title(title, color="white", fontsize=13)
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-60)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    import time

    # ---- CONFIG ----
    USE_MOCK_CIRCUIT = True
    SONG_PATH = None
    TIMESTAMP_S = 0.6
    # -----------------

    nodes, conn, skeletons, meshes = get_circuit(use_mock=USE_MOCK_CIRCUIT)
    feats = load_audio_features(SONG_PATH)
    sim = simulate(nodes, conn, feats)

    points_df = compute_neuron_points(skeletons)
    hull = compute_brain_outline(skeletons)
    hull_points = skeletons[["x", "y", "z"]].to_numpy() if hull is not None else None
    if hull is not None:
        # ConvexHull was computed on a subsample; rebuild points array to
        # match subsample indices used inside compute_brain_outline is
        # unnecessary here since ConvexHull stores its own .points already
        hull_points = hull.points

    times = sim["times"]
    frame_idx = int(np.argmin(np.abs(times - TIMESTAMP_S)))
    activity_row = sim["activity"][frame_idx]

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", f"dots_t{times[frame_idx]:.2f}s.png")

    t0 = time.time()
    render_dot_frame(
        nodes, points_df, sim["body_ids"], activity_row, out_path,
        title=f"Brain activity (dots) at t={times[frame_idx]:.2f}s",
        hull=hull, hull_points=hull_points,
    )
    elapsed = time.time() - t0
    print(f"Saved {out_path}  (rendered in {elapsed:.2f}s)")
