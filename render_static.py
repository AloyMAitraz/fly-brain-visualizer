"""
render_static.py
=================
Day 1-2 goal: get a static 3D render of the real neuron shapes working,
no activity/animation yet. Colors by ROI layer just to confirm the
skeletons and circuit structure look right.

Uses matplotlib's Agg backend so it runs headless (no display needed) --
same as it'll run on your laptop.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from connectome import get_circuit

ROI_COLORS = {
    "JO": "#4FC3F7",     # input layer - cool blue
    "AMMC": "#FFB74D",   # mid layer - orange
    "WED": "#EF5350",    # output layer - red
}


def render_static(nodes, skeletons, out_path):
    fig = plt.figure(figsize=(10, 7), facecolor="#0b0b12")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b0b12")

    roi_by_body = nodes.set_index("bodyId")["roi"].to_dict()

    for body_id, grp in skeletons.groupby("bodyId"):
        roi = roi_by_body.get(body_id, "?")
        color = ROI_COLORS.get(roi, "#888888")
        grp = grp.sort_values("node_id")
        pts = grp[["x", "y", "z"]].to_numpy()

        # draw parent-child segments as a simple stick skeleton
        segs = []
        id_to_xyz = {row.node_id: (row.x, row.y, row.z) for row in grp.itertuples()}
        for row in grp.itertuples():
            if row.parent_id in id_to_xyz:
                segs.append([id_to_xyz[row.parent_id], (row.x, row.y, row.z)])
        if segs:
            lc = Line3DCollection(segs, colors=color, linewidths=1.2, alpha=0.85)
            ax.add_collection3d(lc)
        ax.scatter(*pts.T, color=color, s=4, alpha=0.6)

    ax.set_title("JO neurons -> AMMC & WED (parallel targets)", color="white", fontsize=13)
    ax.set_axis_off()

    # legend
    for roi, color in ROI_COLORS.items():
        ax.scatter([], [], [], color=color, label=roi)
    ax.legend(loc="upper left", facecolor="#0b0b12", labelcolor="white", framealpha=0.3)

    ax.view_init(elev=15, azim=-60)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    import os
    # ---- CONFIG ----
    USE_MOCK_CIRCUIT = True   # set False to render the real fetched circuit
    # -----------------
    nodes, conn, skeletons, meshes = get_circuit(use_mock=USE_MOCK_CIRCUIT)
    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", "static_render.png")
    render_static(nodes, skeletons, out_path)
    print(f"Saved {out_path}  ({len(nodes)} neurons, {len(skeletons)} skeleton points)")
