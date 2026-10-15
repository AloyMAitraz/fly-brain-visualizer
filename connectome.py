"""
connectome.py
=============
Pulls the Johnston's-organ -> AMMC -> WED auditory sub-circuit from the
MaleCNS connectome (neuPrint dataset "male-cns:v1.0"), OR builds a
structurally-identical mock dataset when no network/token is available
(e.g. this sandbox).

Real usage (on your laptop, during setup week):
    1. pip install neuprint-python navis
    2. Create an account at https://neuprint.janelia.org and grab your
       API token from your account page.
    3. export NEUPRINT_TOKEN="your-token-here"
    4. python -c "from connectome import get_circuit; get_circuit(use_mock=False)"

That single call caches neurons.csv, synapses.csv, and skeletons/*.swc to
./cache/ so the rest of the pipeline (and demo day) never touches the
network again.
"""

import os
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NEUPRINT_SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

# The pathway we care about: sound -> JO sensory neurons -> AMMC -> WED
#
# Real type names in male-cns:v1.0 (confirmed via find_roi_names.py -- the
# hemibrain-style names "JO-A"/"JO-B"/"JO-C"/"JO-E" don't exist in this
# dataset; it uses much finer-grained subtypes instead). We use the full
# A and B subtype families here, since those are the sound/vibration-
# sensitive JO neurons in fly neuroscience (courtship song, near-field
# sound) -- the C/CA/CL/CM/D/E groups lean more toward gravity/wind
# sensing and aren't as relevant for an audio-driven demo.
JO_TYPES = [
    "JO-A1", "JO-A2", "JO-A3", "JO-A4",
    "JO-B1_a", "JO-B1_b", "JO-B1_c", "JO-B2", "JO-B3", "JO-B4_a", "JO-B4_b",
]
ROIS_OF_INTEREST = ["AMMC(L)", "AMMC(R)", "WED(L)", "WED(R)"]   # real ROI names (hemisphere-suffixed)


def _label_by_total_roi_innervation(bodyIds, client):
    """
    Label each neuron AMMC/WED by where it lives OVERALL (its full ROI
    footprint via roiInfo), not by where its specific synapses with our
    JO neurons happen to be routed. This matters because a JO->downstream
    synapse can physically sit in an intermediate relay zone (e.g. SAD,
    which is anatomically adjacent to AMMC) even when that downstream
    neuron's main territory is genuinely AMMC or WED -- per-edge roi
    labeling was undercounting AMMC almost entirely because of this.

    fetch_adjacencies()'s own neuron table doesn't include roiInfo (only
    fetch_neurons() does), so this does a dedicated small fetch just for
    these bodyIds.
    """
    from neuprint import fetch_neurons, NeuronCriteria as NC

    if not bodyIds:
        return {}

    detail, _ = fetch_neurons(NC(bodyId=list(bodyIds)), client=client)
    labels = {}
    for row in detail.itertuples():
        roi_info = getattr(row, "roiInfo", {})
        if not isinstance(roi_info, dict):
            continue

        def total(names):
            t = 0
            for n in names:
                counts = roi_info.get(n, {})
                if isinstance(counts, dict):
                    t += counts.get("pre", 0) + counts.get("post", 0)
            return t

        ammc = total(["AMMC(L)", "AMMC(R)"])
        wed = total(["WED(L)", "WED(R)"])
        if ammc == 0 and wed == 0:
            continue
        labels[row.bodyId] = "AMMC" if ammc >= wed else "WED"
    return labels


def _fetch_real():
    """Pull the real sub-circuit from neuPrint. Needs NEUPRINT_TOKEN + internet."""
    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "Set NEUPRINT_TOKEN env var first (get it from your neuprint.janelia.org account page)."
        )

    client = Client(NEUPRINT_SERVER, dataset=DATASET, token=token)

    # NOTE: we pass client=client explicitly to every call below instead
    # of relying on neuprint-python's "default client" auto-detection.
    # That auto-detection uses a weakref internally and was proving
    # unreliable in testing (client kept appearing "not set" even when
    # constructed successfully) -- passing it explicitly sidesteps the
    # issue entirely and is more robust regardless of environment.

    # 1. Johnston's organ sensory neurons (the "input" layer)
    jo_neurons, _ = fetch_neurons(NC(type=JO_TYPES), client=client)

    # 2. Everything downstream of them, restricted to AMMC/WED so the demo
    #    circuit stays small (few hundred neurons, not the whole brain).
    downstream_neurons, conn = fetch_adjacencies(
        sources=NC(type=JO_TYPES),
        targets=NC(rois=ROIS_OF_INTEREST),
        client=client,
    )

    # 3. Union of node table (JO + downstream), deduplicated
    nodes = pd.concat([jo_neurons, downstream_neurons]).drop_duplicates(subset="bodyId")

    # 3b. Add a simple "roi" column (JO / AMMC / WED) matching the schema
    # the rest of the pipeline expects (activity_model.py, render_*.py).
    # Real neuprint data doesn't come with this -- JO neurons are tagged
    # directly (already searched by type), and downstream neurons get
    # labeled by their OVERALL ROI footprint (see
    # _label_by_total_roi_innervation -- per-edge roi labeling was tried
    # first but badly undercounted AMMC, since JO->downstream synapses
    # often physically route through the adjacent SAD relay zone even
    # when the downstream neuron's real home territory is AMMC).
    jo_body_ids = set(jo_neurons["bodyId"])
    downstream_ids = set(nodes["bodyId"]) - jo_body_ids
    downstream_roi_labels = _label_by_total_roi_innervation(downstream_ids, client)

    def _label_row(row):
        if row["bodyId"] in jo_body_ids:
            return "JO"
        return downstream_roi_labels.get(row["bodyId"], "OTHER")

    nodes["roi"] = nodes.apply(_label_row, axis=1)

    # drop anything that's neither JO nor cleanly AMMC/WED (e.g. passes
    # through the region without real innervation there) so downstream
    # code never has to handle an unexpected roi label
    nodes = nodes[nodes["roi"].isin(["JO", "AMMC", "WED"])].reset_index(drop=True)
    kept_ids = set(nodes["bodyId"])
    conn = conn[conn["bodyId_pre"].isin(kept_ids) & conn["bodyId_post"].isin(kept_ids)]

    # 4. 3D skeletons for rendering (one call per neuron; cached to disk)
    import navis
    from navis.interfaces.neuprint import fetch_skeletons, fetch_roi

    skeletons = fetch_skeletons(nodes["bodyId"].tolist(), client=client)

    # IMPORTANT: per the male-cns:v1.0 dataset notes, raw coordinates are
    # stored in VOXEL units at 8nm/voxel, not nanometers or microns
    # directly. navis.interfaces.neuprint.fetch_skeletons already returns
    # coordinates converted to nanometers internally -- but we convert to
    # MICRONS here (divide by 1000) since that's a much more sane working
    # scale for a brain that's only ~250 microns across (voxel/nm values
    # would otherwise be huge numbers that are awkward to render/animate).
    for skel in skeletons:
        skel.nodes[["x", "y", "z"]] = skel.nodes[["x", "y", "z"]] / 1000.0

    # A handful of bodyIds can fail to fetch a skeleton (server-side gaps
    # -- some neurons in a connectome release don't have a precomputed
    # skeleton). fetch_skeletons() already skips/warns about these rather
    # than crashing; we just need to know which ones actually came back.
    fetched_body_ids = {int(skel.id) for skel in skeletons}
    missing = set(nodes["bodyId"]) - fetched_body_ids
    if missing:
        print(f"Note: {len(missing)} neuron(s) had no skeleton available and were dropped: {sorted(missing)}")

    # Convert navis's NeuronList-of-TreeNeuron format into the SAME flat
    # table schema the mock data uses (bodyId, node_id, parent_id, x, y,
    # z, radius) -- this is what lets render_static.py and
    # render_activity_frame.py work identically on real or mock data
    # without any changes to their code.
    skeleton_rows = []
    for skel in skeletons:
        df = skel.nodes[["node_id", "parent_id", "x", "y", "z"]].copy()
        df["radius"] = skel.nodes["radius"] if "radius" in skel.nodes.columns else 1.0
        df["bodyId"] = int(skel.id)
        skeleton_rows.append(df[["bodyId", "node_id", "parent_id", "x", "y", "z", "radius"]])
    skeletons_df = (
        pd.concat(skeleton_rows, ignore_index=True) if skeleton_rows
        else pd.DataFrame(columns=["bodyId", "node_id", "parent_id", "x", "y", "z", "radius"])
    )

    # Drop any neuron that has no skeleton from nodes/conn too, so every
    # neuron left in the circuit is actually renderable.
    nodes = nodes[nodes["bodyId"].isin(fetched_body_ids)].reset_index(drop=True)
    conn = conn[conn["bodyId_pre"].isin(fetched_body_ids) & conn["bodyId_post"].isin(fetched_body_ids)]

    # 5. ROI mesh outlines (AMMC, WED) for the "brain-shaped" visual
    #    context -- translucent surface the neurons sit inside, rather
    #    than floating in empty space. NOTE: as of this dataset/API
    #    version, mesh fetching returns HTTP 400 for these ROI names
    #    (likely not published for male-cns:v1.0 via this endpoint yet)
    #    -- this degrades gracefully to "no outline" rather than failing
    #    the whole pipeline; neurons still render fine without it.
    meshes = {}
    for roi in ROIS_OF_INTEREST:
        try:
            mesh = fetch_roi(roi, client=client)  # navis.Volume
            mesh.vertices = mesh.vertices / 1000.0  # same voxel->micron fix
            meshes[roi] = mesh
        except Exception as e:
            print(f"Note: no mesh available for {roi} (skipping outline): {e}")

    nodes.to_csv(os.path.join(CACHE_DIR, "neurons.csv"), index=False)
    conn.to_csv(os.path.join(CACHE_DIR, "connections.csv"), index=False)
    skeletons_df.to_csv(os.path.join(CACHE_DIR, "skeletons.csv"), index=False)
    for roi, mesh in meshes.items():
        navis.write_mesh(mesh, os.path.join(CACHE_DIR, f"mesh_{roi}.obj"))

    print(f"Real circuit ready: {len(nodes)} neurons, {len(conn)} connections, {len(skeletons_df)} skeleton points")
    return nodes, conn, skeletons_df, meshes


def _fetch_mock(n_jo=12, n_ammc=8, n_wed=5, seed=0):
    """
    Build a small synthetic circuit with the SAME shape as the real neuprint
    output (bodyId, type, roi, x/y/z skeleton points, weighted edges) so the
    rest of the pipeline can be built and tested without network access.
    """
    rng = np.random.default_rng(seed)
    rows = []
    bid = 1000

    def make_layer(n, roi, type_prefix, center, spread):
        nonlocal bid
        layer_ids = []
        for i in range(n):
            layer_ids.append(bid)
            rows.append({"bodyId": bid, "type": f"{type_prefix}{i}", "roi": roi})
            bid += 1
        return layer_ids

    jo_ids = make_layer(n_jo, "JO", "JO-", center=(0, 0, 0), spread=20)
    ammc_ids = make_layer(n_ammc, "AMMC", "AMMC_", center=(40, 0, 0), spread=20)
    wed_ids = make_layer(n_wed, "WED", "WED_", center=(80, 0, 0), spread=20)

    nodes = pd.DataFrame(rows)

    # Synthetic skeletons: each neuron = a simple branching stick figure
    # (parent-child point list), placed near its ROI's center, so navis/
    # matplotlib can render it as a 3D shape (not just a dot).
    roi_centers = {"JO": (0, 0, 0), "AMMC": (40, 0, 0), "WED": (80, 0, 0)}
    skel_points = []
    for _, row in nodes.iterrows():
        cx, cy, cz = roi_centers[row["roi"]]
        jitter = rng.normal(0, 8, size=3)
        origin = np.array([cx, cy, cz]) + jitter
        n_pts = 6
        pts = origin + rng.normal(0, 4, size=(n_pts, 3)).cumsum(axis=0) * 0.5
        for i, p in enumerate(pts):
            skel_points.append({
                "bodyId": row["bodyId"],
                "node_id": i,
                "parent_id": i - 1 if i > 0 else -1,
                "x": p[0], "y": p[1], "z": p[2],
                "radius": 1.0,
            })
    skeletons = pd.DataFrame(skel_points)

    # Synthetic weighted connections: JO -> AMMC -> WED, feed-forward,
    # random-ish weights (mirrors real neuprint's `weight` column).
    edges = []
    for j in jo_ids:
        targets = rng.choice(ammc_ids, size=rng.integers(2, 5), replace=False)
        for t in targets:
            edges.append({"bodyId_pre": j, "bodyId_post": t, "weight": int(rng.integers(3, 30))})
    for a in ammc_ids:
        targets = rng.choice(wed_ids, size=rng.integers(1, 3), replace=False)
        for t in targets:
            edges.append({"bodyId_pre": a, "bodyId_post": t, "weight": int(rng.integers(3, 30))})
    conn = pd.DataFrame(edges)

    return nodes, conn, skeletons


def get_circuit(use_mock=True):
    """
    Returns (nodes, conn, skeletons, meshes):
      nodes:     bodyId, type, roi
      conn:      bodyId_pre, bodyId_post, weight
      skeletons: bodyId, node_id, parent_id, x, y, z, radius (mock)
                 OR a list of navis TreeNeuron objects (real)
      meshes:    dict of {roi: navis mesh} for brain-shape outline context
                 (None for mock data -- there's no real shape to outline)

    Caches mock output too, so repeated runs are fast and reproducible.
    """
    cache_nodes = os.path.join(CACHE_DIR, "mock_neurons.csv")
    cache_conn = os.path.join(CACHE_DIR, "mock_connections.csv")
    cache_skel = os.path.join(CACHE_DIR, "mock_skeletons.csv")

    if use_mock:
        if all(os.path.exists(p) for p in [cache_nodes, cache_conn, cache_skel]):
            return (pd.read_csv(cache_nodes), pd.read_csv(cache_conn), pd.read_csv(cache_skel), None)
        nodes, conn, skeletons = _fetch_mock()
        nodes.to_csv(cache_nodes, index=False)
        conn.to_csv(cache_conn, index=False)
        skeletons.to_csv(cache_skel, index=False)
        return nodes, conn, skeletons, None
    else:
        return _fetch_real()


if __name__ == "__main__":
    nodes, conn, skeletons, meshes = get_circuit(use_mock=True)
    print(f"Neurons: {len(nodes)}  |  Connections: {len(conn)}  |  Skeleton points: {len(skeletons)}")
    print(nodes["roi"].value_counts())
