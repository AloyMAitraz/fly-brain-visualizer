"""
activity_model.py
==================
Day 3, part 2: the actual "brain reacting to sound" simulation.

Simple rate-based model (not a full spiking simulation — deliberately, per
the plan, since this is a small sub-circuit on a laptop CPU):

  activity[t] = decay * activity[t-1]                      <- fades over time
              + spread_weight * (W @ activity[t-1])         <- flows along real synapses
              + stimulus_gain * external_input[t]            <- audio driving JO neurons only

Where W is the real (or mock) synaptic weight matrix, normalized so a
neuron's total outgoing influence doesn't blow up regardless of how many
downstream partners it has.

KEY DESIGN CHOICE for "any duration, skip around" support:
  simulate() computes activity for the ENTIRE song once and caches it.
  get_window(start_s, end_s) then just slices that cached array — so
  scrubbing to any point in the song is instant, no re-simulation needed.
"""

import numpy as np
import pandas as pd

DECAY = 0.55           # how fast activity fades per frame (0-1, higher = longer memory)
SPREAD_WEIGHT = 0.30   # how strongly activity flows along synapses per frame
STIMULUS_GAIN = 1.2    # how strongly audio drives JO neurons

# STABILITY NOTE: with row-normalized weights (each neuron's incoming
# synapse weights sum to <=1), the steady-state amplification per layer is
# roughly spread/(1-decay). Keep this BELOW 1.0 or activity cascades and
# saturates through downstream layers (this bit us during tuning: with
# decay=0.85, spread=0.35 the gain was ~6.7x per layer and WED flatlined
# at the clip ceiling almost immediately). 0.30/(1-0.55) = 0.67x is stable
# while still giving a visible decay trail rather than an instant on/off.


def build_adjacency(nodes, conn):
    """
    Build a normalized weight matrix W (n_neurons x n_neurons) from the
    connection table, plus the bodyId -> row index mapping used everywhere
    else (audio driving, rendering, etc.).
    """
    body_ids = nodes["bodyId"].tolist()
    idx = {b: i for i, b in enumerate(body_ids)}
    n = len(body_ids)

    W = np.zeros((n, n))
    for row in conn.itertuples():
        if row.bodyId_pre in idx and row.bodyId_post in idx:
            W[idx[row.bodyId_post], idx[row.bodyId_pre]] += row.weight

    # Normalize each neuron's INCOMING weights to sum to 1 (row-normalize).
    # This is the critical stability fix: a neuron's total activity intake
    # from upstream partners is bounded to 1x regardless of fan-in, so
    # heavily-converged-onto neurons (like WED here) don't accumulate
    # more signal just because more things happen to synapse onto them.
    # (Normalizing only outgoing weights, as before, left fan-in
    # unbounded and caused downstream neurons to saturate/flatline.)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    return W_norm, idx, body_ids


def drive_from_audio(nodes, idx, feats):
    """
    Maps audio features onto the JO (input) neurons only, frame by frame.
    Different JO neurons get driven by different frequency bands, since
    real JO subtypes are frequency-tuned -- gives a richer, less uniform
    pulse than driving every JO neuron identically.
    """
    n_frames = len(feats["times"])
    n_neurons = len(idx)
    stimulus = np.zeros((n_frames, n_neurons))

    jo_rows = nodes[nodes["roi"] == "JO"]
    bands = ["low_band", "mid_band", "high_band"]

    for i, row in enumerate(jo_rows.itertuples()):
        band = bands[i % len(bands)]           # spread JO neurons across bands
        band_signal = feats[band]
        # combine the band's frequency energy with the overall beat onset,
        # so quiet-but-percussive moments still produce a sharp pulse
        combined = 0.6 * band_signal + 0.4 * feats["onset"]
        stimulus[:, idx[row.bodyId]] = combined

    return stimulus


def simulate(nodes, conn, feats, decay=DECAY, spread=SPREAD_WEIGHT, gain=STIMULUS_GAIN):
    """
    Runs the full activity simulation across the ENTIRE song once.
    Returns a dict: {activity: (n_frames, n_neurons), body_ids, times}
    """
    W, idx, body_ids = build_adjacency(nodes, conn)
    stimulus = drive_from_audio(nodes, idx, feats)

    n_frames, n_neurons = stimulus.shape
    activity = np.zeros((n_frames, n_neurons))

    a = np.zeros(n_neurons)
    for t in range(n_frames):
        spread_input = W @ a
        a = decay * a + spread * spread_input + gain * stimulus[t]
        a = np.clip(a, 0.0, 3.0)  # avoid runaway feedback blowup
        activity[t] = a

    # normalize final activity to 0-1 for rendering (keep relative dynamics)
    max_val = activity.max()
    if max_val > 1e-9:
        activity = activity / max_val

    return {
        "activity": activity,       # shape (n_frames, n_neurons)
        "body_ids": body_ids,       # column order matches activity's 2nd axis
        "times": feats["times"],    # row order matches activity's 1st axis
    }


def get_window(sim_result, start_s=None, end_s=None):
    """
    Slice a cached simulation to any [start_s, end_s) window -- this is
    what makes 'skip forward and view any duration' instant: no
    re-simulation, just a numpy slice.

    start_s=None -> from the beginning. end_s=None -> to the end.
    """
    times = sim_result["times"]
    start_s = times[0] if start_s is None else start_s
    end_s = times[-1] if end_s is None else end_s

    mask = (times >= start_s) & (times <= end_s)
    return {
        "activity": sim_result["activity"][mask],
        "body_ids": sim_result["body_ids"],
        "times": times[mask],
    }


if __name__ == "__main__":
    from connectome import get_circuit
    from audio_input import load_audio_features

    nodes, conn, skeletons, meshes = get_circuit(use_mock=True)
    feats = load_audio_features(None)  # synthetic test tone

    sim = simulate(nodes, conn, feats)
    print(f"Simulated activity shape: {sim['activity'].shape}  (frames x neurons)")
    print(f"Activity range: {sim['activity'].min():.2f} - {sim['activity'].max():.2f}")

    # sanity check: slice a 2-second window from the middle
    window = get_window(sim, start_s=2.0, end_s=4.0)
    print(f"Window (2s-4s): {window['activity'].shape[0]} frames")
