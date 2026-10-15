# fly-brain-audio

Feed a song into a real *Drosophila* connectome and watch the neurons light up.

This takes the real wiring diagram of a fly's auditory pathway (Johnston's organ
sensory neurons and the brain regions they feed), drives the sensory neurons with
features pulled from an audio file, lets activity spread along the real synapses,
and renders the result as a video.

## What this is, and isn't

- The **wiring is real**: neurons, connections and 3D shapes come from the MaleCNS
  connectome (see Data & credit below).
- The **activity is a simple model**: a rate-based simulation (activity decays,
  and flows along synaptic weights). It is a visualization of how a measured
  wiring diagram reacts to an input. It is **not** a faithful biophysical simulation,
  and the fly is not "hearing" or "enjoying" the music.
- A finding from the real data: Johnston's organ neurons project to **AMMC and WED in
  parallel** (hundreds of direct edges to each, zero AMMC-to-WED edges in this
  subset), not as a strict relay chain. So both layers light up together.

## Quick start (no account, no data download)

Everything runs out of the box on a small synthetic circuit and a synthetic click track.

```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python render_video.py
```

You also need [ffmpeg](https://ffmpeg.org/) installed and on your PATH.
Output: `out/brain_clip.mp4`.

## Files

| File | What it does |
|---|---|
| `connectome.py` | Gets the circuit: real (neuPrint) or mock. Returns neurons, connections, 3D skeletons. |
| `audio_input.py` | Audio file to per-frame amplitude, beat onset, and low/mid/high band energy (librosa). |
| `activity_model.py` | Rate-based activity model; `simulate()` once, `get_window()` to slice any time range. |
| `render_dots.py` | Fast renderer: one glowing dot per neuron inside a translucent hull outline. |
| `render_video.py` | Renders frames over a time window, stitches them with ffmpeg, adds the audio. |
| `render_static.py` | One static image of the circuit's 3D shapes. |
| `render_activity_frame.py` | One frame with full branching skeletons (very slow on real data). |
| `plot_activity.py` | Line chart of mean activity per layer over time. |

Each script has a small `CONFIG` block near the top or bottom (`USE_MOCK_CIRCUIT`,
`SONG_PATH`, time window).

## Using real data and your own song

1. Create a free account at <https://neuprint.janelia.org> and copy your API token
   from your account page. Treat it like a password and never commit it.
2. Set it as an environment variable (the code never stores it):
   ```
   $env:NEUPRINT_TOKEN = "your-token"        # PowerShell
   export NEUPRINT_TOKEN="your-token"        # Mac/Linux
   ```
3. In `render_video.py` set:
   ```python
   USE_MOCK_CIRCUIT = False
   SONG_PATH = "my_song.mp3"
   START_S, END_S = 0.0, 20.0
   ```
4. Run `python render_video.py`.

The first real run fetches about 200 neurons and their skeletons (a minute or two).
Results are written to `cache/`.

Use audio you have the rights to. If you publish the video, a copyrighted song can get
it flagged or muted; royalty-free tracks avoid that.

## Data & credit

Connectome: **MaleCNS v1.0** (`male-cns:v1.0` on neuPrint), from the Janelia FlyEM
Team Project, the Cambridge Drosophila Connectomics Group, and Google Connectomics.
See <https://male-cns.janelia.org> for the dataset and its terms of use.

Built with [neuprint-python](https://github.com/connectome-neuprint/neuprint-python),
[navis](https://github.com/navis-org/navis) and [librosa](https://librosa.org).

## Circuit definition

Sensory input: Johnston's organ types `JO-A1..A4` and `JO-B1_a..B4_b` (the vibration /
sound-sensitive families). Downstream: neurons they directly synapse onto that innervate
`AMMC` or `WED`, labeled by where each neuron has most of its synapses overall.
Coordinates are converted to microns.

## Known limitations

- The activity model is deliberately simple; parameters (`DECAY`, `SPREAD_WEIGHT`,
  `STIMULUS_GAIN`) were tuned by eye so the output pulses with the beat without saturating.
- Each neuron is drawn as one dot at the centroid of its skeleton, not at its cell body.
- The brain outline is a convex hull of the neuron points. The per-region mesh endpoint
  returned HTTP 400 for this dataset, so it is not used.
- `connectome.py` writes `cache/` but does not yet reload from it; real mode always
  re-fetches.
- Full-skeleton rendering of the real circuit takes about 40 s per frame, which is why
  the dot renderer exists.
