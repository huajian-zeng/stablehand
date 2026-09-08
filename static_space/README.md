---
title: StableHand · Motion Gallery
emoji: ✋
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
fullWidth: true
header: mini
short_description: Explore precomputed world-space hand motion on HOT3D and ARCTIC.
---

# StableHand · Motion Gallery

A static gallery of 10 precomputed StableHand examples: six HOT3D and four
ARCTIC sequences. Compare the input video, ground truth, and StableHand in a
single synchronized player. Filter examples, slow playback, or step through
individual frames. This gallery plays saved results; it does not accept uploads
or run inference. No Python service, GPU, remote API, or JavaScript dependencies
are required at runtime. Fonts, posters, and videos are served locally.

## Preview

Open `index.html` directly in your browser. For an HTTP preview, run
`npx http-server . -p 8789 -a 127.0.0.1 -c-1` from this directory and open
http://127.0.0.1:8789. Use a server that supports HTTP Range requests so browser
video seeking works; Python's basic `http.server` does not support byte ranges.

## Media preparation

The parent repository's `scripts/prepare_static_demo.py` reads existing
`outputs/hot3d/*_world3d.mp4` and `outputs/arctic/*_world3d.mp4` renders. It trims
their burned-in title/footer margins, encodes browser-compatible H264 video,
and writes `cases.js`, thumbnails, posters, and `media-provenance.json`.
The three panels remain in their original order, with identical timing.
The blue mesh is the left hand; orange is the right hand, matching the renderer.

Run from the parent repository:

```bash
python scripts/prepare_static_demo.py --source /path/to/stablehand_release
```

## Hugging Face deployment

This directory is the complete Static Space bundle. The README YAML configures
`sdk: static`; it does not set repository visibility. Upload the contents of this
directory to the root of the chosen Space, with `README.md` and `index.html`
at the Space root. Publishing a Space is a separate step from using this gallery
in the code repository.

Do not upload the parent training/inference repository, model weights, MANO
assets, or private validation outputs. The gallery media is derived from the
already prepared HOT3D/ARCTIC demo results; the source datasets retain their
respective terms. Gallery source follows the
[StableHand repository license](https://github.com/huajian-zeng/stablehand/blob/main/LICENSE).
See the [project](https://huajian-zeng.github.io/projects/stablehand/) and
[dataset release](https://huggingface.co/datasets/huajian-zeng/stablehand-data)
for attribution.

## Keyboard controls

When focus is outside a link, button, or native input: Space plays/pauses;
Left/Right step one frame; Home returns to the first frame. All controls are also
available as labeled buttons. The timeline and playback speed use native inputs.
