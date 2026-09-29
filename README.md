# 🎯 Furrey-TurnaroundSplitter for ComfyUI

Automatically splits a character turnaround sheet (front/side/back/side, or any number of
side-by-side views) into separate cropped images — no manual pixel math required.

Built for multi-view image-to-3D pipelines (Trellis2 / Pixal3D, Hunyuan3D, TripoSR, etc.)
that need each view of a turnaround sheet fed in separately, but works with any sheet of
flat-background images placed side by side.

## ✨ Features

* **🔍 Auto-detects the background color** — works with black, white, or any flat/uniform
  background, no need to specify it.
* **📐 Auto-detects each view's boundaries** — finds the gaps between figures and crops
  around them, regardless of image size, aspect ratio, or how many views are in the sheet.
* **🧩 Handles uneven views** — each cropped output keeps its own width (a front/back pose
  is usually wider than a side profile), matching what multi-view 3D pipelines expect.
* **🛠️ Tunable** — padding, background sensitivity, minimum gap/width and an optional
  tight vertical crop are all exposed as widgets for tricky sheets.
* **📝 Debug output** — an `info` text output reports exactly which box was used for each
  view, so you can see what was detected without guessing.

## 📦 Installation

### Via ComfyUI Manager (Recommended)
1. Search for "Furrey-TurnaroundSplitter" (once indexed).
2. Click Install.

### Manual Installation
1. Go to your `ComfyUI/custom_nodes/` directory.
2. Clone this repo:
   ```
   git clone https://github.com/SilentLuxRay/ComfyUI-Furrey-TurnaroundSplitter
   ```
3. Restart ComfyUI.

## 🧠 How it works

The node samples the image's border pixels to learn the background color, then measures
how much each column of the image differs from that color. Columns that are mostly
background mark the gaps between figures; contiguous "content" columns become one detected
view. Each view is then cropped out (with your chosen padding) at its own natural width —
nothing is stretched or resized to force a uniform size.

## ⚙️ Node reference

**Auto-Split Turnaround Sheet** (category `image/transform`)

| Input | Description |
|---|---|
| `image` | The turnaround sheet to split |
| `num_views` | How many views to detect/output (1-8) |
| `padding` | Fixed extra margin (px) added around each detected view. Ignored if `margin_percent` > 0 |
| `margin_percent` | Margin as a % of the view's own size, added on every side (self-scales to any resolution). Many 3D/conditioning pipelines expect the subject to occupy ~90% of the frame (~8-10% margin), not edge-to-edge — 0 disables this and falls back to `padding` |
| `bg_threshold` | Per-pixel distance from the detected background color to count as "content" |
| `min_col_fraction` | Minimum fraction of a column that must be "content" to count it as part of a figure (filters noise) |
| `merge_gap` | Background gaps narrower than this (px) get merged into the same view, instead of splitting it |
| `min_width` | Minimum width (px) for a detected block to count as a real view, not noise |
| `tight_vertical` | Crop each view to its own content height instead of the full image height |
| `pad_to_square` | Pad each cropped view to a square canvas, centered, filled with the detected background color — useful for pipelines (e.g. CLIP-Vision-based conditioning) that expect square, centered inputs |

Outputs: `view_1` … `view_8` (unused slots return an 8×8 black placeholder) and `info`
(a text summary of what was detected, for debugging).

## 🖼️ Example

Feed a `LoadImage` of a 4-view turnaround sheet straight into this node, `num_views=4`,
and wire `view_1`..`view_4` into whatever your pipeline expects per view (RemoveBackground,
crop-to-mask, conditioning, etc.) — no manual x/y/width/height guessing.

## 📐 Bonus node: Scale Mesh (per axis)

A small utility node (category `3d/mesh`) that scales a `MESH`'s vertices independently
per axis (X/Y/Z) around the bounding-box center (or world origin). Useful when a 3D
generator's output comes out squashed or stretched on one axis and there's no exposed
parameter to fix it upstream — impose the correct proportion directly in the workflow,
right after mesh generation and before saving, instead of fixing it by hand in Blender
every time.

| Input | Description |
|---|---|
| `mesh` | The mesh to scale |
| `scale_x` / `scale_y` / `scale_z` | Per-axis multiplier (1.0 = unchanged) |
| `pivot` | `bbox_center` (scale around the mesh's own bounding-box center) or `origin` (scale around world 0,0,0) |

---

Created by Furrey for the ComfyUI community.
