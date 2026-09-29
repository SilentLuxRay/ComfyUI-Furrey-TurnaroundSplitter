import torch

MAX_VIEWS = 8


class AutoSplitTurnaroundSheet:
    """Detects N side-by-side character views in a turnaround sheet by finding
    background-colored gap columns between them, and crops each one out.
    Works regardless of image size/aspect ratio or number of views, as long as
    each view is separated from its neighbors by a column of flat background."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_views": ("INT", {"default": 4, "min": 1, "max": MAX_VIEWS}),
                "padding": ("INT", {"default": 12, "min": 0, "max": 512}),
                "bg_threshold": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                                            "tooltip": "Per-pixel distance from the detected background color, above which a pixel counts as 'content'."}),
                "min_col_fraction": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005,
                                                 "tooltip": "Minimum fraction of a column's pixels that must be 'content' for the column to count as part of a figure (filters noise)."}),
                "merge_gap": ("INT", {"default": 8, "min": 0, "max": 512,
                                       "tooltip": "Background gaps narrower than this (in px) get merged into the same figure block."}),
                "min_width": ("INT", {"default": 20, "min": 1, "max": 4096,
                                       "tooltip": "Minimum block width in px to be considered a real figure, not noise."}),
                "tight_vertical": ("BOOLEAN", {"default": False,
                                                "tooltip": "Crop each view to its own tight vertical content bounds instead of the full image height."}),
                "pad_to_square": ("BOOLEAN", {"default": False,
                                               "tooltip": "Pad each cropped view to a square canvas (centered, filled with the detected background color). Useful for pipelines (e.g. CLIPVisionEncode-based conditioning) that expect square, centered inputs."}),
            }
        }

    RETURN_TYPES = tuple(["IMAGE"] * MAX_VIEWS) + ("STRING",)
    RETURN_NAMES = tuple(f"view_{i+1}" for i in range(MAX_VIEWS)) + ("info",)
    FUNCTION = "split"
    CATEGORY = "image/transform"

    def split(self, image, num_views, padding, bg_threshold, min_col_fraction,
              merge_gap, min_width, tight_vertical, pad_to_square=False):
        ref = image[0]  # H, W, C
        H, W = ref.shape[0], ref.shape[1]

        border = torch.cat([ref[0, :, :], ref[-1, :, :], ref[:, 0, :], ref[:, -1, :]], dim=0)
        bg_color = border.mean(dim=0)

        diff = (ref - bg_color).abs().mean(dim=-1)  # H, W
        content_mask = diff > bg_threshold

        col_frac = content_mask.float().mean(dim=0)  # W
        col_has_content = (col_frac > min_col_fraction).tolist()

        runs = []
        start = None
        for x in range(W):
            if col_has_content[x] and start is None:
                start = x
            elif not col_has_content[x] and start is not None:
                runs.append((start, x - 1))
                start = None
        if start is not None:
            runs.append((start, W - 1))

        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] - 1 <= merge_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        merged = [(a, b) for a, b in merged if (b - a + 1) >= min_width]

        if len(merged) > num_views:
            merged = sorted(merged, key=lambda r: -(r[1] - r[0]))[:num_views]
            merged = sorted(merged, key=lambda r: r[0])
        elif len(merged) > MAX_VIEWS:
            merged = merged[:MAX_VIEWS]

        info_lines = [f"image {W}x{H}, bg_color~{[round(float(c),3) for c in bg_color.tolist()]}, "
                       f"detected {len(merged)} view(s) (asked for {num_views})"]

        outputs = []
        for i in range(MAX_VIEWS):
            if i < len(merged):
                a, b = merged[i]
                x0 = max(0, a - padding)
                x1 = min(W, b + 1 + padding)
                if tight_vertical:
                    sub = content_mask[:, a:b + 1]
                    rows_frac = sub.float().mean(dim=1)
                    rows_has = (rows_frac > min_col_fraction).nonzero(as_tuple=True)[0]
                    if len(rows_has) > 0:
                        y0 = max(0, int(rows_has.min()) - padding)
                        y1 = min(H, int(rows_has.max()) + 1 + padding)
                    else:
                        y0, y1 = 0, H
                else:
                    y0, y1 = 0, H
                crop = image[:, y0:y1, x0:x1, :]
                if pad_to_square:
                    ch, cw = crop.shape[1], crop.shape[2]
                    side = max(ch, cw)
                    canvas = bg_color.view(1, 1, 1, -1).expand(crop.shape[0], side, side, crop.shape[-1]).clone()
                    oy, ox = (side - ch) // 2, (side - cw) // 2
                    canvas[:, oy:oy + ch, ox:ox + cw, :] = crop
                    crop = canvas
                    info_lines.append(f"  view_{i+1}: x={x0}, y={y0}, width={x1-x0}, height={y1-y0} -> padded to {side}x{side}")
                else:
                    info_lines.append(f"  view_{i+1}: x={x0}, y={y0}, width={x1-x0}, height={y1-y0}")
                outputs.append(crop)
            else:
                outputs.append(torch.zeros((1, 8, 8, image.shape[-1]), dtype=image.dtype, device=image.device))

        info = "\n".join(info_lines)
        return tuple(outputs) + (info,)


NODE_CLASS_MAPPINGS = {
    "AutoSplitTurnaroundSheet": AutoSplitTurnaroundSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoSplitTurnaroundSheet": "Auto-Split Turnaround Sheet",
}
