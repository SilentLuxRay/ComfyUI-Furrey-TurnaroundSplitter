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
                "padding": ("INT", {"default": 12, "min": 0, "max": 512,
                                     "tooltip": "Fixed margin in pixels. Ignored if margin_percent > 0."}),
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
                "margin_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                              "tooltip": "Margin as a percentage of the view's own size, added on ALL sides "
                                                         "(including top/bottom, even outside the source sheet's own bounds) "
                                                         "-- self-scales to any resolution, unlike a fixed pixel padding. "
                                                         "0 = disabled, use 'padding' instead. Many 3D/conditioning pipelines "
                                                         "expect the subject to occupy ~90% of the frame (about 8-10% margin), "
                                                         "not edge-to-edge."}),
            }
        }

    RETURN_TYPES = tuple(["IMAGE"] * MAX_VIEWS) + ("STRING",)
    RETURN_NAMES = tuple(f"view_{i+1}" for i in range(MAX_VIEWS)) + ("info",)
    FUNCTION = "split"
    CATEGORY = "image/transform"

    def split(self, image, num_views, padding, bg_threshold, min_col_fraction,
              merge_gap, min_width, tight_vertical, pad_to_square=False, margin_percent=0.0):
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
                if tight_vertical:
                    sub = content_mask[:, a:b + 1]
                    rows_frac = sub.float().mean(dim=1)
                    rows_has = (rows_frac > min_col_fraction).nonzero(as_tuple=True)[0]
                    if len(rows_has) > 0:
                        cy0, cy1 = int(rows_has.min()), int(rows_has.max()) + 1
                    else:
                        cy0, cy1 = 0, H
                else:
                    cy0, cy1 = 0, H
                cx0, cx1 = a, b + 1
                content = image[:, cy0:cy1, cx0:cx1, :]  # the tight content, no margin yet
                ch, cw = content.shape[1], content.shape[2]

                if margin_percent > 0:
                    m = int(round(max(ch, cw) * margin_percent / 100.0))
                    mx = my = m
                else:
                    mx = my = padding

                out_h, out_w = ch + 2 * my, cw + 2 * mx
                if pad_to_square:
                    side = max(out_h, out_w)
                    out_h = out_w = side

                canvas = bg_color.view(1, 1, 1, -1).expand(content.shape[0], out_h, out_w, content.shape[-1]).clone()
                oy, ox = (out_h - ch) // 2, (out_w - cw) // 2
                canvas[:, oy:oy + ch, ox:ox + cw, :] = content
                crop = canvas
                info_lines.append(f"  view_{i+1}: content=({cx0},{cy0})-({cx1},{cy1}) size={cw}x{ch}, "
                                   f"margin={mx}px/{my}px -> output {out_w}x{out_h}")
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
