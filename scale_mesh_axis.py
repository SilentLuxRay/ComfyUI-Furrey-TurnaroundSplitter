import torch
import copy


class ScaleMeshAxis:
    """Scales a mesh's vertex positions independently per axis (X/Y/Z), around a
    chosen pivot. Use it to correct a squashed/stretched axis coming out of a 3D
    generator, by imposing the proportion you want directly in the workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.01}),
                "scale_z": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.01}),
                "pivot": (["bbox_center", "origin"], {"default": "bbox_center",
                           "tooltip": "bbox_center: scale around the mesh's own bounding-box center. "
                                      "origin: scale around world (0,0,0) -- use if the mesh is already "
                                      "positioned intentionally and you don't want it re-centered."}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "scale"
    CATEGORY = "3d/mesh"

    def scale(self, mesh, scale_x, scale_y, scale_z, pivot):
        out = copy.copy(mesh)
        v = out.vertices.clone()  # (B, N, 3)
        scale = torch.tensor([scale_x, scale_y, scale_z], dtype=v.dtype, device=v.device)

        counts = getattr(mesh, "vertex_counts", None)
        for b in range(v.shape[0]):
            n = int(counts[b]) if counts is not None else v.shape[1]
            vb = v[b, :n]
            if pivot == "origin":
                p = torch.zeros(3, dtype=v.dtype, device=v.device)
            else:
                mn = vb.min(dim=0).values
                mx = vb.max(dim=0).values
                p = (mn + mx) / 2
            v[b, :n] = (vb - p) * scale + p

        out.vertices = v
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ScaleMeshAxis": ScaleMeshAxis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleMeshAxis": "Scale Mesh (per axis)",
}
