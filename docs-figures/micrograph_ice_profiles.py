"""
Generate the ice-profile figure for docs/user-guide/micrograph.md.

Calls `IceProfile.thickness` directly, the same method
`build_ice_profile`/`MicrographGenerator` use, so the figure cannot drift
from what the profile modes actually compute.

Run with: uv run python docs-figures/micrograph_ice_profiles.py
Saves PNGs directly into docs/assets/images/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specter.ice import IceProfile

OUT_DIR = "docs/assets/images"
NXY = 256
PIXEL_SIZE = 8.0  # Å/pixel -> ~2048 Å field, a typical micrograph_size crop of a hole


def main() -> None:
    profiles = {
        "flat": IceProfile(mode="flat", mean_thickness=500.0),
        "wedge": IceProfile(mode="wedge", thickness_range=(250.0, 900.0), angle=30.0),
        "meniscus": IceProfile(
            mode="meniscus",
            hole_radius=6000.0,
            rim_thickness=1500.0,
            hole_offset=(4500.0, 0.0),
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    vmax = max(p.thickness(NXY, PIXEL_SIZE).max().item() for p in profiles.values())
    im = None
    for ax, (name, profile) in zip(axes, profiles.items()):
        field = profile.thickness(NXY, PIXEL_SIZE).numpy()
        im = ax.imshow(field, cmap="gray_r", vmin=0.0, vmax=vmax, origin="lower")
        ax.set_title(f"{name}\n{field.min():.0f}–{field.max():.0f} Å")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("ice thickness (Å)")
    fig.suptitle("MicrographConfig ice_profile modes (same field of view)")
    fig.savefig(f"{OUT_DIR}/micrograph-ice-profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/micrograph-ice-profiles.png")


if __name__ == "__main__":
    main()
