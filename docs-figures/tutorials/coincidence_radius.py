"""
Figure and printed output for docs/tutorials/coincidence-radius.md.

Run with: uv run python docs-figures/tutorials/coincidence_radius.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("coincidence_radius")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt
import numpy as np
import torch

from specter.detectors import (
    EXCLUSION_RADIUS_PX,
    HARDWARE_FRAME_RATE_HZ,
    TYPICAL_DOSE_RATE_E_PX_S,
    coincidence_occupancy,
    coincidence_radius_for_simulation,
)
from specter.imaging import Exposure, Image

# The simulation: its own pixel size, dose and number of frames.
pixel_size, dose, n_frames = 1.5, 40.0, 40
exposure = Exposure(dose=dose, n_frames=n_frames)

# Uniform illumination: eight empty images at the vacuum intensity of 1.
flat = Image(
    torch.ones(8, 256, 256, device="cuda"),
    pixel_size=pixel_size,
    nxy=256,
    pad_nxy=256,
    ids=torch.arange(8),
)


def simulation_radius(camera: str, dose_rate: float) -> float:
    """The coincidence radius, in simulation pixels, for a camera's dose rate."""
    occupancy = coincidence_occupancy(
        exclusion_radius_px=EXCLUSION_RADIUS_PX[camera],
        dose_rate_e_px_s=dose_rate,  # electrons per physical pixel per second
        frame_rate_hz=HARDWARE_FRAME_RATE_HZ[camera],
    )
    return coincidence_radius_for_simulation(
        occupancy, dose_e_per_A2=dose, sim_pixel_size=pixel_size, sim_n_frames=n_frames
    )


rates = np.linspace(0.0, 20.0, 11)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for camera in EXCLUSION_RADIUS_PX:  # the cameras with a calibrated radius
    typical = TYPICAL_DOSE_RATE_E_PX_S[camera]
    radius = simulation_radius(camera, typical)
    print(f"{camera}: {typical:.0f} e⁻/px/s is a radius of {radius:.3f} px")

    counted = []
    for rate in rates:
        r = simulation_radius(camera, rate)
        counts = flat.detect(exposure, coincidence_radius=r, seed=0).array
        counted.append(float(counts.mean()) / (dose * pixel_size**2))
    (line,) = ax.plot(rates, counted, "o-", ms=3, label=camera)
    ax.axvline(typical, color=line.get_color(), lw=0.8, ls=":")

ax.set(xlabel="Dose rate (e⁻ per physical pixel per second)")
ax.set(ylabel="Fraction of electrons counted", ylim=(0.5, 1.02))
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "coincidence_radius")

lost = flat.detect(exposure, coincidence_radius=simulation_radius(camera, 20.0), seed=0)
thumbnail(as_numpy(lost.array[0])[:64, :64], "coincidence_radius", cmap="gray")

transcript.close()
