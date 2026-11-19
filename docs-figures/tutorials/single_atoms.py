"""
Figure and printed output for docs/tutorials/single-atoms.md.

Run with: uv run python docs-figures/tutorials/single_atoms.py
"""

from __future__ import annotations

from _common import Transcript, as_numpy, save, thumbnail

transcript = Transcript("single_atoms")

transcript.start("recipe")
# --8<-- [start:recipe]
import matplotlib.pyplot as plt
import torch

from specter.atom import atom_number, kirkland_atomic_potential_2d
from specter.constants import energy_to_wavelength, interaction_parameter
from specter.fft import fft2, ifft2

# Five single atoms in a row, 10 Å apart, at 200 kV.
elements = ["C", "Si", "Cu", "Au", "U"]
centres = [5.0, 15.0, 25.0, 35.0, 45.0]  # Å
voltage = 200.0  # kV
wavelength = energy_to_wavelength(voltage)  # Å
sigma = interaction_parameter(voltage)  # 1/(V Å)
print(f"wavelength {wavelength * 100:.3f} pm, sigma {sigma * 1e3:.4f} rad/(kV Å)")

n = 512
x = torch.linspace(0, 50, n)
dx = float(x[1] - x[0])
y = torch.linspace(-25, 25, n) - dx / 2
X, Y = torch.meshgrid(x, y, indexing="xy")

# Projected potential (V Å) and the transmission function it gives.
projected = torch.zeros(n, n)
for element, centre in zip(elements, centres):
    radius = torch.sqrt((X - centre) ** 2 + Y**2)
    projected += kirkland_atomic_potential_2d(int(atom_number(element)), radius)
transmission = torch.exp(1j * sigma * projected)

# Coherent bright-field image at the Scherzer condition.
cs, defocus = 1.3e7, 700.0  # Å
k = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
kx, ky = torch.meshgrid(k, k, indexing="ij")
k2 = kx**2 + ky**2
k_max = 1 / (0.64 * (cs * wavelength**3) ** 0.25)
aperture = (k2.sqrt() < k_max).to(transmission.dtype)
chi = torch.pi * wavelength * k2 * (0.5 * cs * wavelength**2 * k2 - defocus)
wave = ifft2(
    fft2(transmission, shift=True) * torch.exp(-1j * chi) * aperture, shift=True
)
image = wave.abs() ** 2
print(f"aperture {k_max * wavelength * 1e3:.2f} mrad, resolution {1 / k_max:.2f} Å")

row = n // 2  # the line through the atom centres
for element, centre in zip(elements, centres):
    column = int(round(centre / dx))
    print(f"{element:>2}: image intensity at the centre {image[row, column]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].plot(x, transmission.real[row], label="real")
axes[0].plot(x, transmission.imag[row], label="imaginary")
axes[0].set(xlabel="x (Å)", title="transmission function")
axes[0].legend()
axes[1].imshow(image, cmap="gray", vmin=0.72, vmax=1.03, extent=(0, 50, -25, 25))
axes[1].set(xlabel="x (Å)", ylabel="y (Å)", title="bright-field image")
axes[2].plot(x, image[row])
axes[2].set(xlabel="x (Å)", ylim=(0.5, 1.1), title="image intensity along the row")
for element, centre in zip(elements, centres):
    axes[2].text(centre, 1.05, element, ha="center")
fig.tight_layout()
plt.show()
# --8<-- [end:recipe]
save(fig, "single_atoms")

thumbnail(as_numpy(image)[156:356, 156:356], "single_atoms", cmap="gray")

transcript.close()
