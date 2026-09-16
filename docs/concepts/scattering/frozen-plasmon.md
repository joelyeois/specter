# Frozen-plasmon forward model

SPECTER now provides an **opt-in Python API** for exposure-resolved, Himes-inspired
zero-loss simulation. It uses production multislice, optics and detector code.
Existing generator calls retain their current model. The API requires explicit
material/source data rather than guessing atom identities from a real volume.

## Available components

- `specter.inelastic.PlasmonFilter`: integrate a supplied single-scattering EELS
  spectrum over 7.5–100 eV and take the square-root angular cross-section shape.
  `from_csv` requires a two-column `energy_eV,loss_density` file and provenance.
  `approximate_drude` is an explicitly named development substitute.
- `AtomicPlasmonSpecimen`: render particle atoms and explicit pseudo-water centres
  into paired elastic and absorption-source fields. Both fields use the same
  moving coordinates and solvent exclusion. Species coefficients are supplied
  in V Å³ per centre; they are never fitted to output images.
- `BrownianCoordinates`: seeded solvent motion, replayable for a fixed exposure
  schedule; MSD is in Å² per e⁻/Å². Changing the temporal grid changes the random
  realization, so convergence comparisons need a common trajectory or ensembles.
- `PotentialDoseDamage`: optional empirical damping of the particle's 3D elastic
  potential at the exposure midpoint. It preserves the DC term. It does not
  represent chemical damage or mass loss; it must not be combined with another
  dose envelope for the same damage.
- `FrozenPlasmonForward`: multislice each frozen state, apply optics, sum
  dose-weighted **intensities**, then detect each physical frame once.
- `BaseImager.simulate_frozen`: use an existing particle/micrograph/tilt imager's
  optics and camera with an explicit state provider and exposure schedule.

The new model is invoked through these Python objects. There is no automatic
`absorption_model="frozen_plasmon"` TOML switch: a real-valued volume alone does
not contain the required composition, source strengths or solvent trajectory.

## Minimal runnable example

This homogeneous slab checks normalization. Replace `slab` with an
`AtomicPlasmonSpecimen` for moving particles and water.

```python
import torch
from specter.inelastic import FrozenPlasmonForward, PlasmonFilter, PotentialState
from specter.microscope import Detector
from specter.potential import absorption_potential
from specter.scattering import IterativeScattering

n, nz, dx, voltage = 32, 40, 2.0, 300.0
elastic = torch.full((1, nz, n, n), 4.5)
source = torch.full_like(elastic, absorption_potential(3950.0, voltage))

def slab(step):
    return PotentialState(elastic, source)

# Explicit approximation. padding=0 describes a periodic transverse slab.
filter = PlasmonFilter.approximate_drude(dx, voltage, padding=0)
model = FrozenPlasmonForward(
    IterativeScattering(n, dx, voltage, progressbars=False),
    Detector(dx, noise_model="poisson", n_frames=1),
    filter,
)
result = model(slab, [1.0, 1.0, 1.0], substeps=2, detector_seed=17)
# (frame, batch, y, x); intensities are pre-detector, vacuum-normalized.
print(result.images.shape, result.intensities.mean(), result.metadata)
```

For an existing imager:

```python
result = imager.simulate_frozen(
    specimen, plasmon_filter, [1.0, 1.0, 1.0],
    idx=0, pose=0.0, substeps=4, detector_seed=17,
)
```

`specimen` owns all solvent, crowding, scaling and damage for this call. Cached
imager volumes are not reused. `pose` is explicit; the convenience method uses
one CTF entry for the entire movie. Use `FrozenPlasmonForward` with an optics
callback for per-frame defocus/aperture parameters. For a tilt acquisition,
pass one affine pose per physical frame in `poses` and perform the usual
beam-entry defocus correction in that callback. Exposure advances through the
acquisition order without resetting solvent motion between tilts.

## Normalization and boundaries

The filter is dimensionless and has unit DC. Source coefficients establish the
absorption magnitude. For uniform number density `n` (Å⁻³), MFP `L` (Å), and
interaction parameter `sigma`, an MFP-calibrated coefficient is
`1 / (2 * sigma * L * n)` V Å³ per scattering centre. Different species can use
measured relative cross-section weights. The result is MFP-calibrated, not an
independent prediction of the absolute inelastic cross-section.

Paired fields remain real while rotated. The transverse plasmon operator runs
**after beam-frame sampling**. Its halo fetches neighbouring source pixels
outside the propagation crop. Outside the supplied specimen is vacuum under
zero padding; reflection/border padding must be chosen deliberately for a
continuous material. Check source extent, halo size and propagation padding
separately. `padding=0` uses a periodic transverse convolution and is appropriate
only when that boundary condition is intended.

`IterativeScattering(..., alpha=0)` accepts `absorption_source=...` and
`absorption_filter=...`. With no filter, the paired source is an imaginary
potential directly, allowing an MFP comparison through identical operators.
Both fields retain gradients under tilt and checkpointing. Inputs have shape
`(B,Z,Y,X)` and matching real dtype/device. CPU volumes may stream to a GPU.

## Detector and reconstruction

`Detector.from_intensity` accepts the integrated expected intensity and applies
dose, pixel area, DQE and detector response once. Each supplied frame represents
a physical readout; numerical substeps do not create extra coincidence-loss
frames. Use a detector with `n_frames=1`, and apply external movie dose weights
after rendering. A fixed detector seed leaves the caller's RNG unchanged.

The shared forward is differentiable with `noise_model=None`, and returns
pre-detector intensities for fitting. Keep source priors and trajectories fixed
during optimization. Automatic fitting of a material decomposition or a latent
water trajectory in `TomogramReconstructor` is not implemented. Its existing
unsupported-model guards remain applicable.

## Validation and scope

`tests/test_frozen_plasmon.py` covers Beer–Lambert attenuation, eager/iterative
wave and gradient parity, tilted checkpointing, source halos, exact intensity
integration, detector dose accounting, seeded replay, atomic source strength,
solvent evolution, damage DC preservation and CPU/GPU agreement.

Run the same-particle integration comparison:

```sh
uv run python dev/inelastic_comparison/validate_production.py
```

Results, arrays and a left/right figure are written to
`dev/inelastic_comparison/production_validation/`. This 2 Å integration check
uses the approximate Drude spectrum and the prototype's MFP-calibrated species
weights. It does not supersede the earlier fine-grid numerical audit or prove
absolute experimental agreement.

The implementation supplies the algorithmic path. A publication-faithful
parameterization still requires measured EELS and empirical species strengths,
and a matched experimental/cisTEM benchmark. The model describes loss from the
zero-loss channel; it does not synthesize detected energy-loss electrons with
energy-dependent optics for unfiltered imaging.

Reference: [Himes & Grigorieff (2021), IUCrJ 8, 943–953](https://doi.org/10.1107/S2052252521008538).
