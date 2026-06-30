# Fourier envelope functions (ported from teamtomo)

## Goal

Port the math from teamtomo's `torch_fourier_filter/envelopes.py` (B-factor,
dose, Cs, Cc envelope functions) into specter, without depending on the
teamtomo package itself (it's not currently usable as a dependency). Wire
them into the existing CTF/holography transfer function pipeline so they can
be applied during forward simulation.

## Background

specter's `Aberration` class (`src/specter/aberrations/_aberration.py`)
already composes several aberration phase terms (defocus, Cs, beam tilt,
trefoil, phase shift) into a transfer function, and already applies one
envelope inline — an isotropic B-factor envelope via the `bfactor_envelope`
key in `ctf_params` (`_aberration.py:306-309`). The phase terms themselves
live as pure, stateless functions in `aberrations/_functions.py`, each
operating on precomputed frequency-grid tensors (`k`, `k2`, `kxx`, `kyy`,
`radian`) that `Aberration.__init__` registers as buffers.

teamtomo's `Cs_envelope` and `Cc_envelope` model attenuation of the transfer
function amplitude due to partial spatial and temporal coherence (finite
beam convergence angle, energy spread, voltage/current instability).
`dose_envelope` models the Grant & Grigorieff (2015) loss of high-resolution
signal with accumulated dose. None of these currently exist in specter.

`filters.py`'s `apply_bfactor` is a different thing — it blurs a 3D
real-space scattering potential volume to simulate atomic thermal motion
during potential generation. It is unrelated to image-formation-stage
envelopes and is not touched by this change.

## Design

### New file: `src/specter/aberrations/_envelopes.py`

Pure, stateless functions parallel in style to `_functions.py` — no internal
frequency-grid construction (unlike teamtomo's versions, which build their
own grid via `fftfreq_grid`); they take specter's precomputed `k`/`k2`
tensors and physical parameters already in specter's existing unit
conventions (wavelength in Å, cs/defocus in Å, matching `_functions.py`).

- `b_envelope(k2, bfactor) -> torch.Tensor`
  `exp(-bfactor * k2 / 4)`. Replaces the inline formula currently at
  `_aberration.py:309`.

- `cs_envelope(k, wavelength, cs, defocus, convergence_angle) -> torch.Tensor`
  Spatial-coherence envelope from finite beam convergence semi-angle
  (`convergence_angle`, in mrad — named to avoid collision with
  `Aberration.alpha`, which already means amplitude-contrast ratio).

- `cc_envelope(k2, wavelength, cc, voltage, energy_spread, deltaV_V, deltaI_I) -> torch.Tensor`
  Temporal-coherence envelope from chromatic aberration (`cc`, in Å, same
  unit convention as `cs`) combined with energy spread and voltage/current
  instability.

- `dose_envelope(k, dose, a=0.245, b=-1.665, c=2.81) -> torch.Tensor`
  Grant & Grigorieff (2015) cumulative-dose weighting. Returns all-ones when
  `dose < c` (matching teamtomo's branch).

Exported from `aberrations/__init__.py` alongside the existing phase
functions.

### Changes to `src/specter/aberrations/_aberration.py`

New optional constructor parameters on `Aberration` (all default to
off/`None`, so existing behavior and golden-fixture tests are unaffected):

- `convergence_angle: float | None = None` (mrad) — enables `cs_envelope`.
- `cc: float | None = None` (Å) — enables `cc_envelope`.
- `energy_spread: float = 0.7` (eV), `deltaV_V: float = 0.06e-6`,
  `deltaI_I: float = 0.01e-6` — further `cc_envelope` constants, only used
  when `cc` is set.
- `dose_envelope: bool = False` — enables the dose envelope.

In `transfer_function()`, after `transfer = torch.exp(-1j * chi)` is
computed:

- The existing `bfactor_envelope` handling is refactored to call
  `fn.b_envelope` instead of computing `exp(-bfactor * k2 / 4)` inline.
  Behavior is unchanged.
- If `self.convergence_angle is not None`: multiply `transfer` by
  `cs_envelope`, using the `cs`/`dfu`/`dfv` values already parsed earlier in
  the method (defaulting to 0 if absent, consistent with how the rest of
  the method treats missing keys).
- If `self.cc is not None`: multiply `transfer` by `cc_envelope`.
- If `self.dose_envelope` is `True` and `"dose" in ctf_params`: multiply
  `transfer` by `dose_envelope`.

### Change to `src/specter/imagegenerator/_base.py`

`BaseImager._ctf_batch()` currently merges `bfactor_envelope` into the
per-batch `ctf_params` dict but does not expose `dose_per_angstrom` (a
buffer already tracked for the `Detector`'s Poisson-noise scaling) to
`Aberration`. Add one line so `Aberration`'s `dose_envelope` can consume the
same per-image dose value without requiring it to be duplicated into
`ctf_params` at construction time:

```python
def _ctf_batch(self, idx):
    ctf_batch = {k: getattr(self, k)[idx] for k in self._ctf_param_names}
    if getattr(self, "bfactor_envelope", None) is not None:
        ctf_batch["bfactor_envelope"] = self.bfactor_envelope[idx]
    ctf_batch["dose"] = self.dose_per_angstrom[idx]
    return ctf_batch
```

This is harmless when `dose_envelope=False` (the Aberration model just
never reads the key).

## Testing

- New `tests/test_aberrations_envelopes.py` covering the four pure
  functions directly:
  - At `k=0` (or `k2=0`), every envelope equals 1 (no attenuation at DC).
  - Envelope values decrease monotonically with `k` for physically
    sensible positive inputs (bfactor, cs+defocus, cc, dose above
    threshold `c`).
  - `dose_envelope` returns all-ones when `dose < c`.
- Integration check on `Aberration`: constructing with all new params at
  their defaults (off) produces a transfer function bit-identical to
  current behavior; enabling each one individually attenuates high-`k`
  amplitude relative to the off case.
- No new error-handling paths: missing/absent params default to 0 or "off",
  consistent with `transfer_function()`'s existing convention.
- Existing golden-fixture regression tests in `tests/test_generators.py`
  are unaffected since every new parameter defaults to off.

## Out of scope

- `filters.py`'s `apply_bfactor` (3D potential volume blurring) is untouched
  — different physical purpose (atomic thermal motion vs. image-formation
  envelope), not refactored to share code with the new `b_envelope`.
- No changes to `ghostbuster.py` / reconstruction — these envelopes are
  forward-simulation-side only for now.
- teamtomo's package itself is not added as a dependency; only the math is
  ported, into specter's own units and conventions.
