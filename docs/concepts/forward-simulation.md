# Forward simulation

Once a [specimen](specimens.md) volume \(V(x,y,z)\) exists, however it
was built, turning it into a simulated image is identical physics
regardless of whether \(V\) came from the single-particle or cryo-ET path.
Every `BaseImager` subclass (`ImageGenerator`, `MicrographGenerator`,
`TiltSeriesGenerator`, ...) runs the same three stages:

1. **[Scattering](scattering/index.md)**: propagate the electron wave
   through \(V\) (multislice, Rytov, first Born, or plain projection) to
   get an exit wave. Absorption enters here as the imaginary part of the
   potential: by default it is the real potential scaled by the
   amplitude-contrast ratio, and under `absorption_model="inelastic_mfp"`
   it is derived per material from measured inelastic mean free paths.
2. **[Aberrations](aberrations.md)**: apply the microscope's transfer
   function (defocus, spherical aberration, astigmatism, and the
   associated envelopes) to the exit wave. A generator constructed with
   `optics=None` skips this stage.
3. **[Detector](detector.md)**: model the physical detector's MTF, noise,
   and (for direct electron detectors) coincidence loss.

`TiltSeriesGenerator` runs this same chain once per tilt angle instead of
once per particle. The per-image physics stays fixed. Only the
invocation count and geometry change.

!!! info "Source"
    `specter.imagegenerator._base.BaseImager` is the shared base class
    all of the above inherit from.

## References

- Yeo, J., & Loh, N. D. (2026). Pursuing the physics of cryo-EM image
  formation. In *Current Approaches to Cryo-Electron Microscopy*,
  *Progress in Molecular Biology and Translational Science*. Elsevier.
  [doi:10.1016/bs.pmbts.2026.05.001](https://doi.org/10.1016/bs.pmbts.2026.05.001)
