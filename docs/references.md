# References

SPECTER implements physics, parameterizations, measurements and algorithms
published by others. Each Theory, walkthrough and tutorial page cites the
work it draws on, in the text and in a References section at its end; this
page collects them, with links back to the pages that cite each one.

## Software

SPECTER is built on the libraries below, and follows, ports or reads the
output of the scientific software after them. Each entry names what SPECTER
uses it for.

### Libraries

- **Biopython**: parsing PDB and mmCIF structures. Cock, P. J. A., Antao, T.,
  Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., et al. (2009).
  Biopython: freely available Python tools for computational molecular
  biology and bioinformatics. *Bioinformatics*, 25(11), 1422–1423.
  [doi:10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163)
- **connected-components-3d**: 26-connected labelling of the space between
  membranes into cytosol and lumen in `specter build tomogram`.
  [github.com/seung-lab/connected-components-3d](https://github.com/seung-lab/connected-components-3d)
- **cryosparc-tools**: reading and writing CryoSPARC `.cs` files.
  [github.com/cryoem-uoft/cryosparc-tools](https://github.com/cryoem-uoft/cryosparc-tools)
- **CuPy**: the GPU Euclidean distance transform of the spherical-harmonics
  membrane backend. Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C.
  (2017). CuPy: A NumPy-compatible library for NVIDIA GPU calculations.
  *Workshop on Machine Learning Systems (LearningSys), Neural Information
  Processing Systems (NIPS)*.
  [PDF](http://learningsys.org/nips17/assets/papers/paper_16.pdf)
- **gemmi**: bonded-species atom typing against the Monomer Library, the
  per-element Peng scattering-factor tables, and the density-calculator
  potential builder. Wojdyr, M. (2022). GEMMI: A library for structural
  biology. *Journal of Open Source Software*, 7(73), 4200.
  [doi:10.21105/joss.04200](https://doi.org/10.21105/joss.04200)
- **Lightning**: the training loop and multi-GPU execution of Ghostbuster.
  [github.com/Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning)
- **Matplotlib**: every figure SPECTER draws. Hunter, J. D. (2007).
  Matplotlib: A 2D graphics environment. *Computing in Science &
  Engineering*, 9(3), 90–95.
  [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
- **mrcfile**: reading and writing MRC volumes and image stacks. Burnley,
  T., Palmer, C. M., & Winn, M. (2017). Recent developments in the CCP-EM
  software suite. *Acta Crystallographica Section D: Structural Biology*,
  73(6), 469–477.
  [doi:10.1107/S2059798317007859](https://doi.org/10.1107/S2059798317007859)
- **NumPy**: host-side arrays and the `SeedSequence` behind per-particle
  seeds. Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R.,
  Virtanen, P., Cournapeau, D., et al. (2020). Array programming with NumPy.
  *Nature*, 585(7825), 357–362.
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- **PyTorch**: every array operation of the forward model and the
  reconstruction, on GPU or CPU, and the automatic differentiation that
  makes both differentiable. Paszke, A., Gross, S., Massa, F., Lerer, A.,
  Bradbury, J., Chanan, G., et al. (2019). PyTorch: An imperative style,
  high-performance deep learning library. *Advances in Neural Information
  Processing Systems*, 32.
  [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)
- **RoMa**: conversions between quaternions, rotation vectors, Euler angles
  and rotation matrices, and uniform random rotations. Brégier, R. (2021). Deep regression on manifolds: a
  3D rotation case study. *2021 International Conference on 3D Vision
  (3DV)*, 166–174.
  [doi:10.1109/3DV53792.2021.00027](https://doi.org/10.1109/3DV53792.2021.00027)
- **SciPy**: physical constants, KD-tree neighbour searches and the CPU
  distance-transform fallback. Virtanen, P., Gommers, R., Oliphant, T. E.,
  Haberland, M., Reddy, T., Cournapeau, D., et al. (2020). SciPy 1.0:
  fundamental algorithms for scientific computing in Python. *Nature
  Methods*, 17(3), 261–272.
  [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
- **starfile**: reading and writing RELION `.star` files.
  [github.com/teamtomo/starfile](https://github.com/teamtomo/starfile)
- **torch-ctf** (teamtomo): the transfer function of the second, Python-only
  CTF backend in `specter.ctf`.
  [github.com/teamtomo/teamtomo](https://github.com/teamtomo/teamtomo/tree/main/packages/primitives/torch-ctf)
- **vesin**: neighbour lists for the coarse-grained water potential of the
  ice optimiser and for the packer's conflict check. Bigi, F., Abbott, J. W.,
  Loche, P., Mazitov, A., Tisi, D., Langer, M. F., et al. (2026). metatensor
  and metatomic: Foundational libraries for interoperable atomistic machine
  learning. *The Journal of Chemical Physics*, 164(6), 064113.
  [doi:10.1063/5.0304911](https://doi.org/10.1063/5.0304911);
  [github.com/Luthaf/vesin](https://github.com/Luthaf/vesin)

### Scientific software

- **abTEM**: the per-image `SeedSequence` seeding SPECTER adopted, and the
  weighted defocus ensemble its partial-coherence sum was checked against.
  Madsen, J., & Susi, T. (2021). The abTEM code: transmission electron
  microscopy from first principles. *Open Research Europe*, 1, 24.
  [doi:10.12688/openreseurope.13015.2](https://doi.org/10.12688/openreseurope.13015.2)
- **AreTomo3**: SPECTER reads its tilt-series alignments (`.aln`, `.xf`).
  Zheng, S., Wolff, G., Greenan, G., Chen, Z., Faas, F. G. A., Bárcena, M.,
  Koster, A. J., Cheng, Y., & Agard, D. A. (2022). AreTomo: An integrated
  software package for automated marker-free, motion-corrected cryo-electron
  tomographic alignment and reconstruction. *Journal of Structural Biology:
  X*, 6, 100068.
  [doi:10.1016/j.yjsbx.2022.100068](https://doi.org/10.1016/j.yjsbx.2022.100068);
  [github.com/czimaginginstitute/AreTomo3](https://github.com/czimaginginstitute/AreTomo3)
- **cisTEM simulator**: the exposure-resolved, frozen-plasmon zero-loss
  model. Himes, B., & Grigorieff, N. (2021). Cryo-TEM simulations of
  amorphous radiation-sensitive samples using multislice wave propagation.
  *IUCrJ*, 8(6), 943–953.
  [doi:10.1107/S2052252521008538](https://doi.org/10.1107/S2052252521008538)
- **CryoSPARC**: SPECTER reads and writes its `.cs` particle files and
  follows its pose and transfer-function conventions. Punjani, A.,
  Rubinstein, J. L., Fleet, D. J., & Brubaker, M. A. (2017). cryoSPARC:
  algorithms for rapid unsupervised cryo-EM structure determination.
  *Nature Methods*, 14(3), 290–296.
  [doi:10.1038/nmeth.4169](https://doi.org/10.1038/nmeth.4169)
- **CryoTomoSim (CTS)**: the gold fiducial beads (`gen_beads.m`) and the
  carbon support film (`gen_carbon.m`) are ported from it. Purnell, C.,
  Heebner, J., Swulius, M. T., Hylton, R., Kabonick, S., Grillo, M.,
  Grigoryev, S., Heberle, F., Waxham, M. N., & Swulius, M. T. (2023). Rapid
  synthesis of cryo-ET data for training deep learning models. *bioRxiv*.
  [doi:10.1101/2023.04.28.538636](https://doi.org/10.1101/2023.04.28.538636);
  [github.com/carsonpurnell/cryotomosim_CTS](https://github.com/carsonpurnell/cryotomosim_CTS)
- **InSilicoTEM**: the absorption model built on the inelastic mean free
  path. Vulović, M., Ravelli, R. B. G., van Vliet, L. J., Koster, A. J.,
  Lazić, I., Lücken, U., Rullgård, H., Öktem, O., & Rieger, B. (2013). Image
  formation modeling in cryo-electron microscopy. *Journal of Structural
  Biology*, 183(1), 19–32.
  [doi:10.1016/j.jsb.2013.05.008](https://doi.org/10.1016/j.jsb.2013.05.008)
- **PolNet**: the transmembrane-protein placement construction.
  Martinez-Sanchez, A., Lamm, L., Jasnin, M., & Phelippeau, H. (2024).
  Simulating the cellular context in synthetic datasets for cryo-electron
  tomography. *IEEE Transactions on Medical Imaging*, 43(11), 3742–3754.
  [doi:10.1109/TMI.2024.3398401](https://doi.org/10.1109/TMI.2024.3398401);
  [github.com/anmartinezs/polnet](https://github.com/anmartinezs/polnet)
- **qscore** (K. Jamali): the Python implementation of Q-scores that
  `specter.qscore` follows. Pintilie, G., Zhang, K., Su, Z., Li, S., Schmid,
  M. F., & Chiu, W. (2020). Measurement of atom resolvability in cryo-EM maps
  with Q-scores. *Nature Methods*, 17(3), 328–334.
  [doi:10.1038/s41592-020-0731-1](https://doi.org/10.1038/s41592-020-0731-1);
  [github.com/jamaliki/qscore](https://github.com/jamaliki/qscore)
- **RELION**: SPECTER reads and writes its `.star` files and follows its
  Euler-angle and symmetry conventions. Scheres, S. H. W. (2012). RELION:
  Implementation of a Bayesian approach to cryo-EM structure determination.
  *Journal of Structural Biology*, 180(3), 519–530.
  [doi:10.1016/j.jsb.2012.09.006](https://doi.org/10.1016/j.jsb.2012.09.006)
- **sffit**: the reference implementation of the bonded-species (Shtyrov)
  scattering factors, whose atom-typing rules SPECTER follows. Shtyrov, A.,
  Wilson, H., Slowik, D., Yamashita, K., Li, J., Wojdyr, M., et al. (2026).
  Measurement of atomic scattering factors by cryoelectron microscopy.
  *Proceedings of the National Academy of Sciences*, 123(19), e2528758123.
  [doi:10.1073/pnas.2528758123](https://doi.org/10.1073/pnas.2528758123);
  [github.com/as2875/sffit](https://github.com/as2875/sffit)
- **torch-fourier-filter** (teamtomo): the Fourier-space envelope functions
  in `specter.aberrations` are ported from its `envelopes` module.
  [github.com/teamtomo/teamtomo](https://github.com/teamtomo/teamtomo/tree/main/packages/primitives/torch-fourier-filter)

## Literature

--8<-- "docs-includes/bibliography.md"
