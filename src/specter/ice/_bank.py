from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch

from ..progress import track
from ._ap import APIcemaker
from ._gradient import GradientSKIcemaker
from ._helpers import assemble_big_ice
from ._mcmc import MCMCIcemaker
from ._random import RandomIcemaker


class IceBank(L.LightningModule):
    """
    Standalone ice generator with a pre-built cache of unique cubes.

    The bank is built lazily on the first call to :meth:`generate_ice`, so that
    the underlying algorithm runs on whichever device the module has been moved to.
    Call :meth:`build` explicitly to force an eager build (e.g. after ``.to('cuda')``).

    Parameters
    ----------
    dx : float, optional
        Voxel size in Å. Default is 0.5.
    n : int, optional
        Number of voxels along x and y. Default is 256.
    nz : int, optional
        Number of voxels along z. Defaults to ``n`` (cubic box).
    method : {'gd', 'ap', 'mcmc', 'random'}, optional
        Ice generation algorithm:

        - ``'gd'`` — gradient descent via L-BFGS (:class:`GradientSKIcemaker`, default).
        - ``'ap'`` — alternating projections iterative Fourier amplitude matching
          (:class:`APIcemaker`).
        - ``'mcmc'`` — Metropolis Monte Carlo (:class:`MCMCIcemaker`,
          deprecated — prefer ``'gd'`` or ``'ap'``).
        - ``'random'`` — random molecule placement (:class:`RandomIcemaker`).
    num_unique : int, optional
        Number of unique cubes to cache when :meth:`build` is called. Default is 8.
    build_batch_size : int, optional
        Number of unique cubes to generate per call to the underlying generator
        when building the bank. Lower values reduce peak GPU memory. Default 1.
    **kwargs
        Forwarded to the underlying generator constructor. Use for method-specific
        construction parameters such as ``min_distance`` or ``parameterization``.

    Attributes
    ----------
    generator : GradientSKIcemaker or APIcemaker or MCMCIcemaker or RandomIcemaker
        The underlying generator instance. Access it directly for method-specific
        setup (e.g. ``bank.generator.set_target_gr_from_md(...)`` for MCMC).

    Examples
    --------
    >>> bank = IceBank(dx=0.5, n=256, method='gd', num_unique=8)
    >>> bank.to('cuda')
    >>> bank.build()                                    # runs on CUDA
    >>> ice = bank.generate_ice(batchsize=4)            # fast — augmented from bank

    MCMC requires setting the target g(r) before building:

    >>> bank = IceBank(dx=0.5, n=256, method='mcmc', num_unique=8)
    >>> bank.generator.set_target_gr_from_md('sim.lammpstrj')
    >>> bank.to('cuda')
    >>> bank.build(n_sweeps=200)
    """

    _METHOD_MAP: dict[str, type] = {
        "gd": GradientSKIcemaker,
        "ap": APIcemaker,
        "mcmc": MCMCIcemaker,
        "random": RandomIcemaker,
    }
    _METHOD_LABELS: dict[str, str] = {
        "gd": "gradient descent",
        "ap": "Alternating Projections",
        "mcmc": "Monte Carlo Markov Chain",
        "random": "random",
    }

    def __init__(
        self,
        dx: float = 0.5,
        n: int = 256,
        nz: Optional[int] = None,
        method: str = "gd",
        num_unique: int = 8,
        build_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if method not in self._METHOD_MAP:
            raise ValueError(
                f"method must be one of {list(self._METHOD_MAP)}, got '{method}'"
            )
        self.dx = dx
        self.n = n
        self.nz = nz if nz is not None else n
        self.method = method
        self._num_unique = num_unique
        if build_batch_size < 1:
            raise ValueError("build_batch_size must be >= 1")
        self.build_batch_size = build_batch_size

        generator_cls = self._METHOD_MAP[method]
        # Always build cubic blocks — non-cubic volumes are assembled via generate_big_ice
        self.generator = generator_cls(n=n, dx=dx, **kwargs)
        self.register_buffer("_bank", None, persistent=False)

    def allocate_placeholder(self, num_unique: int | None = None) -> None:
        """
        Allocate a zero-filled bank of the correct shape without running generation.

        Lets a DDP rank that shouldn't do the (expensive) generation work still
        hold a buffer of the right shape, to be filled in by Lightning's
        automatic module-state sync (``_sync_module_states``) from rank 0.

        Parameters
        ----------
        num_unique : int, optional
            Number of unique cubes. Defaults to the value passed to ``__init__``.
        """
        if num_unique is None:
            num_unique = self._num_unique
        self._bank = torch.zeros(
            num_unique, self.nz, self.n, self.n, device=self.device
        )

    # Validated default for method='gd': matches real LDA-80K MD ice
    # (see dev/ice/get_md_reference.py -- E_per_atom is stable to +/-0.0001
    # across widely-separated MD frames) rather than the previous
    # rep_strength=1.0-only default, which converges S(k) fine but leaves
    # badly-overlapping local structure (S(k) matching alone is a weak
    # constraint on local atomic arrangement -- confirmed across every
    # resolution tested this session, from n=16 up through the n=256,
    # dx=1.0 production default). mlbop_strength=0.5 was the best balance
    # of energy-match quality vs. S(k) fidelity found across dx=0.5/1.0/2.0
    # and box sizes up to 256 Å.
    #
    # n_steps=400 replaces the old default of 50, which is nowhere near
    # enough: at n=256,dx=1.0 (~527k atoms), sk_loss is still 0.58 (vs
    # ~0.0002-0.0008 by step 200-400) after only 50 steps. 400 was chosen
    # since sk_loss's practical gains taper off by ~200-300 steps at this
    # scale (~2.2s/step here, so ~15 min); E_per_atom keeps improving more
    # slowly out to 600+ steps for large systems specifically (~527k atoms:
    # -0.11 at 591 steps vs -0.26 to -0.35 for smaller ~31k-atom boxes in
    # the same budget) -- pass n_steps explicitly for more/less than this
    # default trade-off. tol=1e-4/patience=10 (optimize()'s own defaults)
    # still apply on top of this, so smaller/easier configs typically stop
    # well short of the ceiling.
    _GD_DEFAULT_KWARGS: dict[str, Any] = {
        "rep_strength": 0.0,
        "mlbop_strength": 0.5,
        "mlbop_target": -0.413,
        "n_steps": 400,
    }

    def build(self, num_unique: int | None = None, **kwargs: Any) -> None:
        """
        Generate and cache unique ice cubes on the current device.

        Parameters
        ----------
        num_unique : int, optional
            Number of unique cubes to generate. Defaults to the value passed to
            ``__init__`` (default 8).
        **kwargs
            Forwarded to ``generator.generate_ice(batchsize=num_unique, **kwargs)``.
            Method-specific build parameters:

            - ``gd``: ``n_steps``, ``lr``, ``rep_strength``, ``mlbop_strength``,
              ``mlbop_target``. If none of the latter three are given, defaults
              to matching real amorphous ice via ML-BOP energy (see
              :attr:`_GD_DEFAULT_KWARGS`) rather than the geometric-only
              ``rep_strength=1.0`` penalty; pass ``rep_strength=1.0,
              mlbop_strength=0.0`` explicitly to opt back into the old
              behavior.
            - ``ap``: ``reduce_fraction``
            - ``mcmc``: ``n_sweeps``, ``step_size``
        """
        if num_unique is None:
            num_unique = self._num_unique
        if self.method == "gd":
            kwargs = {**self._GD_DEFAULT_KWARGS, **kwargs}
        cubes: list[torch.Tensor] = []
        for start in track(
            range(0, num_unique, self.build_batch_size),
            description=f"Building ice cubes (method: {self._METHOD_LABELS[self.method]})",
            total=(num_unique + self.build_batch_size - 1) // self.build_batch_size,
            transient=True,
        ):
            batchsize = min(self.build_batch_size, num_unique - start)
            cubes.append(
                self.generator.generate_ice(batchsize=batchsize, **kwargs).cpu()
            )
        self._bank = torch.cat(cubes, dim=0)

    def generate_ice(self, batchsize: int = 1, **kwargs: Any) -> torch.Tensor:
        """
        Return ``batchsize`` ice cubes on the current device.

        Lazily calls :meth:`build` on the first invocation so the algorithm runs
        on whichever device the module has been placed on. All subsequent calls draw
        from the cached bank (which is stored on CPU) and move the output to device.

        Parameters
        ----------
        batchsize : int, optional
            Number of cubes to return. Default is 1.
        **kwargs
            Forwarded to :meth:`build` only on the first lazy call.

        Returns
        -------
        ice : torch.Tensor
            Shape ``(batchsize, nz, n, n)`` on the module's current device.
        """
        if self._bank is None:
            self.build(**kwargs)
        assert self._bank is not None
        _, D, H, W = self._bank.shape
        return assemble_big_ice(
            self._bank, (batchsize, D, H, W), replace_faces=False
        ).to(self.device)

    def generate_big_ice(
        self,
        target_shape: tuple[int, int, int, int],
        num_unique: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Tile ice into a large volume, drawing from the bank if one exists.

        If :meth:`build` has been called, tiles are drawn and augmented from the
        cached bank. If no bank exists, generates ``num_unique`` fresh cubes and
        tiles them, forwarding ``**kwargs`` to the generator.

        Parameters
        ----------
        target_shape : tuple of int
            Output shape ``(B, nz, ny, nx)``.
        num_unique : int, optional
            Number of unique cubes to generate on-the-fly when no bank exists.
            Defaults to the value passed to ``__init__`` (default 8). Ignored
            when a bank is present.
        **kwargs
            Forwarded to ``generator.generate_ice`` only when no bank exists.

        Returns
        -------
        big_ice : torch.Tensor
            Tiled ice volume of shape ``target_shape``.
        """
        if self._bank is None:
            if num_unique is None:
                num_unique = self._num_unique
            self.build(num_unique=num_unique, **kwargs)
        assert self._bank is not None
        return assemble_big_ice(self._bank, target_shape, replace_faces=True)
