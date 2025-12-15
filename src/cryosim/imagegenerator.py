import lightning as L
import torch
from .microscope import Aberration, Detector
from .scattering import Scattering
from .icemaker import NaiveIcemaker, Icemaker
from . import rotations
from .rotations import Rotation
import torch.nn.functional as F
from .crowding import CrowdWithDuplicates
from .potential import PotentialBuilder
import torch.nn as nn
from cryosim.detectors import k3_300kv, k3_200kv, perfect_detector


class BaseImageGenerator(L.LightningModule):
    def __init__(
        self,
        pixel_size,
        energy,
        dose_per_angstrom,
        nxy=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        ice_model=None,
        ice_thickness=None,
        klim=None,
        flip_curvature=False,
        alpha=0.0,
        crowd_min_distance=None,
        crowd_max_distance_z=None,
        pad_fft=False,
        detector_model=None,
        anisomag=None,
        ctf_params=None,
        progressbars=True,
    ):
        super().__init__()
        self.pixel_size = pixel_size
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.ice_model = ice_model
        self.ice_thickness = ice_thickness
        self.klim = klim
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.crowd_min_distance = crowd_min_distance
        self.pad_fft = pad_fft
        self.nxy = nxy
        self.progressbars = progressbars

        if self.pad_fft and self.nxy is not None:
            self.pad_nxy = self.nxy + (self.nxy // 2) * 2
        elif self.nxy is not None:
            self.pad_nxy = self.nxy
        else:
            self.pad_nxy = None  # Must be set by subclass if nxy is not known yet

        self.detector_model = detector_model
        if detector_model is None:
            self.detector_mtf = None
        elif self.nxy is not None:
            self._init_detector_mtf()

        # Ice thickness logic
        if ice_model is None:
            self.nz = self.nxy if self.nxy else 0
        else:
            if ice_thickness is None:
                self.nz = self.nxy if self.nxy else 0
            else:
                if self.nxy and ice_thickness < self.nxy * pixel_size:
                    self.nz = self.nxy
                else:
                    self.nz = int(ice_thickness // pixel_size)

        if crowd_max_distance_z is None:
            crowd_max_distance_z = self.nz
        self.crowd_max_distance_z = crowd_max_distance_z

        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)

        # CTF Params
        if ctf_params is not None:
            self.ctf_params = nn.ParameterDict(
                {k: nn.Parameter(v) for k, v in ctf_params.items()}
            )
            if self.scattering_model not in ["projection", "ctf"]:
                self._shift_ctf_params()
        else:
            self.ctf_params = None

    def _init_detector_mtf(self):
        if self.detector_model == "k3_300kv":
            self.register_buffer("detector_mtf", k3_300kv(self.nxy, self.pixel_size))
        elif self.detector_model == "k3_200kv":
            self.register_buffer("detector_mtf", k3_200kv(self.nxy, self.pixel_size))
        elif self.detector_model == "perfect":
            self.register_buffer(
                "detector_mtf", perfect_detector(self.nxy, self.pixel_size)
            )

    def _shift_ctf_params(self):
        if "dfu" in self.ctf_params:
            shifted = self.ctf_params["dfu"].detach() - (self.nz * self.pixel_size) / 2
            self.ctf_params["dfu"] = nn.Parameter(shifted)
        if "dfv" in self.ctf_params:
            shifted = self.ctf_params["dfv"].detach() - (self.nz * self.pixel_size) / 2
            self.ctf_params["dfv"] = nn.Parameter(shifted)

    def _init_modules(self):
        # Should be called after pad_nxy and nz are finalized
        self.scattering = Scattering(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            self.dose_per_angstrom,
            scattering_model=self.scattering_model,
            klim=self.klim,
            flip_curvature=self.flip_curvature,
            nz=self.nz,
            alpha=self.alpha,
            progressbars=self.progressbars if hasattr(self, "progressbars") else False,
        )

        self.aberration = Aberration(
            self.pad_nxy,
            self.pixel_size,
            self.energy,
            aberration_model=self.aberration_model,
            alpha=self.alpha,
        )

        self.detector = Detector(
            self.pixel_size,
            self.dose_per_angstrom,
            aberration_model=self.aberration_model,
            noise_model=self.noise_model,
            mtf=self.detector_mtf,
        )

    def solvate(self, V):
        # generates ice with size (B x Z x Y x X)
        ice = self.icemaker.generate_ice(batchsize=len(V))

        if self.pad_fft:
            ice = F.pad(
                ice,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="reflect",
            )

        # pad V in z-axis if ice_thickness is not None
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis
                    0,
                    0,  # y-axis
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
            )
        icemask = V.detach().clone()
        icemask[icemask < 10] = 1
        icemask[icemask >= 10] = 0
        V = V + ice * icemask
        if hasattr(self, "icemask"):
            self.icemask = icemask.detach().cpu()
        return V

    def process_volume(self, V, idx):
        # adds crowd
        if self.crowd_min_distance is not None:
            with torch.no_grad():
                for i, v in enumerate(V):
                    vols = self.crowd()
                    if not isinstance(vols, float):
                        self.vols = vols.detach().cpu()
                    V[i] += vols

        # add ice
        if self.ice_model is not None:
            with torch.no_grad():
                V = self.solvate(V)

        # scatter V
        self.exitwaves = self.scattering(V)

        # aberrate exitwaves
        ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

        # image/noise
        if self.anisomag is None:
            images = self.detector(self.detector_waves, nxy=self.nxy)
        else:
            images = self.detector(self.detector_waves, self.anisomag[idx], self.nxy)
        return images

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def predict_epoch_end(self, outputs):
        # outputs is a list of batch predictions from THIS GPU
        preds = torch.cat(outputs, dim=0)

        # gather across all GPUs
        preds_all = self.trainer.strategy.all_gather(preds)

        # return only once on rank 0
        if self.trainer.is_global_zero:
            return preds_all.cpu()


class ImageGeneratorFromCoordinates(BaseImageGenerator):
    def __init__(
        self,
        coordinates,
        atomic_numbers,
        nxy,
        pixel_size,
        quaternions,
        translations,
        ctf_params,
        energy,
        dose_per_angstrom,
        anisomag=None,
        ice_model=None,
        ice_thickness=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        klim=None,
        flip_curvature=False,
        alpha=0.0,
        crowd_min_distance=None,
        crowd_max_distance_z=None,
        pad_fft=False,
        conv_backend="fftconvolve",
        detector_model=None,
    ):
        super().__init__(
            pixel_size=pixel_size,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            ice_model=ice_model,
            ice_thickness=ice_thickness,
            klim=klim,
            flip_curvature=flip_curvature,
            alpha=alpha,
            crowd_min_distance=crowd_min_distance,
            crowd_max_distance_z=crowd_max_distance_z,
            pad_fft=pad_fft,
            detector_model=detector_model,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=False,  # Default in original was implicit/not set, but used in others
        )

        # register buffers
        self.coordinates = nn.Parameter(coordinates)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        self.atomic_numbers = atomic_numbers

        # initialize modules
        self.potentialbuilder = PotentialBuilder(
            self.nxy,
            self.pixel_size,
            self.atomic_numbers,
            trainable=True,
            conv_backend=conv_backend,
        )
        self.atomic_numbers = (
            atomic_numbers  # Needed for potentialbuilder init but not stored in super
        )

        # Pre-calculate V for crowd initialization if needed
        self.V = self.potentialbuilder(self.coordinates.detach())

        self._init_modules()

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method="3d",
                n_points=torch.inf,
                seed="origin",
            )

        if ice_model is not None:
            if ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(n=self.nxy, dx=pixel_size, nz=self.nz)
            elif ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                )

    def rotate(self, Q, T):
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        r_coordinates = R.apply(self.coordinates, T=T)
        return r_coordinates

    def forward(self, idx):
        # rotate coordinates, returns (B x N x 3)
        coordinates = self.rotate(self.quaternions[idx], self.translations[idx])

        # sample coordinates to volume
        V = self.potentialbuilder(coordinates)

        # pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis, last dim
                    0,
                    0,  # y-axis, second last dim
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
                mode="constant",
            )
        # pad xy
        if self.pad_fft:
            V = F.pad(
                V,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="constant",
            )

        return self.process_volume(V, idx)


class ImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        scattering_potential,
        pixel_size,
        quaternions,
        translations,
        ctf_params,
        energy,
        dose_per_angstrom,
        anisomag=None,
        ice_model=None,
        ice_thickness=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        klim=None,
        flip_curvature=False,
        alpha=0.0,
        crowd_min_distance=None,
        crowd_max_distance_z=None,
        pad_fft=False,
        progressbars=True,
        parameterization="kirkland",
        detector_model=None,
    ):
        nxy = scattering_potential.shape[-1]
        super().__init__(
            pixel_size=pixel_size,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            ice_model=ice_model,
            ice_thickness=ice_thickness,
            klim=klim,
            flip_curvature=flip_curvature,
            alpha=alpha,
            crowd_min_distance=crowd_min_distance,
            crowd_max_distance_z=crowd_max_distance_z,
            pad_fft=pad_fft,
            detector_model=detector_model,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=progressbars,
        )

        self.parameterization = parameterization

        # register buffers
        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)

        self._init_modules()

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method="3d",
                n_points=torch.inf,
                seed="origin",
                progressbars=self.progressbars,
            )

        if ice_model is not None:
            if ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                    progressbars=self.progressbars,
                )
            elif ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                    progressbars=self.progressbars,
                    parameterization=parameterization,
                )

    def rotate(self, Q, T):
        if len(Q.shape) < 2:
            Q = Q.unsqueeze(0)
        if len(T.shape) < 2:
            T = T.unsqueeze(0)
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        theta = rotations.build_affine_matrix(R.as_matrix(), T)
        V = rotations.rotate_volume(self.V, theta, origin="relion")
        return V

    def forward(self, idx):
        # rotate V, returns (B x Z x Y x X)
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        # pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(
                V,
                (
                    0,
                    0,  # x-axis, last dim
                    0,
                    0,  # y-axis, second last dim
                    zpad_px // 2,
                    self.nz - zpad_px // 2 - V.shape[1],  # z-axis
                ),
                mode="constant",
            )
        # pad xy
        if self.pad_fft:
            V = F.pad(
                V,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="constant",
            )

        return self.process_volume(V, idx)


class MicrographGenerator(BaseImageGenerator):
    def __init__(
        self,
        scattering_potential,
        micrograph_size,
        pixel_size,
        ctf_params,
        energy,
        dose_per_angstrom,
        anisomag=None,
        ice_model=None,
        ice_thickness=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        klim=None,
        alpha=0.0,
        crowd_min_distance=None,
        crowd_max_distance_z=None,
        pad_fft=False,
        chunk_size=None,
        move_to_cpu=True,
        water_air_interface=True,
        detector_model=None,
    ):
        # Determine nxy
        if isinstance(micrograph_size, int):
            nxy = micrograph_size
        elif (
            isinstance(micrograph_size, (tuple, list))
            and micrograph_size[0] == micrograph_size[1]
        ):
            nxy = micrograph_size[0]
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")

        super().__init__(
            pixel_size=pixel_size,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            ice_model=ice_model,
            ice_thickness=ice_thickness,
            klim=klim,
            alpha=alpha,
            crowd_min_distance=crowd_min_distance,
            crowd_max_distance_z=crowd_max_distance_z,
            pad_fft=pad_fft,
            detector_model=detector_model,
            anisomag=anisomag,
            ctf_params=ctf_params,
        )

        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.water_air_interface = water_air_interface

        if ice_thickness is None or (
            ice_thickness < scattering_potential.shape[0] * pixel_size
        ):
            self.nz = scattering_potential.shape[0]
        else:
            self.nz = int(ice_thickness // pixel_size)
            self.ice_thickness = self.nz * pixel_size
            # Re-shift CTF params if nz changed
        if self.scattering_model not in ["projection", "ctf"]:
            self._shift_ctf_params()

        # Re-init crowd_max_distance_z if it was None (it defaults to self.nz in base)
        if crowd_max_distance_z is None:
            self.crowd_max_distance_z = self.nz

        self.register_buffer("V", scattering_potential)

        # Re-init modules with correct nz
        self._init_modules()

        self.crowd = CrowdWithDuplicates(
            scattering_potential,
            pixel_size,
            self.crowd_min_distance,
            nxy_out=self.pad_nxy if pad_fft else self.nxy,
            nz_out=self.nz,
            max_distance_z=self.crowd_max_distance_z,
            max_distance_xy=None,
            method="3d",
            n_points=torch.inf,
            seed="random",
            chunk_size=chunk_size,
            move_to_cpu=self.move_to_cpu,
            water_air_interface=water_air_interface,
        )

        if ice_model is not None:
            if ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(n=self.nxy, dx=pixel_size, nz=self.nz)
            elif ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=256,  # Original hardcoded 256?
                    dx=pixel_size,
                    nz=256,  # Original hardcoded 256?
                    chunk_size=self.chunk_size,
                )

    def solvate(self, V):
        # generates ice with size (B x Z x Y x X)
        # MicrographGenerator has specific solvate logic (generate_big_ice)
        self.ice = self.icemaker.generate_big_ice(V.shape)

        if self.pad_fft:
            self.ice = F.pad(
                self.ice,
                (
                    self.nxy // 2,
                    self.nxy // 2,  # x-axis, last dim
                    self.nxy // 2,
                    self.nxy // 2,  # y-axis, second last dim
                    0,
                    0,  # z-axis
                ),
                mode="reflect",
            )

        icemask = V < 10  # boolean mask, same shape, no copy of V
        V += self.ice * icemask
        # self.icemask = icemask #save as attribute just to check
        return V

    def forward(self, idx):
        # MicrographGenerator forward is different: starts with empty V
        V = torch.zeros(
            len(idx), self.nz, self.nxy, self.nxy
        )  # Changed from empty to zeros for safety

        # adds crowd
        if self.crowd_min_distance is not None:
            with torch.no_grad():
                for i, v in enumerate(V):
                    self.vols = self.crowd()
                    V[i] += self.vols
                    # V[i] += self.crowd()

        # add ice
        if self.ice_model is not None:
            with torch.no_grad():
                V = self.solvate(V)

        # scatter V
        self.exitwaves = self.scattering(V)

        # aberrate exitwaves
        ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

        # image/noise
        if self.anisomag is None:
            images = self.detector(
                self.detector_waves, nxy=None
            )  # nxy=None in original
        else:
            images = self.detector(self.detector_waves, self.anisomag[idx], nxy=None)
        return images


class TiltSeriesGenerator(MicrographGenerator):
    def __init__(
        self,
        scattering_potential,
        micrograph_size,
        pixel_size,
        ctf_params,
        energy,
        dose_per_angstrom,
        angles,
        sample_size=None,
        anisomag=None,
        ice_model=None,
        ice_thickness=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        klim=None,
        alpha=0.0,
        crowd_min_distance=None,
        crowd_max_distance_z=None,
        pad_fft=False,
        chunk_size=None,
        move_to_cpu=True,
        water_air_interface=True,
        detector_model=None,
    ):
        super().__init__(
            scattering_potential=scattering_potential,
            micrograph_size=micrograph_size,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            energy=energy,
            dose_per_angstrom=dose_per_angstrom,
            anisomag=anisomag,
            ice_model=ice_model,
            ice_thickness=ice_thickness,
            scattering_model=scattering_model,
            aberration_model=aberration_model,
            noise_model=noise_model,
            klim=klim,
            alpha=alpha,
            crowd_min_distance=crowd_min_distance,
            crowd_max_distance_z=crowd_max_distance_z,
            pad_fft=pad_fft,
            chunk_size=chunk_size,
            move_to_cpu=move_to_cpu,
            water_air_interface=water_air_interface,
            detector_model=detector_model,
        )

        # Store user preference
        self.user_sample_size = sample_size
        self.angles = angles

        # We don't determine self.sample_nxy here anymore because it depends on angles
        # unless user provided it explicitly.
        if sample_size is not None:
            if isinstance(sample_size, int):
                self.sample_nxy = sample_size
            elif (
                isinstance(sample_size, (tuple, list))
                and sample_size[0] == sample_size[1]
            ):
                self.sample_nxy = sample_size[0]
            else:
                raise ValueError("sample_size must have same dimensions in x and y.")

            if self.sample_nxy < self.nxy:
                raise ValueError("sample_size cannot be smaller than micrograph_size.")
        else:
            self.sample_nxy = None  # Will be calculated later

        self.generate_volume()

    def generate_volume(self):
        # Determine sample_nxy
        if self.user_sample_size is not None:
            sample_nxy = self.sample_nxy
        else:
            # Calculate required size based on max angle
            # S >= (W + T * sin(theta)) / cos(theta)
            # We assume rotation around X axis, so Y dimension is affected.
            # W = self.nxy * self.pixel_size
            # T = self.nz * self.pixel_size

            # Convert to tensor for calculation
            max_angle_deg = torch.max(torch.abs(self.angles))
            theta = max_angle_deg / 180 * torch.pi

            # Calculate required size in pixels
            # Note: The formula assumes we want to cover the full projected width.
            # S * cos(theta) >= W + T * sin(theta)
            # S >= (W + T * sin(theta)) / cos(theta)

            # Safety margin?
            # required_size = (W + T * torch.sin(theta)) / torch.cos(theta)

            # mine
            required_size = self.nxy / torch.cos(theta) + self.nz * torch.tan(theta)
            sample_nxy = int(torch.ceil(required_size))

            # Ensure even number for FFT efficiency (optional but good practice)
            if sample_nxy % 2 != 0:
                sample_nxy += 1

            print(
                f"Auto-calculated sample size: {sample_nxy} (Max angle: {max_angle_deg:.1f} deg)"
            )

        # 1. Generate the base volume V once (Frozen for the series)
        # Initialize V with sample_nxy
        V = torch.zeros(1, self.nz, sample_nxy, sample_nxy)

        # Add crowd
        if self.crowd_min_distance is not None:
            # We need to temporarily update self.nxy to sample_nxy for crowd generation?

            # Re-initialize crowd for the larger volume
            crowd_gen = CrowdWithDuplicates(
                self.V,
                self.pixel_size,
                self.crowd_min_distance,
                nxy_out=sample_nxy,  # Use sample size
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method="3d",
                n_points=torch.inf,
                seed="random",
                chunk_size=self.chunk_size,
                move_to_cpu=self.move_to_cpu,
                water_air_interface=self.water_air_interface,
            )

            with torch.no_grad():
                for i, v in enumerate(V):
                    vols = crowd_gen()
                    V[i] += vols

        # Add ice
        if self.ice_model is not None:
            if self.ice_model == "randomchoice":
                self.icemaker = NaiveIcemaker(
                    n=self.nxy, dx=self.pixel_size, nz=self.nz
                )
            elif self.ice_model == "iterative":
                self.icemaker = Icemaker(
                    n=256,  # Original hardcoded 256?
                    dx=self.pixel_size,
                    nz=256,  # Original hardcoded 256?
                    chunk_size=self.chunk_size,
                )

            with torch.no_grad():
                ice = self.icemaker.generate_big_ice(V.shape)
                icemask = V < 10
                V += ice * icemask

        # V is now our frozen volume (B, Z, sample_nxy, sample_nxy)
        self.vol = V.detach().cpu()

    def generate_tilt_series(self, idx):
        """
        Generate a tilt series for the given batch indices.

        Args:
            angles: list or tensor of tilt angles in degrees.
            idx: tensor of batch indices (to select CTF/Anisomag parameters).

        Returns:
            tilt_series: (B, N_angles, Y, X) tensor of images.
        """

        tilt_series = []
        for angle in self.angles:
            exitwaves = self.scattering.multislice_and_tilt(self.vol, angle)

            # 3. Aberration
            ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
            detector_waves = self.aberration(exitwaves, ctf_batch)

            # 4. Detection
            if self.anisomag is None:
                image = self.detector(detector_waves, nxy=None)
            else:
                image = self.detector(detector_waves, self.anisomag[idx], nxy=None)

            tilt_series.append(image)

        return torch.stack(tilt_series, dim=1)  # (B, N_angles, Y, X)
