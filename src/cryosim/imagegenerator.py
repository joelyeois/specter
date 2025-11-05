import lightning as L
import numpy as np
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

class ImageGeneratorFromCoordinates(L.LightningModule):
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
        conv_backend='fftconvolve'
    ):
        """
        A particle image generator module to simulate 2D particles from the input
        coordinates and atomic numbers.

        Parameters
        ----------
        coordinates : (N,3) tensor
            The list of atomic coordinates.
        atomic_numbers : (N,) tensor
            The corresponding list of atomic numbers.
        nxy : int or tuple
            The desired size of the images.
        pixel_size: float
            Pixel size in angstroms. Assumes dz is also pixel_size for now.
        quaternions: 2D tensor
            Batch of quaternions with shape (N, 4). Follows scipy's convention of
            (x,y,z,w).
        translations: 2D tensor
            Batch of translations with shape (N, 2) in angstroms. translations[:, 0]
            x-translations whilst translations[:, 1] contrains y-translations in
            angstroms.
        ctf_params: 2D tensor
            Batch of CTF parameters with shape (N, 4), where the 4 columns are
            [Cs, defocusU, defocusV, defocusAngle], in angstroms and degrees.
        energy: float
            Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
        dose_per_angstrom: float
            Dose of the electron beam in e-/A^2.
        anisomag: None or 3D tensor
            Specifies 2x2 anisotropic matrix for each image.
        ice_model: str or None
            Specifies ice algorithm. Set to None for no ice. Options includes only
            'randomchoice' (fast) for now.
        ice_thickness: float
            Specifies the thickness of ice in Angstroms. Typically 100–1000 A. Must
            be same or larger than FOV of the particle.
        scattering_mode: str
            Specifies scattering model to use. Options include 'multislice',
            'firstborn', 'projection' and 'ctf', in order of increasing approximations.
        aberration_model: str
            Specifies aberration model to use. Options include 'holography' and 'ctf'.
        noise_model : str
            Specifies noise model. Currently only 'poisson' available.
        klim: float
            Kirkland [1] explains that setting klim = 0.66 is necessary to avoid
            aliasing for FFT methods (multislice and first Born). But this numerically
            lowers the spatial frequency information in the resultant exitwaves, so
            default is set to None.
        flip_curvature: bool
            This corresponds to positive/negative Ewald sphere curvature
            ambiguity. Set to False for positive, and True for negative (CryoSPARC).
            Only affects multislice and first Born models.
        alpha: float
            The amplitude contrast ratio to use for the CTF model. Common values
            are 0.07 and 0.1.
        crowd_min_distance: float, optional
            If not None, adds duplicate particles to simulate crowding at this
            distance in Angstroms around the particle.
        crowd_max_distance_z: float, optional
            Maximum distance in z to crowd. If None, defaults to ice_thickness which
            may overcrowd.
        pad_fft : bool
            Set to True to reduce border artifacts at the expense of memory and speed.
        conv_backend : str, optional
            Toggles the convolution method for sampling the coordinates into a 3D map.
            'conv3d' or 'fftconvolve',

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).
           [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
           Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.
        """
        super().__init__()

        # model params
        self.atomic_numbers = atomic_numbers
        self.nxy = nxy
        self.pixel_size = pixel_size
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.ice_model = ice_model
        self.ice_thickness = ice_thickness
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.klim = klim
        self.crowd_min_distance = crowd_min_distance
        self.pad_fft = pad_fft
        if self.pad_fft:
            self.pad_nxy = self.nxy + (self.nxy // 2) * 2 #
        else:
            self.pad_nxy = self.nxy

        # compute number of z-axis pixels due to ice thickness
        if ice_model is None:
            self.nz = self.nxy
        else:
            if ice_thickness is None:
                self.nz = self.nxy
            else:
                # thickness of ice must be at least the size of particle FOV.
                if ice_thickness < self.nxy * pixel_size:
                    self.nz = self.nxy
                else:
                    self.nz = int(ice_thickness // pixel_size)
        if crowd_max_distance_z is None:
            crowd_max_distance_z = self.nz
        self.crowd_max_distance_z = crowd_max_distance_z
            
        # register buffers
        self.coordinates = nn.Parameter(coordinates)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)
        
        # for dynamic/kinematic scattering, we need to account for the defocus
        # implicit in the scattering module
        self.ctf_params = nn.ParameterDict({
            k: nn.Parameter(v) for k, v in ctf_params.items()
        })
        if self.scattering_model not in ['projection', 'ctf']:
            if "dfu" in self.ctf_params:
                shifted = self.ctf_params["dfu"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfu"] = nn.Parameter(shifted)
            if "dfv" in self.ctf_params:
                shifted = self.ctf_params["dfv"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfv"] = nn.Parameter(shifted)

        # initialize modules
        self.potentialbuilder = PotentialBuilder(
            self.nxy,
            self.pixel_size,
            self.atomic_numbers,
            trainable=True,
            conv_backend='fftconvolve',
        )
        self.V = self.potentialbuilder(self.coordinates.detach())
        
        self.scattering = Scattering(
            self.pad_nxy,
            pixel_size,
            energy,
            dose_per_angstrom,
            scattering_model=scattering_model,
            klim=klim,
            flip_curvature=flip_curvature,
            nz=self.nz,
            alpha=alpha
        )

        self.aberration = Aberration(
            self.pad_nxy,
            pixel_size,
            energy,
            aberration_model=aberration_model,
            alpha=alpha,
        )

        self.detector = Detector(
            pixel_size,
            dose_per_angstrom,
            aberration_model=aberration_model,
            noise_model=noise_model
        )

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method='3d',
                n_points=torch.inf, seed='origin'
            )

        if ice_model is not None:
            if ice_model == 'randomchoice':
                self.icemaker = NaiveIcemaker(n=self.nxy,
                                              dx=pixel_size,
                                              nz=self.nz)
            elif ice_model == 'iterative':
                self.icemaker = Icemaker(n=self.nxy,
                                        dx=pixel_size,
                                        nz=self.nz,)

    def rotate(self, Q, T):
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        r_coordinates = R.apply(self.coordinates, T=T)
        return r_coordinates

    def solvate(self, V):
        # generates ice with size (B x Z x Y x X)
        ice = self.icemaker.generate_ice(batchsize=len(V))

        if self.pad_fft:
            ice = F.pad(ice,
                      (self.nxy // 2, self.nxy // 2,# x-axis, last dim
                       self.nxy // 2, self.nxy // 2,# y-axis, second last dim
                       0, 0,  # z-axis
                      ),
                     mode='reflect')

        # pad V in z-axis if ice_thickness is not None
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(V,
                      (0,0,# x-axis
                       0,0,# y-axis
                       zpad_px//2, self.nz - zpad_px//2 - V.shape[1],  # z-axis
                      ))
        icemask = V.detach().clone()
        icemask[icemask<0.01] = 1
        icemask[icemask>=0.01] = 0
        V = V + ice * icemask
        self.icemask = icemask.detach().cpu() #save as attribute just to check
        return V

    def forward(self, idx):
        #rotate coordinates, returns (B x N x 3)
        coordinates = self.rotate(self.quaternions[idx], self.translations[idx])

        #sample coordinates to volume
        V = self.potentialbuilder(coordinates)

        #pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(V,
                      (0, 0,# x-axis, last dim
                       0, 0,# y-axis, second last dim
                       zpad_px//2, self.nz - zpad_px//2 - V.shape[1],  # z-axis
                      ),
                      mode='constant')
        #pad xy
        if self.pad_fft:
            V = F.pad(V,
                      (self.nxy // 2, self.nxy // 2,# x-axis, last dim
                       self.nxy // 2, self.nxy // 2,# y-axis, second last dim
                       0, 0,  # z-axis
                      ),
                     mode='constant')

        #adds crowd
        if self.crowd_min_distance is not None:
            with torch.no_grad():
                for i, v in enumerate(V):
                    vols = self.crowd()
                    if not isinstance(vols, float):
                        self.vols = vols.detach().cpu()
                    V[i] += vols

        #add ice
        if self.ice_model is not None:
            with torch.no_grad():
                V = self.solvate(V)
            
        #scatter V
        self.exitwaves = self.scattering(V)
        
        #aberrate exitwaves
        ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)
        
        #image/noise
        if self.anisomag is None:
            images = self.detector(self.detector_waves)
        else:
            images = self.detector(self.detector_waves, anisomag=self.anisomag[idx])

        if self.pad_fft:
            return images[:, self.nxy // 2: -self.nxy // 2, self.nxy // 2: -self.nxy // 2]
        else:
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

class ImageGenerator(L.LightningModule):
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
        parameterization='kirkland'
    ):
        """
        A particle image generator module to simulate 2D particles from the input
        3D map.

        Parameters
        ----------
        scattering_potential : 3D tensor
            The 3D scattering potential of the particle.
        pixel_size: float
            Pixel size in angstroms. Assumes dz is also pixel_size for now.
        quaternions: 2D tensor
            Batch of quaternions with shape (N, 4). Follows scipy's convention of
            (x,y,z,w).
        translations: 2D tensor
            Batch of translations with shape (N, 2) in angstroms. translations[:, 0]
            x-translations whilst translations[:, 1] contrains y-translations in
            angstroms.
        ctf_params: 2D tensor
            Batch of CTF parameters with shape (N, 4), where the 4 columns are
            [Cs, defocusU, defocusV, defocusAngle], in angstroms and degrees.
        energy: float
            Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
        dose_per_angstrom: float
            Dose of the electron beam in e-/A^2.
        anisomag: None or 3D tensor
            Specifies 2x2 anisotropic matrix for each image.
        ice_model: str or None
            Specifies ice algorithm. Set to None for no ice. Options includes only
            'randomchoice' (fast) for now.
        ice_thickness: float
            Specifies the thickness of ice in Angstroms. Typically 100–1000 A. Must
            be same or larger than FOV of the particle.
        scattering_mode: str
            Specifies scattering model to use. Options include 'multislice',
            'firstborn', 'projection' and 'ctf', in order of increasing approximations.
        aberration_model: str
            Specifies aberration model to use. Options include 'holography' and 'ctf'.
        noise_model : str
            Specifies noise model. Currently only 'poisson' available.
        klim: float
            Kirkland [1] explains that setting klim = 0.66 is necessary to avoid
            aliasing for FFT methods (multislice and first Born). But this numerically
            lowers the spatial frequency information in the resultant exitwaves, so
            default is set to None.
        flip_curvature: bool
            This corresponds to positive/negative Ewald sphere curvature
            ambiguity. Set to False for positive, and True for negative (CryoSPARC).
            Only affects multislice and first Born models.
        alpha: float
            The amplitude contrast ratio to use for the CTF model. Common values
            are 0.07 and 0.1.
        crowd_min_distance: float, optional
            If not None, adds duplicate particles to simulate crowding at this
            distance in Angstroms around the particle.
        crowd_max_distance_z: float, optional
            Maximum distance in z to crowd. If None, defaults to ice_thickness which
            may overcrowd.
        pad_fft : bool
            Set to True to reduce border artifacts at the expense of memory and speed.

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).
           [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
           Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.
        """
        super().__init__()

        # model params
        self.nxy = scattering_potential.shape[-1]
        self.pixel_size = pixel_size
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.ice_model = ice_model
        self.ice_thickness = ice_thickness
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.klim = klim
        self.crowd_min_distance = crowd_min_distance
        self.pad_fft = pad_fft
        if self.pad_fft:
            self.pad_nxy = self.nxy + (self.nxy // 2) * 2 #
        else:
            self.pad_nxy = self.nxy
        self.progressbars = progressbars
        self.parameterization = parameterization

        # compute number of z-axis pixels due to ice thickness
        if ice_model is None:
            self.nz = self.nxy
        else:
            if ice_thickness is None:
                self.nz = self.nxy
            else:
                # thickness of ice must be at least the size of particle FOV.
                if ice_thickness < self.nxy * pixel_size:
                    self.nz = self.nxy
                else:
                    self.nz = int(ice_thickness // pixel_size)
        if crowd_max_distance_z is None:
            crowd_max_distance_z = self.nz
        self.crowd_max_distance_z = crowd_max_distance_z
            
        # register buffers
        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
       
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)
        
        # for dynamic/kinematic scattering, we need to account for the defocus
        # implicit in the scattering module
        self.ctf_params = nn.ParameterDict({
            k: nn.Parameter(v) for k, v in ctf_params.items()
        })
        if self.scattering_model not in ['projection', 'ctf']:
            if "dfu" in self.ctf_params:
                shifted = self.ctf_params["dfu"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfu"] = nn.Parameter(shifted)
            if "dfv" in self.ctf_params:
                shifted = self.ctf_params["dfv"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfv"] = nn.Parameter(shifted)
                

        # initialize modules
        self.scattering = Scattering(
            self.pad_nxy,
            pixel_size,
            energy,
            dose_per_angstrom,
            scattering_model=scattering_model,
            klim=klim,
            flip_curvature=flip_curvature,
            nz=self.nz,
            alpha=alpha,
            progressbars=self.progressbars
        )

        self.aberration = Aberration(
            self.pad_nxy,
            pixel_size,
            energy,
            aberration_model=aberration_model,
            alpha=alpha,
        )

        self.detector = Detector(
            pixel_size,
            dose_per_angstrom,
            aberration_model=aberration_model,
            noise_model=noise_model,
        )

        if self.crowd_min_distance is not None:
            self.crowd = CrowdWithDuplicates(
                self.V,
                pixel_size,
                self.crowd_min_distance,
                nxy_out=self.pad_nxy if pad_fft else self.nxy,
                nz_out=self.nz,
                max_distance_z=self.crowd_max_distance_z,
                max_distance_xy=None,
                method='3d',
                n_points=torch.inf, seed='origin',
                progressbars=self.progressbars
            )

        if ice_model is not None:
            if ice_model == 'randomchoice':
                self.icemaker = NaiveIcemaker(
                    n=self.nxy,
                    dx=pixel_size,nz=self.nz,
                    progressbars=self.progressbars
                )
            elif ice_model == 'iterative':
                self.icemaker = Icemaker(
                    n=self.nxy,
                    dx=pixel_size,
                    nz=self.nz,
                    progressbars=self.progressbars,
                    parameterization=parameterization
                )

    def rotate(self, Q, T):
        if len(Q.shape) < 2:
            Q = Q.unsqueeze(0)
        if len(T.shape) < 2:
            T = T.unsqueeze(0)
        R = Rotation.from_quat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        theta = rotations.build_affine_matrix(R.as_matrix(), T)
        V = rotations.rotate_volume(self.V, theta, origin='relion')
        return V
        
    def solvate(self, V):
        # generates ice with size (B x Z x Y x X)
        ice = self.icemaker.generate_ice(batchsize=len(V))

        if self.pad_fft:
            ice = F.pad(ice,
                      (self.nxy // 2, self.nxy // 2,# x-axis, last dim
                       self.nxy // 2, self.nxy // 2,# y-axis, second last dim
                       0, 0,  # z-axis
                      ),
                     mode='reflect')

        # pad V in z-axis if ice_thickness is not None
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(V,
                      (0,0,# x-axis
                       0,0,# y-axis
                       zpad_px//2, self.nz - zpad_px//2 - V.shape[1],  # z-axis
                      ))
        icemask = V.detach().clone()
        icemask[icemask<0.01] = 1
        icemask[icemask>=0.01] = 0
        V = V + ice * icemask
        self.icemask = icemask.detach().cpu() #save as attribute just to check
        return V

    def forward(self, idx):
        #rotate V, returns (B x Z x Y x X)
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        #pad z
        if self.ice_thickness is not None:
            zpad_px = self.nz - self.nxy
            V = F.pad(V,
                      (0, 0,# x-axis, last dim
                       0, 0,# y-axis, second last dim
                       zpad_px//2, self.nz - zpad_px//2 - V.shape[1],  # z-axis
                      ),
                      mode='constant')
        #pad xy
        if self.pad_fft:
            V = F.pad(V,
                      (self.nxy // 2, self.nxy // 2,# x-axis, last dim
                       self.nxy // 2, self.nxy // 2,# y-axis, second last dim
                       0, 0,  # z-axis
                      ),
                     mode='constant')

        #adds crowd
        if self.crowd_min_distance is not None:
            with torch.no_grad():
                for i, v in enumerate(V):
                    vols = self.crowd()
                    if not isinstance(vols, float):
                        self.vols = vols.detach().cpu()
                    V[i] += vols

        #add ice
        if self.ice_model is not None:
            with torch.no_grad():
                V = self.solvate(V)

        #scatter V
        self.exitwaves = self.scattering(V)

        #aberrate exitwaves
        ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)

        #image/noise
        if self.anisomag is None:
            images = self.detector(self.detector_waves)
        else:
            images = self.detector(self.detector_waves, anisomag=self.anisomag[idx])

        if self.pad_fft:
            return images[:, self.nxy // 2: -self.nxy // 2, self.nxy // 2: -self.nxy // 2]
        else:
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


class MicrographGenerator(L.LightningModule):
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
    ):
        """
        A micrograph generator module to simulate 2D micrographs from the input
        3D map.

        Parameters
        ----------
        scattering_potential : 3D tensor
            The 3D scattering potential of the particle.
        pixel_size: float
            Pixel size in angstroms. Assumes dz is also pixel_size for now.
        ctf_params: 2D tensor
            Batch of CTF parameters with shape (N, 4), where the 4 columns are
            [Cs, defocusU, defocusV, defocusAngle], in angstroms and degrees.
        energy: float
            Energy of the electron beam in keV. Typical values are 100/120/200/300 keV.
        dose_per_angstrom: float
            Dose of the electron beam in e-/A^2.
        anisomag: None or 3D tensor
            Specifies 2x2 anisotropic matrix for each image.
        ice_model: str or None
            Specifies ice algorithm. Set to None for no ice. Options includes only
            'randomchoice' (fast) for now.
        ice_thickness: float
            Specifies the thickness of ice in Angstroms. Typically 100–1000 A. Must
            be same or larger than FOV of the particle.
        scattering_mode: str
            Specifies scattering model to use. Options include 'multislice',
            'firstborn', 'projection' and 'ctf', in order of increasing approximations.
        aberration_model: str
            Specifies aberration model to use. Options include 'holography' and 'ctf'.
        noise_model : str
            Specifies noise model. Currently only 'poisson' available.
        klim: float
            Kirkland [1] explains that setting klim = 0.66 is necessary to avoid
            aliasing for FFT methods (multislice and first Born). But this numerically
            lowers the spatial frequency information in the resultant exitwaves, so
            default is set to None.
        alpha: float
            The amplitude contrast ratio to use for the CTF model. Common values
            are 0.07 and 0.1.
        crowd_min_distance: float, optional
            If not None, adds duplicate particles to simulate crowding at this
            distance in Angstroms around the particle.
        crowd_max_distance_z: float, optional
            Maximum distance in z to crowd. If None, defaults to ice_thickness which
            may overcrowd.
        pad_fft : bool
            Set to True to reduce border artifacts at the expense of memory and speed.
        move_to_cpu : bool
            Set to True to save GPU memory for intermediate calculations.

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).
           [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
           Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.
        """
        super().__init__()

        # model params
        self.pixel_size = pixel_size
        if isinstance(micrograph_size, int):
            self.nxy = micrograph_size
        elif isinstance(micrograph_size, (tuple, list)) and micrograph_size[0] == micrograph_size[1]:
            self.nxy, _ = micrograph_size
        else:
            raise ValueError("micrograph_size must have same dimensions in x and y.")
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.ice_model = ice_model
        self.ice_thickness = ice_thickness
        self.alpha = alpha
        self.klim = klim
        self.crowd_min_distance = crowd_min_distance
        self.chunk_size = chunk_size
        self.move_to_cpu = move_to_cpu
        self.pad_fft = pad_fft
        if self.pad_fft:
            self.pad_nxy = self.nxy + (self.nxy // 2) * 2 #
        else:
            self.pad_nxy = self.nxy

        # compute number of z-axis pixels due to ice thickness
        if ice_model is None:
            self.nz = scattering_potential.shape[0]
        else:
            if ice_thickness is None or ice_thickness == 0:
                self.nz = scattering_potential.shape[0]
            else:
                self.nz = int(ice_thickness // pixel_size)
        if crowd_max_distance_z is None:
            crowd_max_distance_z = self.nz
        self.crowd_max_distance_z = crowd_max_distance_z
            
        # register buffers
        self.register_buffer("V", scattering_potential)
        
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)
        
        # for dynamic/kinematic scattering, we need to account for the defocus
        # implicit in the scattering module
        self.ctf_params = nn.ParameterDict({
            k: nn.Parameter(v) for k, v in ctf_params.items()
        })
        if self.scattering_model not in ['projection', 'ctf']:
            if "dfu" in self.ctf_params:
                shifted = self.ctf_params["dfu"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfu"] = nn.Parameter(shifted)
            if "dfv" in self.ctf_params:
                shifted = self.ctf_params["dfv"].detach() - (self.nz * pixel_size) / 2
                self.ctf_params["dfv"] = nn.Parameter(shifted)

        # initialize modules
        self.scattering = Scattering(
            self.pad_nxy,
            pixel_size,
            energy,
            dose_per_angstrom,
            scattering_model=scattering_model,
            klim=klim,
            nz=self.nz,
            alpha=alpha
        )

        self.aberration = Aberration(
            self.pad_nxy,
            pixel_size,
            energy,
            aberration_model=aberration_model,
            alpha=alpha,
        )

        self.detector = Detector(
            pixel_size,
            dose_per_angstrom,
            aberration_model=aberration_model,
            noise_model=noise_model
        )

        self.crowd = CrowdWithDuplicates(
            scattering_potential,
            pixel_size,
            self.crowd_min_distance,
            nxy_out=self.pad_nxy if pad_fft else self.nxy,
            nz_out=self.nz,
            max_distance_z=self.crowd_max_distance_z,
            max_distance_xy=None,
            method='3d',
            n_points=torch.inf,
            seed='random',
            chunk_size=chunk_size,
            move_to_cpu=self.move_to_cpu
        )

        if ice_model is not None:
            if ice_model == 'randomchoice':
                self.icemaker = NaiveIcemaker(n=self.nxy,
                                              dx=pixel_size,
                                              nz=self.nz)
            elif ice_model == 'iterative':
                self.icemaker = Icemaker(n=256,
                                         dx=pixel_size,
                                         nz=256,
                                         chunk_size=self.chunk_size,)

    def solvate(self, V):
        # generates ice with size (B x Z x Y x X)
        self.ice = self.icemaker.generate_big_ice(V.shape)

        if self.pad_fft:
            self.ice = F.pad(self.ice,
                      (self.nxy // 2, self.nxy // 2,# x-axis, last dim
                       self.nxy // 2, self.nxy // 2,# y-axis, second last dim
                       0, 0,  # z-axis
                      ),
                     mode='reflect')

        icemask = V < 0.01  # boolean mask, same shape, no copy of V
        V += self.ice * icemask
        # self.icemask = icemask #save as attribute just to check
        return V


    def forward(self, idx):
        V = torch.empty(len(idx), self.nz, self.nxy, self.nxy)

        #adds crowd
        if self.crowd_min_distance is not None:
            with torch.no_grad():
                for i, v in enumerate(V):
                    self.vols = self.crowd()
                    V[i] += self.vols
                    # V[i] += self.crowd()

        #add ice
        if self.ice_model is not None:
            with torch.no_grad():
                V = self.solvate(V)

        #scatter V
        self.exitwaves = self.scattering(V)

        #aberrate exitwaves
        ctf_batch = {k: v[idx] for k, v in self.ctf_params.items()}
        self.detector_waves = self.aberration(self.exitwaves, ctf_batch)
        
        #image/noise
        if self.anisomag is None:
            images = self.detector(self.detector_waves)
        else:
            images = self.detector(self.detector_waves, anisomag=self.anisomag[idx])

        if self.pad_fft:
            return images[:, self.nxy // 2: -self.nxy // 2, self.nxy // 2: -self.nxy // 2]
        else:
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