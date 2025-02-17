import lightning as L
import numpy as np
import torch
from fft_tools import fft2, fftn, ifft2, ifftn
from microscope import Aberration, Detector
from scattering import Scattering
from icemaker import NaiveIcemaker
import rotations

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
        ice_model=None,
        scattering_model="multislice",
        aberration_model="holography",
        noise_model="poisson",
        klim=None,
        flip_curvature=False,
        alpha=0.0,
    ):
        """
        A scattering module to compute the 2D exitwave from a 3D scattering
        potential. Various scattering modes are available.

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
        ice_model: str or None
            Specifices ice algorithm. Set to None for no ice. Options includes only
            'randomchoice' (fast) for now.
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

        Notes
        -----
        .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy (Springer
           US, Boston, MA, 2010).
           [2] P. A. Penczek, “Image Restoration in Cryo-Electron Microscopy” in
           Methods in Enzymology (Academic Press Inc., 2010)vol. 482, pp. 35–72.
        """
        super().__init__()

        # register buffers
        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        cs, dfu, dfv, dfang = torch.unbind(ctf_params, dim=-1)
        self.register_buffer("cs", cs)
        self.register_buffer("dfu", dfu)
        self.register_buffer("dfv", dfv)
        self.register_buffer("dfang", dfang)

        # model params
        self.n = scattering_potential.shape[0]
        self.pixel_size = pixel_size
        self.energy = energy
        self.dose_per_angstrom = dose_per_angstrom
        self.dose_per_pixel = dose_per_angstrom * pixel_size**2
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model
        self.noise_model = noise_model
        self.ice_model = ice_model
        self.flip_curvature = flip_curvature
        self.alpha = alpha
        self.klim = klim

        # initialize modules
        self.scattering = Scattering(
            self.n,
            pixel_size,
            energy,
            dose_per_angstrom,
            scattering_model=scattering_model,
            klim=klim,
            flip_curvature=flip_curvature,
        )

        self.aberration = Aberration(
            self.n,
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

        if ice_model == 'randomchoice':
            self.icemaker = NaiveIcemaker(n=V.shape[-1], dx=pixel_size)

    def rotate(self, Q, T):
        R = rotations.quaternion_to_rotation_matrix(Q)
        T = rotations.translations_angstrom_to_torch(T, self.n, self.pixel_size)
        theta = rotations.build_affine_matrix(R, T)
        V = rotations.rotate_volume(self.V, theta, origin='relion')
        return V
        
    def solvate(self, V):
        if len(V.shape) == 4:
            batchsize = len(V)
        elif len(V.shape) == 3:
            batchsize = 1
        # generates ice with size (B x Z x Y x X)
        ice = self.icemaker.generate_random_icecube(batchsize=batchsize)
        icemask = V.detach().clone()
        icemask[icemask<10] = 1
        icemask[icemask>=10] = 0
        V = V + ice.to(self.device) * icemask
        self.icemask = icemask #save as attribute just to check
        return V
        
    def forward(self, idx):
        #rotate V, returns (B x Z x Y x X)
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        #add ice
        if self.ice_model == 'randomchoice':
            V = self.solvate(V)
                
        #scatter V
        exitwaves = self.scattering(V)
        
        #aberrate exitwaves
        exitwaves = self.aberration(exitwaves, self.cs[idx], self.dfu[idx], self.dfv[idx], self.dfang[idx])
        
        #image/noise
        images = self.detector(exitwaves)
        return images