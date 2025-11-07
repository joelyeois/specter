import numpy as np
import torch
from cryosparc.dataset import Dataset
from scipy.spatial.transform import Rotation
import starfile
import mrcfile
import os
import pandas as pd


def extract_parameters_from_csfile(
    csfile_path, return_class="0", rotation_representation="quaternion"
):
    """
    Extracts pose and ctf parameters from CryoSPARC .cs files and converts them to
    Ghostbuster parameters

    Parameters
    ----------
    csfile_path : str
        Path of the .cs file.
    return_class: str
        Specifies which particle class to return. Options are '0', '1', ... , 'all'.

    Return
    ------
    energy_kev : 1D tensor
        Energy in keV.
    pixel_size : 1D tensor
        Pixel sizes in Angstrom.
    alpha : 1D tensor
        Amplitude contrast ratio.
    rotations : 2D tensor
        Quaternions with shape (N, 4).
    translations_A : 2D tensor
        xy-translations with shape (N, 2).
    ctf_params : 2D tensor
        CTF parameters with shape (N, 7). Parameters are Cs, dfu, dfv, dfang, tiltx,
        tilty, phaseshift.
    indices : 1D tensor
        Contains the indices of the specific particles.

    Notes
    -----
    .. To-do

    """
    dataset = Dataset.load(csfile_path)

    # extract translations
    translations_px = torch.as_tensor(dataset["alignments3D/shift"])
    pixel_size = torch.as_tensor(dataset["alignments3D/psize_A"])
    translations_A = translations_px * pixel_size[..., None]
    if torch.allclose(pixel_size[0], pixel_size.mean()):
        pixel_size = pixel_size[0]
    else:
        print("Pixel size is not same for all particles.")

    # extract spherical aberration
    cs_mm = torch.as_tensor(dataset["ctf/cs_mm"])
    cs_A = cs_mm * 1e7

    # extract defocus
    dfang_rad = torch.as_tensor(dataset["ctf/df_angle_rad"])
    dfu_A = torch.as_tensor(dataset["ctf/df1_A"])
    dfv_A = torch.as_tensor(dataset["ctf/df2_A"])

    # extract amplitude contrast
    alpha = torch.as_tensor(dataset["ctf/amp_contrast"])
    if torch.allclose(alpha[0], alpha.mean()):
        alpha = alpha[0]
    else:
        print("Alpha is not same for all particles.")

    # extract energy
    energy_kev = torch.as_tensor(dataset["ctf/accel_kv"])
    if torch.allclose(energy_kev[0], energy_kev.mean()):
        energy_kev = energy_kev[0]
    else:
        print("Energy is not same for all particles.")

    # extract rotations
    r = Rotation.from_rotvec(dataset["alignments3D/pose"])
    if rotation_representation == "quaternion":
        rotations = torch.from_numpy(r.as_quat()).type(torch.float32)
    elif rotation_representation == "rotvec":
        # rotations = torch.from_numpy(r.as_rotvec()).type(torch.float32)
        rotations = torch.as_tensor(dataset["alignments3D/pose"])

    # extract split
    split = torch.as_tensor(dataset["alignments3D/split"].astype(int))

    # extract beamtilt
    beamtiltx_rad = torch.arcsin(torch.as_tensor(dataset["ctf/tilt_A"][:, 0] / cs_A))
    beamtilty_rad = torch.arcsin(torch.as_tensor(dataset["ctf/tilt_A"][:, 1] / cs_A))

    # extract phaseshift
    phaseshift_rad = torch.as_tensor(dataset["ctf/phase_shift_rad"])

    # extract ctf shift, and add to translations
    beamshift_A = torch.as_tensor(dataset["ctf/shift_A"])
    translations_A -= beamshift_A

    # extract trefoil
    tref1 = torch.as_tensor(dataset["ctf/trefoil_A"][:, 0] / 1000)
    tref2 = torch.as_tensor(dataset["ctf/trefoil_A"][:, 1] / 1000)

    # build ctf_params
    ctf_params = torch.stack(
        [
            cs_A,
            dfu_A,
            dfv_A,
            dfang_rad,
            beamtiltx_rad,
            beamtilty_rad,
            phaseshift_rad,
            tref1,
            tref2,
        ],
        dim=-1,
    )

    # extract per-particle scale factors
    scale = torch.as_tensor(dataset["alignments3D/alpha"])

    # extract anisotropic magnification
    # cryosparc defines the M matrix in Fourier space, and stores it after
    # subtracting away the identity. Ghostbuster uses the real-space M instead.
    anisomag = torch.as_tensor(dataset["ctf/anisomag"]).reshape(-1, 2, 2)
    if torch.allclose(torch.tensor(0.0), torch.sum(anisomag)):
        anisomag = None
    else:
        anisomag = anisomag + torch.eye(2).unsqueeze(0)
        # Compute the real-space equivalent matrix
        anisomag = torch.inverse(anisomag.mT)
        if return_class == "0":
            anisomag = anisomag[split == 0]
        elif return_class == "1":
            anisomag = anisomag[split == 1]

    if return_class == "all":
        indices = torch.arange(len(split))
        return (
            energy_kev,
            pixel_size,
            alpha,
            rotations,
            translations_A,
            ctf_params,
            scale,
            anisomag,
            indices,
        )
    elif return_class == "0":
        indices = torch.squeeze(torch.nonzero(split == 0))
        return (
            energy_kev,
            pixel_size,
            alpha,
            rotations[split == 0],
            translations_A[split == 0],
            ctf_params[split == 0],
            scale[split == 0],
            anisomag,
            indices,
        )
    elif return_class == "1":
        indices = torch.squeeze(torch.nonzero(split == 1))
        return (
            energy_kev,
            pixel_size,
            alpha,
            rotations[split == 1],
            translations_A[split == 1],
            ctf_params[split == 1],
            scale[split == 1],
            anisomag,
            indices,
        )


def create_particle_starfile(
    particles,
    rotations=None,
    translations=None,
    alpha=0.1,
    folderpath="",
    energy=None,
    dx=None,
    starfilename="particles",
    mrcfilename=None,
    ctf_params=None,
):
    """
    Saves a particle stack as MRCS and creates a RELION .star file for tomographic reconstruction.

    Parameters
    ----------
    particles : np.ndarray, shape (N, H, W)
        Stack of 2D particle images.
    rotations : np.ndarray, optional
        Rotation information per particle. Can be quaternions (N,4), rotation matrices (N,3,3),
        or rotation vectors (N,3). Default None.
    translations : np.ndarray, optional
        Optional translations per particle. Not currently applied to STAR.
    alpha : float, optional
        Amplitude contrast ratio. Default 0.1.
    folderpath : str, optional
        Directory to save MRCS and STAR files. Default '' (current folder).
    energy : float, optional
        Electron beam voltage in kV or eV. Required for STAR file.
    Cs_mm : float, optional
        Spherical aberration in mm. Default 2.0.
    dx : float, optional
        Pixel size in Å. Default 1.0.
    starfilename : str, optional
        Name of the STAR file. Default 'particles'.
    mrcfilename : str, optional
        Name of the MRCS file. Defaults to same as starfilename.
    defocusU, defocusV, defocusAngle : list or np.ndarray, optional
        CTF parameters for each particle.
    originXAngst, originYAngst : list or np.ndarray, optional
        Origin shifts for each particle. Default zeros.
    randomsubset : int, optional
        If specified, randomly sample this many particles.

    Returns
    -------
    star_path : str
        Path to the saved STAR file.
    """

    # create directory if specified
    if folderpath != "":
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

    # create projections mrcs file
    if mrcfilename is None:
        mrcfilename = starfilename
    mrcs_path = os.path.join(folderpath, mrcfilename + ".mrcs")
    with mrcfile.new(mrcs_path, overwrite=True) as mrc:
        mrc.set_data(particles.numpy().astype(np.float32))

    # convert rotations to Relion euler
    if len(rotations.shape) == 3:
        # N x 3 x 3 -> rotation matrices
        R = Rotation.from_matrix(rotations)
    elif rotations.shape[-1] == 4:
        # N x 4 -> quaternions
        R = Rotation.from_quat(rotations)
    elif rotations.shape[-1] == 3:
        # N x 3 -> rotvec
        R = Rotation.from_rotvec(rotations)
    euler = R.as_euler("ZYZ", degrees=True)

    # create associated starfile
    n = len(particles)
    d = {
        "rlnVoltage": energy,
        "rlnSphericalAberration": ctf_params[:, 0] / 1e7,
        "rlnAmplitudeContrast": alpha,
        "rlnImagePixelSize": dx,
        "rlnAngleRot": euler[:, 0],
        "rlnAngleTilt": euler[:, 1],
        "rlnAnglePsi": euler[:, 2],
        "rlnImageName": [str(i) + "@" + mrcfilename + ".mrcs" for i in range(1, n + 1)],
    }

    d["rlnDefocusU"] = ctf_params[:, 1]
    d["rlnDefocusV"] = ctf_params[:, 2]
    d["rlnDefocusAngle"] = ctf_params[:, 3]
    d["rlnPhaseShift"] = ctf_params[:, 5]

    d["rlnOriginXAngst"] = translations[:, 0]
    d["rlnOriginYAngst"] = translations[:, 1]

    particles_df = pd.DataFrame(data=d)

    star_path = os.path.join(folderpath, starfilename + ".star")
    starfile.write(particles_df, star_path, overwrite=True)
    print("Saved at: " + star_path)
