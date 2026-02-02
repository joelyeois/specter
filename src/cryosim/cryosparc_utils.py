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
    Extract poses and CTF parameters from CryoSPARC .cs file.

    Parameters
    ----------
    csfile_path : str
        Path of the .cs file.
    return_class : str, optional
        Specifies which particle class to return. Options are '0', '1', or 'all'.
        Default is '0'.
    rotation_representation : str, optional
        Representation of rotations. 'quaternion' or 'rotvec'. Default is 'quaternion'.

    Returns
    -------
    energy_kev : torch.Tensor
        Energy in keV.
    pixel_size : torch.Tensor
        Pixel sizes in Ångstrom.
    alpha : torch.Tensor
        Amplitude contrast ratio.
    rotations : torch.Tensor
        Quaternions with shape (N, 4) or rotation vectors.
    translations_A : torch.Tensor
        xy-translations in Ångstrom with shape (N, 2).
    ctf_params : torch.Tensor
        CTF parameters with shape (N, 7). Parameters are (Cs, dfu, dfv, dfang, tiltx, tilty, phaseshift).
    scale : torch.Tensor
        Per-particle scale factors.
    anisomag : torch.Tensor or None
        Anisotropic magnification matrices (N, 2, 2) or None if identity.
    indices : torch.Tensor
        Indices of the extracted particles from the dataset.
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
    dfang_deg = dfang_rad / torch.pi * 180
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

        # correct for anisotropic shift
        corrected_shifts = translations_A.unsqueeze(
            -1
        )  # Add a dimension to make it (B, 2, 1)

        # Perform batch matrix multiplication
        corrected_shifts = torch.bmm(anisomag, corrected_shifts)

        # Remove the last dimension to get (B, 2)
        corrected_shifts = corrected_shifts.squeeze(-1)
        translations_A = corrected_shifts

    if return_class == "all":
        indices = torch.arange(len(split))

        # build ctf_params
        ctf_params = {
            "cs": cs_A,
            "dfu": dfu_A,
            "dfv": dfv_A,
            "dfang": dfang_deg,
            "tiltx": beamtiltx_rad,
            "tilty": beamtilty_rad,
            "phaseshift": phaseshift_rad,
            "tref1": tref1,
            "tref2": tref2,
        }
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

        # build ctf_params
        ctf_params = {
            "cs": cs_A[split == 0],
            "dfu": dfu_A[split == 0],
            "dfv": dfv_A[split == 0],
            "dfang": dfang_deg[split == 0],
            "tiltx": beamtiltx_rad[split == 0],
            "tilty": beamtilty_rad[split == 0],
            "phaseshift": phaseshift_rad[split == 0],
            "tref1": tref1[split == 0],
            "tref2": tref2[split == 0],
        }
        return (
            energy_kev,
            pixel_size,
            alpha,
            rotations[split == 0],
            translations_A[split == 0],
            ctf_params,
            scale[split == 0],
            anisomag[split == 0],
            indices,
        )
    elif return_class == "1":
        indices = torch.squeeze(torch.nonzero(split == 1))

        # build ctf_params
        ctf_params = {
            "cs": cs_A[split == 1],
            "dfu": dfu_A[split == 1],
            "dfv": dfv_A[split == 1],
            "dfang": dfang_deg[split == 1],
            "tiltx": beamtiltx_rad[split == 1],
            "tilty": beamtilty_rad[split == 1],
            "phaseshift": phaseshift_rad[split == 1],
            "tref1": tref1[split == 1],
            "tref2": tref2[split == 1],
        }
        return (
            energy_kev,
            pixel_size,
            alpha,
            rotations[split == 1],
            translations_A[split == 1],
            ctf_params,
            scale[split == 1],
            anisomag[split == 0],
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
    Save particle stack as MRCS and create RELION .star file.

    Parameters
    ----------
    particles : torch.Tensor or np.ndarray
        Stack of 2D particle images, shape (N, H, W).
    rotations : torch.Tensor or np.ndarray, optional
        Pose information per particle. Can be quaternions (N,4), matrices (N,3,3),
        or rotation vectors (N,3). Default is None.
    translations : torch.Tensor or np.ndarray, optional
        Translations per particle (x, y) in Ångstrom. Default is None.
    alpha : float, optional
        Amplitude contrast ratio. Default is 0.1.
    folderpath : str, optional
        Directory to save MRCS and STAR files. Default is "" (current directory).
    energy : float, optional
        Electron beam energy in kV. Required.
    dx : float, optional
        Pixel size in Ångstrom. Required.
    starfilename : str, optional
        Name of the output STAR file (without extension). Default is "particles".
    mrcfilename : str, optional
        Name of the output MRCS file (without extension). Defaults to `starfilename`.
    ctf_params : torch.Tensor, optional
        CTF parameters for each particle, shape (N, K).
        Expected columns: [Cs (Å), dfu (Å), dfv (Å), dfang (rad), ..., phaseshift (rad)].

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
