import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

def random_quaternion(convention="xyzw"):
    """
    Generates a random quaternion. 

    Parameters
    ----------
    convention : str
        Quaternion convention. Scipy Rotation uses 'xyzw' convention, whereas the 
        numpy-quaternion library uses 'wxyz'.

    Returns
    -------
    quat : tensor
        A random quaternion.
    """
    U, V, W = torch.rand(3)
    if convention == "wxyz":
        quat = torch.tensor(
            [
                torch.sqrt(1 - U) * torch.sin(2 * torch.pi * V),
                torch.sqrt(1 - U) * torch.cos(2 * torch.pi * V),
                torch.sqrt(U) * torch.sin(2 * torch.pi * W),
                torch.sqrt(U) * torch.cos(2 * torch.pi * W),
            ]
        )
    if convention == "xyzw":
        quat = torch.tensor(
            [
                torch.sqrt(1 - U) * torch.cos(2 * torch.pi * V),
                torch.sqrt(U) * torch.sin(2 * torch.pi * W),
                torch.sqrt(U) * torch.cos(2 * torch.pi * W),
                torch.sqrt(1 - U) * torch.sin(2 * torch.pi * V),
            ]
        )
    return quat


def random_rotvec():
    """
    Generates a random rotation vector.

    Returns
    -------
    rotvec : tensor
        A random rotation vector.
    """
    U, V, theta = torch.rand(3)
    theta = theta * torch.pi
    psi = 2 * torch.pi * U
    phi = torch.arccos(2 * V - 1)
    x = torch.sin(phi) * torch.cos(psi)
    y = torch.sin(phi) * torch.sin(psi)
    z = torch.cos(phi)
    rotvec = torch.tensor([x,y,z]) * theta
    return rotvec


def rotate_coordinates(coordinates, quats, inverse=False):
    """
    Rotates a set of xyz coordinates based on input quaternions.

    Parameters
    ----------
    coordinates : array_like, shape (3,) or (N,3)
        Each vectors[i] represents a vector in 3D space. A single vector can either 
        be specified with shape (3, ) or (1, 3). The number of rotations and number 
        of vectors given must follow standard numpy broadcasting rules: either one 
        of them equals unity or they both equal each other.
    quat : array_like, shape (N, 4) or (4,)
        Each row is a (possibly non-unit norm) quaternion representing an active 
        rotation, in scalar-last (x, y, z, w) format. Each quaternion will be 
        normalized to unit norm.
    inverse : boolean, optional
        If True then the inverse of the rotation(s) is applied to the input vectors. 
        Default is False.

    Returns
    -------
    rotate_coordinates : array_like, shape (3,) or (N,3)
        The rotated coordinates.
    """
    r = R.from_quat(quats)
    return r.apply(coordinates, inverse=inverse)


def rotate_volume(V, theta, origin="relion"):
    """
    Rotates a single 3D volume based on the batch of 3x4 affine transform matrices

    Written by Tristan Bepler.

    Parameters
    ----------
    V : Z x Y x X torch.Tensor
        The volume to be rotated, must be real-valued.
    theta : (B,3,4) torch.Tensor
        B is the batch size. Concatenates a 3x3 rotation matrix and a 3x1 translation vector.
    origin : str
        Convention for the index of the origin of rotation. "relion" defines the
        origin to be at [nz//2, ny//2, nx//2], whereas "center" sets it to
        [(nz + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]

    Returns
    -------
    V : B x Z x Y x X
        The rotated volumes.
    """
    align_corners = False

    B = theta.size(0)
    nz, ny, nx = V.size()

    # create the coordinate grids depending on origin convention.
    if origin == "relion":
        null_rot = torch.zeros_like(theta)
        null_rot[:, 0, 0] = 1.0
        null_rot[:, 1, 1] = 1.0
        null_rot[:, 2, 2] = 1.0
        grid = F.affine_grid(null_rot, (B, 1, nz, ny, nx), align_corners=align_corners)

        # rotate the grid around the center defined by RELION
        cz = nz // 2
        cy = ny // 2
        cx = nx // 2
        center = grid[:, cz, cy, cx]  # (B, 3)
        center = center.unsqueeze(1)
        grid = grid.view(B, nz * ny * nx, 3)
        grid = (grid - center).bmm(theta[..., :-1].transpose(1, 2)) + center

        # translate the grid
        dx = theta[..., -1].unsqueeze(1)  # (B, 1, 3)
        grid = grid + dx
        grid = grid.view(B, nz, ny, nx, 3)
    elif origin == "center":
        grid = F.affine_grid(theta, (B, 1, nz, ny, nx), align_corners=align_corners)

    # transform the volume
    V = V.unsqueeze(0).unsqueeze(1)  # (1 x 1 x Z x Y x X)
    V = V.expand(B, 1, nz, ny, nx)
    V_ = F.grid_sample(V, grid, align_corners=align_corners, padding_mode="border")

    return V_.squeeze(1)  # B x Z x X x Y

def quaternion_to_rotation_matrix(q):
    """
    Converts quaternions (x,y,z,w) to rotation matrices. 
    Matches with scipy.rotation.from_quat(q).as_matrix().

    Written by Tristan Bepler.

    Parameters
    ----------
    q : 2D tensor
        Batch of quaternions with shape (N, 4).

    Returns
    -------
    R : 3D tensor
        Batch of rotation matrices with shape (N, 3, 3).
    """
    s = 1 / torch.sum(q**2, axis=-1)
    qr = q[..., 3]
    qi = q[..., 0]
    qj = q[..., 1]
    qk = q[..., 2]
    shape = q.shape[:-1] + (3, 3)
    R = torch.stack(
        [
            1 - 2 * s * (qj**2 + qk**2),
            2 * s * (qi * qj - qk * qr),
            2 * s * (qi * qk + qj * qr),
            2 * s * (qi * qj + qk * qr),
            1 - 2 * s * (qi**2 + qk**2),
            2 * s * (qj * qk - qi * qr),
            2 * s * (qi * qk - qj * qr),
            2 * s * (qj * qk + qi * qr),
            1 - 2 * s * (qi**2 + qj**2),
        ],
        axis=-1,
    ).reshape(*shape)
    return R

def translations_angstrom_to_torch(Txy, n, voxel_size):
    """
    Builds a batch of normalized translation vectors from rlnOriginXAngst and
    rlnOriginYAngst in starfiles.

    Torch affine matrix uses a normalized coordinate system, where the coordinates
    of each axis ranges from [-1, 1]. Therefore, we need to do a coordinate
    transformation to match this Torch coordinate system for translations.

    Parameters
    ----------
    Txy : 2D tensor
        Translation vector of shape (N,2), built from [rlnOriginXAngst, rlnOriginYAngst]. 
        In angstroms.
    n : int
        Number of pixels in x/y direction.
    voxel_size : float
        Voxel size in angstroms.

    Returns
    -------
    T : 2D tensor
        Batch of Torch normalized translation vectors with shape (N, 3).
    """
    num = len(Txy)
    tz = torch.zeros(num, device=Txy.device)
    T = torch.concat([Txy, tz[..., None]], dim=-1)
    T *= 2 / n / voxel_size
    return T

def build_affine_matrix(R, T):
    """
    Builds a batch of Torch's affine matrices (N, 3, 4) from a batch of rotation
    matrices (N, 3, 3) and Torch normalized translation vectors (N, 3).

    CryoSPARC performs shifts before rotations. However, the affine matrix by 
    definition performs rotations before shifts. As such, we need to modify the
    translation vector, T, by
    T_1' = R_11T_1 + R_12T_2 + R_13T_3
    T_2' = R_21T_1 + R_22T_2 + R_23T_3
    T_3' = R_31T_1 + R_32T_2 + R_33T_3

    Parameters
    ----------
    R : 3D tensor
        Batch of rotation matrices with shape (N, 3, 3).
    T : 2D tensor
        Batch of Torch normalized translation vectors with shape (N, 3). Note that
        this is not the same as the shifts directly from CryoSPARC/RELION starfiles.
        Those shifts must be normalized using translations_angstrom_to_torch.

    Returns
    -------
    R : 3D tensor
        Batch of rotation matrices with shape (N, 3, 3).
    """
    if T is None:
        T = torch.zeros_like(R[..., 0])
    #old
    # theta = torch.concat([R, T.unsqueeze(2)], dim=-1)
    
    #new
    Tprime = torch.zeros_like(T)
    Tprime[:,0] = R[:,0,0] * T[:,0] + R[:,0,1] * T[:,1] + R[:,0,2] * T[:,2]
    Tprime[:,1] = R[:,1,0] * T[:,0] + R[:,1,1] * T[:,1] + R[:,1,2] * T[:,2]
    Tprime[:,2] = R[:,2,0] * T[:,0] + R[:,2,1] * T[:,1] + R[:,2,2] * T[:,2]
    theta = torch.concat([R, Tprime.unsqueeze(2)], dim=-1)
    return theta