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
        Convention for the index of the origin of rotation

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
        cz = int(nz / 2)
        cy = int(ny / 2)
        cx = int(nx / 2)
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