import torch
import torch.nn.functional as F
import numpy as np


class Rotation:
    """
    3D rotation class with support for quaternions, rotation vectors, and rotation matrices.
    Internally stores rotation as a unit quaternion in xyzw format.
    """

    def __init__(self, quat: torch.Tensor, eps: float = 1e-6):
        """
        Internal constructor. Stores a unit quaternion (xyzw).
        """
        norm = quat.norm(dim=-1, keepdim=True)
        if torch.any(torch.abs(norm - 1.0) > eps):
            # Auto-normalize if not unit
            quat = quat / norm
        self._quat = quat

    # ----------------- Constructors -----------------
    @classmethod
    def from_quat(cls, quat: torch.Tensor, scalar_first: bool = False):
        """
        Create from quaternion.

        Parameters
        ----------
        quat : torch.Tensor
            Quaternion tensor [..., 4].
        scalar_first : bool
            If True, interpret as [w, x, y, z].
            If False, interpret as [x, y, z, w] (default, same as internal format).
        """
        if scalar_first:
            # Convert [w, x, y, z] → [x, y, z, w]
            quat = torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)
        return cls(quat)

    @classmethod
    def from_rotvec(cls, rotvec: torch.Tensor):
        """
        Create from rotation vector (axis * angle)
        """
        theta = rotvec.norm(dim=-1, keepdim=True)
        axis = rotvec / theta.clamp(min=1e-8)
        xyz = axis * torch.sin(theta / 2)
        w = torch.cos(theta / 2)
        quat = torch.cat([xyz, w], dim=-1)
        return cls(quat)

    @classmethod
    def from_matrix(cls, R: torch.Tensor):
        """
        Create from rotation matrix (3x3)
        """
        # Source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        m = R
        batch_mode = len(R.shape) == 3
        if not batch_mode:
            m = R.unsqueeze(0)

        tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        quat = torch.zeros((m.shape[0], 4), device=R.device, dtype=R.dtype)

        # Compute quaternion
        mask = tr > 0
        t = tr[mask] + 1
        S = torch.sqrt(t) * 2
        quat[mask, 3] = 0.25 * S
        quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / S
        quat[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / S
        quat[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / S

        mask = ~mask
        # Pick largest diagonal element
        cond1 = (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2]) & mask
        t1 = 1 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2]
        S1 = torch.sqrt(t1) * 2
        quat[cond1, 0] = 0.25 * S1
        quat[cond1, 1] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / S1
        quat[cond1, 2] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / S1
        quat[cond1, 3] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / S1

        cond2 = (m[:, 1, 1] >= m[:, 2, 2]) & mask & ~cond1
        t2 = 1 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2]
        S2 = torch.sqrt(t2) * 2
        quat[cond2, 0] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / S2
        quat[cond2, 1] = 0.25 * S2
        quat[cond2, 2] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / S2
        quat[cond2, 3] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / S2

        cond3 = mask & ~cond1 & ~cond2
        t3 = 1 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1]
        S3 = torch.sqrt(t3) * 2
        quat[cond3, 0] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / S3
        quat[cond3, 1] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / S3
        quat[cond3, 2] = 0.25 * S3
        quat[cond3, 3] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / S3

        if not batch_mode:
            quat = quat.squeeze(0)
        return cls(quat)

    # ----------------- Conversion Methods -----------------
    def as_quat(self, scalar_first: bool = False):
        """
        Return quaternion.

        Parameters
        ----------
        scalar_first : bool
            If True, return [w, x, y, z].
            If False, return [x, y, z, w] (default, internal format).
        """
        if scalar_first:
            return torch.cat([self._quat[..., 3:], self._quat[..., :3]], dim=-1)
        return self._quat

    def as_rotvec(self):
        """Return rotation vector (axis * angle)"""
        xyz, w = self._quat[..., :3], self._quat[..., 3:]
        norm_xyz = xyz.norm(dim=-1, keepdim=True)
        angle = 2 * torch.atan2(norm_xyz, w)
        scale = torch.where(
            norm_xyz > 1e-8, angle / norm_xyz, torch.zeros_like(norm_xyz)
        )
        return xyz * scale

    def as_matrix(self):
        """Return 3x3 rotation matrix"""
        x, y, z, w = self._quat.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        xw, yw, zw = x * w, y * w, z * w
        R = torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - zw),
                2 * (xz + yw),
                2 * (xy + zw),
                1 - 2 * (xx + zz),
                2 * (yz - xw),
                2 * (xz - yw),
                2 * (yz + xw),
                1 - 2 * (xx + yy),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        if R.shape[0] == 1:
            return R.squeeze(0)
        return R

    # ----------------- Operations -----------------
    def inv(self):
        """Return inverse rotation"""
        q = self._quat.clone()
        q[..., :3] *= -1
        return Rotation(q)

    def apply(self, vectors, inverse=True, T=None):
        """
        Apply rotation(s) to a set of vectors, optionally using the inverse and translation.

        Parameters
        ----------
        vectors : torch.Tensor
            Shape (N,3)
        inverse : bool
            If True, apply the inverse rotation.
        T : torch.Tensor or None
            Translation per rotation batch. Shape (B,3).
            If last dimension is 2, z-translation is assumed zero.

        Returns
        -------
        rotated : torch.Tensor
            Shape: (N,3) for single rotation, (B,N,3) for batch rotations
        """
        R = self.as_matrix()  # (3,3) or (B,3,3)
        if inverse:
            R = R.transpose(-2, -1)

        N = vectors.shape[0]

        # Apply rotation
        if R.ndim == 2:  # single rotation
            rotated = vectors @ R.T  # (N,3)
        else:  # batch of rotations
            B = R.shape[0]
            vectors_exp = vectors.unsqueeze(0).expand(B, N, 3)  # (B,N,3)
            rotated = torch.einsum("bij,bkj->bki", R, vectors_exp)  # (B,N,3)

        if T is None:
            return rotated
        else:
            return translate_coordinates(rotated, T, inverse=inverse)

    def __mul__(self, other):
        """Compose rotations (quaternion multiplication)"""
        x1, y1, z1, w1 = self._quat.unbind(-1)
        x2, y2, z2, w2 = other._quat.unbind(-1)
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        quat = torch.stack([x, y, z, w], dim=-1)
        return Rotation(quat)


def translate_coordinates(vectors, T, inverse=False):
    """
    Apply translation to points, with broadcasting.

    Parameters
    ----------
    vectors : torch.Tensor
        Shape: (N,3) or (B,N,3)
    T : torch.Tensor
        Shape: (3,), (1,3), or (B,3). Last dim can be 2 or 3.
    inverse : bool
        If True, subtract translation instead of adding.

    Returns
    -------
    translated : torch.Tensor
        Same shape as vectors
    """
    # Pad z=0 if last dim is 2
    if T.shape[-1] == 2:
        T_full = F.pad(T, (0, 1))  # pad last dim with 1 zero
    elif T.shape[-1] == 3:
        T_full = T
    else:
        raise ValueError("T must have last dimension 2 or 3")

    sign = -1 if inverse else 1

    if vectors.ndim == 2:  # single rotation
        if T_full.ndim == 1:
            T_full = T_full.unsqueeze(0)  # (1,3)
        return vectors + sign * T_full
    else:  # batch rotation
        B, N, _ = vectors.shape
        if T_full.ndim == 2:
            if T_full.shape[0] == B:
                T_full = T_full[:, None, :]
            elif T_full.shape[0] == 1:
                T_full = T_full
            else:
                raise ValueError("T shape not compatible with batch size")
        return vectors + sign * T_full


def random_quaternion(batchsize=1, convention="xyzw", device="cpu"):
    """
    Generate uniformly random unit quaternions using Shoemake's method.
    """

    u1, u2, u3 = torch.rand(3, batchsize, device=device)

    sqrt_u1 = torch.sqrt(u1)
    sqrt_1u1 = torch.sqrt(1 - u1)

    q_w = sqrt_1u1 * torch.sin(2 * torch.pi * u2)
    q_x = sqrt_1u1 * torch.cos(2 * torch.pi * u2)
    q_y = sqrt_u1 * torch.sin(2 * torch.pi * u3)
    q_z = sqrt_u1 * torch.cos(2 * torch.pi * u3)

    if convention == "xyzw":
        quats = torch.stack([q_x, q_y, q_z, q_w], dim=-1)
    elif convention == "wxyz":
        quats = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
    else:
        raise ValueError("convention must be 'xyzw' or 'wxyz'")

    if batchsize == 1:
        return quats.squeeze(0)
    return quats


def random_rotvec(batchsize=1, device="cpu"):
    """
    Generate uniformly random rotation vectors using the Rotation3D class.

    Parameters
    ----------
    batchsize : int
        Number of rotation vectors to generate.
    device : str or torch.device
        Device for the output tensor.

    Returns
    -------
    rotvecs : torch.Tensor
        Tensor of shape (batchsize, 3) with rotation vectors (axis * angle).
    """
    quats = random_quaternion(batchsize=batchsize, convention="xyzw", device=device)
    R = Rotation.from_quat(quats)
    rotvecs = R.as_rotvec()

    if batchsize == 1:
        return rotvecs.squeeze(0)
    return rotvecs


def random_rotation_matrix(batchsize=1, device="cpu"):
    """
    Generate uniformly random 3x3 rotation matrices using the Rotation3D class.

    Parameters
    ----------
    batchsize : int
        Number of rotation matrices to generate.
    device : str or torch.device
        Device for the output tensor.

    Returns
    -------
    rotmats : torch.Tensor
        Tensor of shape (batchsize, 3, 3) containing rotation matrices.
    """
    quats = random_quaternion(batchsize=batchsize, convention="xyzw", device=device)
    R = Rotation.from_quat(quats)
    rotmats = R.as_matrix()

    if batchsize == 1:
        return rotmats.squeeze(0)
    return rotmats


def rotate_volume(V, theta, origin="relion", padding_mode="border"):
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
    V_ = F.grid_sample(V, grid, align_corners=align_corners, padding_mode=padding_mode)

    return V_.squeeze(1)  # B x Z x X x Y


def translations_angstrom_to_torch(T, n, voxel_size):
    """
    Builds a batch of normalized translation vectors from rlnOriginXAngst and
    rlnOriginYAngst in starfiles.

    Torch affine matrix uses a normalized coordinate system, where the coordinates
    of each axis ranges from [-1, 1]. Therefore, we need to do a coordinate
    transformation to match this Torch coordinate system for translations.

    Parameters
    ----------
    T : 2D tensor
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
    num = len(T)
    if T.shape[-1] == 2:
        tz = torch.zeros(num, device=T.device)
        T = torch.concat([T, tz[..., None]], dim=-1)
    T *= 2 / n / voxel_size
    return T


def build_affine_matrix(R, T=None):
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
    # old
    # theta = torch.concat([R, T.unsqueeze(2)], dim=-1)

    # new
    Tprime = torch.zeros_like(T)
    Tprime[:, 0] = R[:, 0, 0] * T[:, 0] + R[:, 0, 1] * T[:, 1] + R[:, 0, 2] * T[:, 2]
    Tprime[:, 1] = R[:, 1, 0] * T[:, 0] + R[:, 1, 1] * T[:, 1] + R[:, 1, 2] * T[:, 2]
    Tprime[:, 2] = R[:, 2, 0] * T[:, 0] + R[:, 2, 1] * T[:, 1] + R[:, 2, 2] * T[:, 2]
    theta = torch.concat([R, Tprime.unsqueeze(2)], dim=-1)
    return theta


def rotations_angular_difference(r1, r2, rotation_representation="rotvec"):
    """
    Calculates the smallest angles of rotation needed to a batch of 3D rotations (r1)
    to match another (r2).

    Parameters
    ----------
    r1, r2 : 2D tensors
        Batch of rotation representations (N, ...) each.
    rotation_representation : str
        The rotation representation of the input. Supports only 'quaternion' and
        'rotvec' for now.

    Returns
    -------
    angles : 1D tensor
        The smallest angular difference.

    Notes
    -----
    .. [1] https://math.stackexchange.com/a/4001635

    """
    # use scipy Rotation module
    if rotation_representation == "rotvec":
        r1 = Rotation.from_rotvec(r1)
        r2 = Rotation.from_rotvec(r2)
    elif rotation_representation == "quaternion":
        r1 = Rotation.from_quat(r1)
        r2 = Rotation.from_quat(r2)

    # invert one of them
    r1_inv = r1.inv()

    # convert to rotation matrices
    r1_inv_m = r1_inv.as_matrix()
    r2_m = r2.as_matrix()

    # compute relative angle
    re_m = np.matmul(r1_inv_m, r2_m)
    angles_rad = np.arccos((np.trace(re_m, axis1=-1, axis2=-2) - 1) / 2)
    angles = angles_rad / np.pi * 180
    return angles
