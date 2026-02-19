import torch
import torch.nn.functional as F
import numpy as np
from .fft_tools import fft3, ifft3
import lightning as L


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
        Create from rotation vector (axis * angle).

        Parameters
        ----------
        rotvec : torch.Tensor
            Rotation vector of shape (..., 3).

        Returns
        -------
        Rotation
            New Rotation instance.
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
        Create from rotation matrix (3x3).

        Parameters
        ----------
        R : torch.Tensor
            Rotation matrix of shape (..., 3, 3).

        Returns
        -------
        Rotation
            New Rotation instance.
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
        """
        Return rotation vector (axis * angle).

        Returns
        -------
        rotvec : torch.Tensor
            Rotation vector of shape (..., 3).
        """
        xyz, w = self._quat[..., :3], self._quat[..., 3:]
        norm_xyz = xyz.norm(dim=-1, keepdim=True)
        angle = 2 * torch.atan2(norm_xyz, w)
        scale = torch.where(
            norm_xyz > 1e-8, angle / norm_xyz, torch.zeros_like(norm_xyz)
        )
        return xyz * scale

    def as_matrix(self):
        """
        Return 3x3 rotation matrix.

        Returns
        -------
        R : torch.Tensor
            Rotation matrix of shape (..., 3, 3).
        """
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
        """
        Return inverse rotation.

        Returns
        -------
        Rotation
            Inverse rotation.
        """
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
        """
        Compose rotations (quaternion multiplication).

        Parameters
        ----------
        other : Rotation
            Other rotation to apply.

        Returns
        -------
        Rotation
            Composed rotation (self * other).
        """
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

    Parameters
    ----------
    batchsize : int, optional
        Number of quaternions to generate. Default 1.
    convention : str, optional
        Quaternion convention ('xyzw' or 'wxyz'). Default 'xyzw'.
    device : str or torch.device, optional
        Device for output tensor. Default 'cpu'.

    Returns
    -------
    quats : torch.Tensor
        Random unit quaternions. Shape (batchsize, 4) or (4,) if batchsize=1.
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


def rotate_volume(
    V, theta, origin="relion", padding_mode="border", align_corners=False
):
    """
    Rotates a single 3D volume based on the batch of 3x4 affine transform matrices.

    Written by Tristan Bepler.

    Parameters
    ----------
    V : torch.Tensor
        The volume to be rotated, must be real-valued. Shape (Z, Y, X).
    theta : torch.Tensor
        Batch of affine matrices, Bx3x4.
        Concatenates a 3x3 rotation matrix and a 3x1 translation vector.
    origin : str, optional
        Convention for the index of the origin of rotation. "relion" defines the
        origin to be at [nz//2, ny//2, nx//2], whereas "center" sets it to
        [(nz + 1) / 2, (ny + 1) / 2, (nx + 1) / 2]. Default "relion".
    padding_mode : str, optional
        Padding mode for grid_sample. Default "border".

    Returns
    -------
    V_rotated : torch.Tensor
         The rotated volumes. Shape (B, Z, Y, X).
    """

    B = theta.size(0)
    nz, ny, nx = V.size()

    # create the coordinate grids depending on origin convention.
    if origin == "relion":
        null_rot = torch.zeros_like(theta)
        null_rot[:, 0, 0] = 1.0
        null_rot[:, 1, 1] = 1.0
        null_rot[:, 2, 2] = 1.0
        grid = F.affine_grid(null_rot, (B, 1, nz, ny, nx), align_corners=align_corners)

        # scale coordinates by (nx-1)/2, (ny-1)/2, (nz-1)/2 to make them isotropic
        scale = torch.tensor(
            [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2],
            device=theta.device,
            dtype=theta.dtype,
        ).view(1, 1, 3)

        # rotate the grid around the center defined by RELION
        cz = nz // 2
        cy = ny // 2
        cx = nx // 2
        center = grid[:, cz, cy, cx]  # (B, 3)
        center = center.unsqueeze(1)
        grid = grid.view(B, nz * ny * nx, 3)

        # Apply isotropic rotation
        grid = (grid - center) * scale
        grid = grid.bmm(theta[..., :-1].transpose(1, 2))
        grid = (grid / scale) + center

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

    return V_.squeeze(1)  # B x Z x Y x X


def rotate_volume_fourier(
    V, theta, origin="relion", padding_mode="border", align_corners=False
):
    # Fourier domain
    V_f = fft3(V)  # Z x X x Y

    # rotate real and imag parts
    V_f_rot_real = rotate_volume(
        V_f.real, theta, origin="relion", padding_mode="border", align_corners=False
    )
    V_f_rot_imag = rotate_volume(
        V_f.imag, theta, origin="relion", padding_mode="border", align_corners=False
    )
    V_f_rot = torch.complex(V_f_rot_real, V_f_rot_imag)
    V_rot = ifft3(V_f_rot)
    return V_rot.real  # B x Z x X x Y


def translations_angstrom_to_torch(T, n, voxel_size):
    """
    Builds a batch of normalized translation vectors from rlnOriginXAngst and rlnOriginYAngst.

    Torch affine matrix uses a normalized coordinate system, where the coordinates
    of each axis ranges from [-1, 1]. Therefore, we need to do a coordinate
    transformation to match this Torch coordinate system for translations.

    Parameters
    ----------
    T : torch.Tensor
        Translation vector of shape (N, 2), built from [rlnOriginXAngst, rlnOriginYAngst]. In Å.
    n : int
        Number of pixels in x/y direction.
    voxel_size : float
        Voxel size in Å.

    Returns
    -------
    T_norm : torch.Tensor
        Batch of Torch normalized translation vectors with shape (N, 3).
    """
    num = len(T)
    if T.shape[-1] == 2:
        tz = torch.zeros(num, device=T.device)
        T = torch.concat([T, tz[..., None]], dim=-1)
    T_norm = T * 2 / n / voxel_size
    return T_norm


def build_affine_matrix(R, T=None):
    if R.ndim == 2:
        R = R.unsqueeze(0)
    if T is not None and T.ndim == 1:
        T = T.unsqueeze(0)
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
    R : torch.Tensor
        Batch of rotation matrices with shape (N, 3, 3).
    T : 2D tensor
        Batch of Torch normalized translation vectors with shape (N, 3). Note that this is not the same as the shifts directly from CryoSPARC/RELION starfiles. Those shifts must be normalized using translations_angstrom_to_torch.

    Returns
    -------
    theta : torch.Tensor
        Batch of affine matrices with shape (N, 3, 4).
    """
    if T is None:
        T = R.new_zeros(R.shape[0], 3)

    Tprime = torch.zeros_like(T)
    Tprime[:, 0] = R[:, 0, 0] * T[:, 0] + R[:, 0, 1] * T[:, 1] + R[:, 0, 2] * T[:, 2]
    Tprime[:, 1] = R[:, 1, 0] * T[:, 0] + R[:, 1, 1] * T[:, 1] + R[:, 1, 2] * T[:, 2]
    Tprime[:, 2] = R[:, 2, 0] * T[:, 0] + R[:, 2, 1] * T[:, 1] + R[:, 2, 2] * T[:, 2]
    theta = torch.concat([R, Tprime.unsqueeze(2)], dim=-1)
    return theta


def rotations_angular_difference(r1, r2, rotation_representation="rotvec"):
    """
    Calculates the smallest angles of rotation needed for a batch of 3D rotations (r1) to match another (r2).

    Parameters
    ----------
    r1 : torch.Tensor
        Batch of rotation representations (N, ...).
    r2 : torch.Tensor
        Batch of rotation representations (N, ...).
    rotation_representation : str, optional
        The rotation representation of the input. Supports only 'quaternion' and
        'rotvec'. Default 'rotvec'.

    Returns
    -------
    angles : torch.Tensor
        The smallest angular difference in degrees.

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


class VolumeRotator(L.LightningModule):
    """
    3D volume rotator with cached base grid and RELION / PyTorch center conventions.
    """

    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        origin: str = "relion",
        align_corners: bool = False,
        padding_mode: str = "border",
        mode: str = "real",
        init_base_grid=True,
    ):
        """
        Initialize a 3D VolumeRotator with cached base grid and rotation center.

        This class supports rotating 3D volumes in either real space or Fourier space.
        The base grid and rotation center are cached as buffers for efficiency.

        Parameters
        ----------
        nz, ny, nx : int
            Dimensions of the volume along the Z, Y, and X axes.
        origin : str, optional
            Convention for the rotation origin. Options:
            - "relion": sets the origin at [nz//2, ny//2, nx//2] following RELION convention.
            - "center": sets the origin at the center of the volume using PyTorch convention.
            Default is "relion".
        align_corners : bool, optional
            Passed to `torch.nn.functional.affine_grid` and `grid_sample`. Determines
            how the normalized coordinates are aligned with the voxel corners.
            Default is False.
        padding_mode : str, optional
            Passed to `torch.nn.functional.grid_sample`. Options include "zeros", "border", or "reflection".
            Default is "border".
        mode : str, optional
            Default rotation mode for `forward`. Options:
            - "real": rotate in real space.
            - "fourier": rotate in Fourier space.
            Default is "real".

        Raises
        ------
        ValueError
            If `origin` is not one of ("relion", "center") or `mode` is not one of ("real", "fourier").

        Attributes
        ----------
        base_grid : torch.Tensor (buffer)
            Identity sampling grid for the volume, used to construct per-batch grids.
        center : torch.Tensor (buffer)
            The voxel coordinates of the rotation center.
        """
        super().__init__()

        # Validate inputs
        if origin not in ("relion", "center"):
            raise ValueError(f"Unknown origin: {origin}. Must be 'relion' or 'center'.")
        if mode not in ("real", "fourier"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'real' or 'fourier'.")

        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.origin = origin
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.mode = mode  # default rotation mode

        # -------------------------------
        # Rotation center
        # -------------------------------
        # Calculate center based on dimensions and conventions
        if origin == "relion":
            cz, cy, cx = nz // 2, ny // 2, nx // 2
        else:
            cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2

        if self.align_corners:
            center = torch.tensor(
                [
                    2 * cx / (nx - 1) - 1,
                    2 * cy / (ny - 1) - 1,
                    2 * cz / (nz - 1) - 1,
                ]
            )
        else:
            center = torch.tensor(
                [
                    2 * (cx + 0.5) / nx - 1,
                    2 * (cy + 0.5) / ny - 1,
                    2 * (cz + 0.5) / nz - 1,
                ]
            )
        self.register_buffer("center", center.view(1, 1, 3))

        if init_base_grid:
            self._build_base_grid()

        # -------------------------------
        # Precompute XY grid for slice sampling
        # -------------------------------
        y = torch.arange(ny)
        x = torch.arange(nx)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy_grid = torch.stack([xx, yy], dim=-1).float()  # (Y, X, 2)
        self.register_buffer("xy_grid", xy_grid)  # buffer, reused for all slice calls

    # ------------------------------------------------------------------
    # Core grid construction
    # ------------------------------------------------------------------
    def _build_base_grid(self) -> torch.Tensor:
        """
        Build base grid.
        """
        eye = torch.eye(3, 4).unsqueeze(0)  # (1, 3, 4)
        base_grid = F.affine_grid(
            eye, (1, 1, self.nz, self.ny, self.nx), align_corners=self.align_corners
        )
        self.register_buffer("base_grid", base_grid)

    def _build_grid(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Build a sampling grid from cached base grid and affine parameters.

        theta: (B, 3, 4)
        Returns: (B, Z, Y, X, 3)
        """
        B = theta.shape[0]
        R = theta[..., :3]  # (B, 3, 3)
        t = theta[..., 3]  # (B, 3)

        if not hasattr(self, "base_grid"):
            self._build_base_grid()
        grid = self.base_grid.expand(B, -1, -1, -1, -1)
        grid = grid.view(B, -1, 3)

        # scale factors to make grid isotropic
        scale = torch.tensor(
            [(self.nx - 1) / 2, (self.ny - 1) / 2, (self.nz - 1) / 2],
            device=theta.device,
            dtype=theta.dtype,
        ).view(1, 1, 3)

        if self.origin == "relion":
            # Recenter to RELION origin and apply isotropic rotation
            grid = (grid - self.center) * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + self.center
            grid = grid + t.unsqueeze(1)
        else:
            # PyTorch center convention with isotropic rotation
            grid = grid * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + t.unsqueeze(1)

        grid = grid.view(B, self.nz, self.ny, self.nx, 3)
        return grid

    # ------------------------------------------------------------------
    # Real-space rotation
    # ------------------------------------------------------------------
    def rotate_real(self, V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate a volume in real space.
        V: (Z, Y, X)
        theta: (B, 3, 4)
        Returns: (B, Z, Y, X)
        """
        B = theta.shape[0]
        grid = self._build_grid(theta)

        V = V.unsqueeze(0).unsqueeze(1)  # (1, 1, Z, Y, X)
        V = V.expand(B, 1, self.nz, self.ny, self.nx)

        V_rot = F.grid_sample(
            V,
            grid,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )

        return V_rot.squeeze(1)  # (B, Z, Y, X)

    # ------------------------------------------------------------------
    # Fourier-space rotation
    # ------------------------------------------------------------------
    def rotate_fourier(self, V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate a volume in Fourier space via real/imag decomposition.
        """
        V_f = fft3(V)  # complex, (Z, Y, X)

        V_f_rot_real = self.rotate_real(V_f.real, theta)
        V_f_rot_imag = self.rotate_real(V_f.imag, theta)

        V_f_rot = torch.complex(V_f_rot_real, V_f_rot_imag)
        V_rot = ifft3(V_f_rot)

        return V_rot.real  # (B, Z, Y, X)

    # ------------------------------------------------------------------
    # Forward with optional per-call mode override
    # ------------------------------------------------------------------
    def forward(
        self, V: torch.Tensor, theta: torch.Tensor, mode: str = None
    ) -> torch.Tensor:
        """
        Forward pass with optional mode override.
        mode: "real" or "fourier"; defaults to self.mode
        """
        if mode is None:
            mode = self.mode

        if mode not in ("real", "fourier"):
            raise ValueError(f"Unknown mode: {mode}. Must be 'real' or 'fourier'.")

        if mode == "real":
            return self.rotate_real(V, theta)
        else:
            return self.rotate_fourier(V, theta)

    # ------------------------------------------------------------------
    # Sample a single tilted slice
    # ------------------------------------------------------------------
    def sample_rotated_slices(
        self,
        V: torch.Tensor,
        theta: torch.Tensor,
        slice_indices: torch.Tensor,
        roi_center: tuple = None,
        roi_size: tuple = None,
        padding_mode: str = None,
    ):
        """
        Sample multiple Z-slices from a rotated volume, restricted to a ROI in XY.
        The slice_index=0 corresponds to the center of the volume.

        Parameters
        ----------
        V : (Z,Y,X) torch.Tensor
            Input 3D volume.
        theta : (B,3,4) torch.Tensor
            Affine rotation matrix.
        slice_indices : (K,) or list or int
            One or more slice indices along Z of the rotated volume, relative to the center.
        roi_center : tuple (cy, cx)
            Center pixel of the region of interest in Y,X coordinates of the original volume.
            Defaults to full volume center.
        roi_size : tuple (ny_roi, nx_roi)
            Size of the region to sample.
            Defaults to full volume size.
        padding_mode : str
            Passed to grid_sample. Defaults to self.padding_mode.

        Returns
        -------
        slices_roi : (B, K, ny_roi, nx_roi) torch.Tensor
            Interpolated 2D slices of the rotated volume, cropped to ROI.
        """
        if padding_mode is None:
            padding_mode = self.padding_mode

        nz, ny, nx = self.nz, self.ny, self.nx
        B = theta.shape[0]

        if not isinstance(slice_indices, torch.Tensor):
            slice_indices = torch.as_tensor(slice_indices, device=V.device)

        if slice_indices.ndim == 0:
            slice_indices = slice_indices.unsqueeze(0)

        K = slice_indices.shape[0]

        # -------------------------------
        # Determine ROI
        # -------------------------------
        if roi_size is None:
            roi_size = (ny, nx)
        if roi_center is None:
            roi_center = (ny // 2, nx // 2)

        ny_roi, nx_roi = roi_size
        cy, cx = roi_center

        # -------------------------------
        # Isotropic scale factors (normalized to physical units)
        # -------------------------------
        dtype = V.dtype
        device = V.device

        # Scale factor to convert normalized [-1, 1] to pixel units [-half, half]
        scale = torch.tensor(
            [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2], device=device, dtype=dtype
        ).view(1, 1, 3)

        # -------------------------------
        # Calculate pixel-unit coordinates (relative to center)
        # -------------------------------
        z_rel = slice_indices.to(dtype=dtype, device=device)  # (K,)
        y_rel_center = float(cy - ny // 2)
        x_rel_center = float(cx - nx // 2)

        y_rel_grid = torch.arange(ny_roi, device=device, dtype=dtype) - (ny_roi // 2)
        x_rel_grid = torch.arange(nx_roi, device=device, dtype=dtype) - (nx_roi // 2)
        yy_rel, xx_rel = torch.meshgrid(y_rel_grid, x_rel_grid, indexing="ij")

        # Global pixel-unit relative coordinates for the K slices
        # last dim is (x, y, z)
        x_pix = (x_rel_center + xx_rel).unsqueeze(0).expand(K, -1, -1)
        y_pix = (y_rel_center + yy_rel).unsqueeze(0).expand(K, -1, -1)
        z_pix = z_rel.view(K, 1, 1).expand(-1, ny_roi, nx_roi)

        points_pix = torch.stack([x_pix, y_pix, z_pix], dim=-1).unsqueeze(
            0
        )  # (1, K, ny_roi, nx_roi, 3)

        # -------------------------------
        # Apply rotation in pixel space
        # -------------------------------
        R = theta[..., :3]  # (B, 3, 3)
        t = theta[..., 3]  # (B, 3) normalized

        points_pix = points_pix.expand(B, -1, -1, -1, -1)
        points_pix_flat = points_pix.reshape(B, -1, 3)

        if self.origin == "relion":
            # 1. Normalize points to [-1, 1] relative to V center
            points_norm = points_pix_flat / scale + self.center
            # 2. Isotropic rotation (Scale -> Rotate -> Unscale)
            grid = (points_norm - self.center) * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + self.center
            # 3. Translate
            grid = grid + t.unsqueeze(1)
        else:
            # PyTorch center convention
            points_norm = points_pix_flat / scale
            grid = points_norm * scale
            grid = grid @ R.transpose(1, 2)
            grid = (grid / scale) + t.unsqueeze(1)

        grid = grid.view(B, K, ny_roi, nx_roi, 3)

        # -------------------------------
        # Sample volume
        # -------------------------------
        if V.ndim == 3:
            V_in = V.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1, -1)
        elif V.ndim == 4:  # Assume (B, Z, Y, X)
            V_in = V.unsqueeze(1)
        else:  # (B, 1, Z, Y, X)
            V_in = V

        slices_roi = F.grid_sample(
            V_in, grid, align_corners=self.align_corners, padding_mode=padding_mode
        )

        return slices_roi[:, 0]  # (B, K, ny_roi, nx_roi)
