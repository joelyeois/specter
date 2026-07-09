from __future__ import annotations

import torch
import torch.nn.functional as F


class Rotation:
    """
    3D rotation class with support for quaternions, rotation vectors, and rotation matrices.
    Internally stores rotation as a unit quaternion in xyzw format.
    """

    def __init__(self, quat: torch.Tensor, eps: float = 1e-6) -> None:
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
    def from_quat(cls, quat: torch.Tensor, scalar_first: bool = False) -> Rotation:
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
    def from_rotvec(cls, rotvec: torch.Tensor) -> Rotation:
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
    def from_matrix(cls, R: torch.Tensor) -> Rotation:
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
    def as_quat(self, scalar_first: bool = False) -> torch.Tensor:
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

    def as_rotvec(self) -> torch.Tensor:
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

    def as_matrix(self) -> torch.Tensor:
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
    def inv(self) -> Rotation:
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

    def apply(
        self,
        vectors: torch.Tensor,
        inverse: bool = True,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def __mul__(self, other: Rotation) -> Rotation:
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


def rotate_coordinates(vectors: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors by quaternions.

    Parameters
    ----------
    vectors : torch.Tensor
        (N, 3) or (3,) coordinates.
    quat : torch.Tensor
        (B, 4) or (4,) quaternions.

    Returns
    -------
    rotated : torch.Tensor
        Rotated coordinates.
    """
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    R = Rotation.from_quat(quat)
    rotated = R.apply(vectors, inverse=False)
    if rotated.ndim == 3 and rotated.shape[1] == 1:
        rotated = rotated.squeeze(1)
    return rotated


def translate_coordinates(
    vectors: torch.Tensor, T: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
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
