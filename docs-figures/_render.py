"""Shared 3D isosurface rendering helpers for docs-figures/ scripts.

Not domain math -- just the matplotlib/marching-cubes plumbing common to
every hero shot, factored out once both membrane shape backends needed it.
"""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

TEAL = np.array([0.0, 0.45, 0.45])


def shade_mesh(
    verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, base_color: np.ndarray
):
    """Poly3DCollection with Gouraud-like Lambertian shading -- shades each
    face by its vertices' averaged, renormalized marching-cubes NORMAL
    (gradient-based, smoothly varying across the surface), not the face's
    own flat geometric normal (cross product of its own edges): flat
    geometric normals are piecewise-constant per triangle and make even a
    smooth isosurface look faceted under directional light, since
    neighboring triangles' normals then jump discretely at each edge."""
    tris = verts[faces]
    face_normals = normals[faces].mean(axis=1)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    light_dir = np.array([0.4, 0.35, 0.85])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(-face_normals @ light_dir, 0.0, 1.0)
    intensity = 0.55 + 0.45 * intensity
    colors = base_color[None, :] * intensity[:, None]
    return Poly3DCollection(tris, facecolor=colors, edgecolor="none", antialiased=False)


def isosurface_axes(
    ax, phi_field: np.ndarray, spacing_angstrom: float, base_color: np.ndarray
):
    verts, faces, normals, _ = marching_cubes(
        phi_field, level=0.0, spacing=(spacing_angstrom,) * 3
    )
    mesh = shade_mesh(verts, faces, normals, base_color)
    ax.add_collection3d(mesh)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo).max()
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
