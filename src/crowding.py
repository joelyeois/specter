import torch

# def poisson_disk_neighbors(particle_diameter, n_points, max_radius=5.0,
#                            max_attempts=5000):
#     """
#     Generate 2D coordinates around the origin using Poisson-disk sampling.

#     Each coordinate is at least a distance `particle_diameter` away from the
#     origin and from all other coordinates. Coordinates are sampled within a
#     circle of radius `max_radius`.

#     Parameters
#     ----------
#     particle_diameter : float
#         Minimum center-to-center distance between coordinates (diameter of
#         each particle).
#     n_points : int
#         Number of coordinates to generate.
#     max_radius : float, optional
#         Maximum distance from the origin to sample coordinates (default: 5.0).
#     max_attempts : int, optional
#         Maximum number of candidate coordinates to try before stopping
#         (default: 5000).

#     Returns
#     -------
#     coordinates : torch.Tensor, shape (k, 2)
#         Tensor of 2D coordinates generated. `k` may be less than `n_points`
#         if sampling fails due to high density constraints.
#     """
#     coordinates = torch.empty((0, 2))

#     for _ in range(max_attempts):
#         if coordinates.shape[0] >= n_points:
#             break

#         # Sample candidate coordinate in polar coordinates
#         angle = torch.rand(1) * 2 * torch.pi
#         radius = particle_diameter + (max_radius - particle_diameter) * torch.rand(1)
#         candidate_coord = torch.stack(
#             (radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1
#         ).view(1, 2)  # ensure shape is (1, 2)

#         # Skip candidate if too close to origin
#         if candidate_coord.norm() < particle_diameter:
#             continue

#         # Accept candidate if sufficiently far from all existing coordinates
#         if coordinates.shape[0] == 0 or torch.all(
#             torch.norm(coordinates - candidate_coord, dim=1) >= particle_diameter
#         ):
#             coordinates = torch.cat([coordinates, candidate_coord], dim=0)

#     return coordinates


def poisson_disk_neighbors(particle_diameter, n_points=torch.inf,
                             max_radius=5.0, k=30):
    """
    Fast 2D Poisson-disk sampling with Bridson's algorithm.
    Guarantees origin is the first point.

    Parameters
    ----------
    particle_diameter : float
        Minimum center-to-center spacing.
    n_points : int
        Total points to generate (including origin).
    max_radius : float, optional
        Max radius from origin.
    k : int, optional
        Number of candidate points to try per active point.

    Returns
    -------
    pts : (m,2) torch.Tensor
        Sampled 2D coordinates. First row is [0,0].
    """
    r = particle_diameter
    R = max_radius
    r2 = r ** 2

    pts = [torch.zeros(2)]       # first point at origin
    active = [0]

    # we exclude the origin in our counter
    n_points += 1
    
    while active and len(pts) < n_points:
        idx = torch.randint(len(active), (1,)).item()
        center = pts[active[idx]]

        # generate k candidates in parallel
        theta = torch.rand(k) * 2 * torch.pi
        radius = r + r * torch.rand(k)
        candidates = center.unsqueeze(0) + torch.stack(
            (radius * torch.cos(theta), radius * torch.sin(theta)), dim=1
        )

        # reject candidates outside the circle
        mask = candidates.pow(2).sum(dim=1) <= R * R
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            active.pop(idx)
            continue

        # vectorized distance check against all existing points
        if pts:
            pts_tensor = torch.stack(pts)
            diff = candidates[:, None, :] - pts_tensor[None, :, :]
            dist2 = (diff ** 2).sum(dim=2)   # shape (num_candidates, num_pts)
            min_dist2, _ = dist2.min(dim=1)
            candidates = candidates[min_dist2 >= r2]

        if candidates.shape[0] > 0:
            pts.append(candidates[0])
            active.append(len(pts) - 1)
        else:
            active.pop(idx)

    if n_points == torch.inf:
        return torch.stack(pts[1:])
    else:
        return torch.stack(pts[1:n_points])

