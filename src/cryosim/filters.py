import torch
from skimage.filters import butterworth
from .fft_tools import fftn, ifftn


def normalize_particles(particles, mask_diameter_pixels=None):
    """
    Normalize particle images using masked mean and standard deviation.

    Computes normalization statistics (mean and standard deviation) using a
    circular mask to exclude background regions, then normalizes the particle
    images to have zero mean and unit variance.

    Parameters
    ----------
    particles : torch.Tensor
        Particle images with shape (N, size, size) where N is the number of
        particles.
    mask_diameter_pixels : int, optional
        Diameter of the circular mask in pixels. Regions outside this diameter
        are excluded from normalization statistics. Default is None, which uses
        the full image size.

    Returns
    -------
    means : torch.Tensor
        Mean value for each particle, shape (N,).
    stds : torch.Tensor
        Standard deviation for each particle, shape (N,).
    normalized_particles : torch.Tensor
        Normalized particle images with shape (N, size, size).
    """
    if mask_diameter_pixels is None:
        mask_diameter_pixels = particles.shape[-1]
    mask = 1 - circle2d(particles.shape[-1], mask_diameter_pixels)
    masked_particles = particles * mask[None, ...]  # Element-wise multiplication
    means = masked_particles.sum(dim=(-1, -2)) / mask.sum()

    vars = (mask * (particles - means[:, None, None]) ** 2).sum(dim=(-1, -2)) / (
        mask.sum()
    )
    stds = torch.sqrt(vars)

    normalized_particles = (particles - means[:, None, None]) / stds[:, None, None]
    return means, stds, normalized_particles


def circle2d(N, d):
    """
    Generate a 2D binary mask of a filled circle.

    Creates an NxN tensor with 1s inside a centered circle of diameter d
    and 0s outside the circle.

    Parameters
    ----------
    N : int
        Size of the output grid in pixels (N x N).
    d : int
        Diameter of the circle in pixels.

    Returns
    -------
    circle : torch.Tensor
        Binary mask with shape (N, N). Values are 1.0 inside the circle
        and 0.0 outside.
    """
    circle = torch.zeros((N, N))
    x = torch.linspace(-1, 1, N)
    y = x
    [X, Y] = torch.meshgrid(x, y, indexing="ij")
    circle[X**2 + Y**2 <= (d / N) ** 2] = 1
    return circle


def butter(images):
    """
    Applies butterworth filter to 2D images.

    Ref: 'https://discuss.cryosparc.com/t/inspect-raw-images-of-particles-of-certain-2-3d-classes/12261/8'

    Parameters
    ----------
    images : 2D or 3D tensor or ndarray
        Can be a single or a batch of 2D particles with shape (N, size, size).

    Returns
    -------
    filtered : 2D or 3D tensor or ndarray
        Butterworth filtered images.

    """

    istensor = False
    n = images.shape[-1]
    if torch.is_tensor(images):
        istensor = True
        images = images.numpy()

    if len(images.shape) == 3:
        channel_axis = 0
    elif len(images.shape) == 2:
        channel_axis = None

    filtered = butterworth(
        images,
        cutoff_frequency_ratio=6 / n,
        high_pass=False,
        order=1,
        channel_axis=channel_axis,
    )

    if istensor:
        return torch.from_numpy(filtered)
    else:
        return filtered


def apply_bfactor(volume, pixel_size, bfactor):
    """
    Apply B-factor blurring to a 3D scattering potential volume.

    Applies temperature factor (B-factor) blurring in Fourier space using
    the formula exp(-B/4 * k²), which simulates thermal motion effects
    on atomic scattering amplitudes.

    Parameters
    ----------
    volume : torch.Tensor
        3D scattering potential volume with shape (n, n, n). Assumed to be
        cubic and real-valued.
    pixel_size : float
        The pixel size in Å.
    bfactor : float
        B-factor (temperature factor). Higher values increase blurring.
        If bfactor=0.0, returns the original volume unchanged.

    Returns
    -------
    newvolume : torch.Tensor
        B-factor blurred volume with same shape as input. Returns real-valued
        tensor if input is real, complex tensor if input is complex.

    Notes
    -----
    The B-factor is applied in Fourier space as:
    F_blurred(k) = F(k) * exp(-B/4 * k²)
    where k is the spatial frequency magnitude.
    """
    if bfactor == 0.0:
        return volume
    else:
        kx = torch.fft.fftshift(torch.fft.fftfreq(volume.shape[-1], pixel_size))
        KZ, KY, KX = torch.meshgrid(kx, kx, kx, indexing="ij")
        k2 = KZ**2 + KY**2 + KX**2
        newvolume = ifftn(fftn(volume) * torch.exp(-bfactor / 4 * k2))

    if torch.is_complex(volume):
        return newvolume
    else:
        return torch.real(newvolume)


def chimera_gaussian_sigma_to_bfactor(sigma):
    """
    Convert ChimeraX Gaussian width to B-factor.

    Converts the Gaussian standard deviation (sigma) used in ChimeraX
    into the equivalent crystallographic B-factor.

    Parameters
    ----------
    sigma : float or torch.Tensor
        Gaussian width (standard deviation) in Å.

    Returns
    -------
    bfactor : float or torch.Tensor
        B-factor, calculated as 8π²σ².

    Notes
    -----
    The relationship between Gaussian width and B-factor is:
    B = 8π²σ²

    This conversion is useful when matching blurring parameters between
    ChimeraX visualization and cryo-EM simulation tools.
    """
    bfactor = 8 * torch.pi**2 * sigma**2
    return bfactor
