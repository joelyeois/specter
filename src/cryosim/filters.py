import torch
from skimage.filters import butterworth
from .fft_tools import fftn, ifftn

def normalize_particles(particles, mask_diameter_pixels=None):
    if mask_diameter_pixels is None:
        mask_diameter_pixels = particles.shape[-1]
    mask = 1 - circle2d(particles.shape[-1], mask_diameter_pixels)
    masked_particles = particles * mask[None, ...]  # Element-wise multiplication
    means = masked_particles.sum(dim=(-1,-2)) / mask.sum()

    vars = (mask * (particles - means[:, None, None]) ** 2).sum(dim=(-1, -2)) / (
                mask.sum()
            )
    stds = torch.sqrt(vars)

    normalized_particles = (particles - means[:, None, None]) / stds[:, None, None]
    return means, stds, normalized_particles

def circle2d(N, d):
    """
    Generates 2D array for a filled-in circle.

    Parameters
    ----------
    N : int
        Length of grid in pixels
    d : int
        Diameter in pixels

    Returns
    -------
    circle : tensor, 2D
        2D array with centered circle
    ....
    """
    circle = torch.zeros((N, N))
    x = torch.linspace(-1, 1, N)
    y = x
    [X, Y] = torch.meshgrid(x, y, indexing="ij")
    circle[X**2 + Y**2 <= (d / N)**2] = 1
    return circle

def butter(images):
    '''
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

    '''

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
        images, cutoff_frequency_ratio=6 / n, high_pass=False, order=1, channel_axis=channel_axis
    )

    if istensor:
        return torch.from_numpy(filtered)
    else:
        return filtered

def apply_bfactor(volume, pixel_size, bfactor):
    '''
    Applies bfactor to a 3D scattering potential volume. I.e., blurs the volume.

    Parameters
    ----------
    volume : 3D
        3D scattering potential, assume cubic with shape (n, n, n).
    pixel_size : float
        The pixel size in angstroms.
    bfactor : float
        The B-factor defined as exp(-B/4 k^2).

    Returns
    -------
    newvolume : 3D tensor
        The b-factor blurred volume.

    '''
    if bfactor == 0.:
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
    '''
    Converts ChimeraX's Gaussian width (sigma) to B-factor.
    '''
    bfactor = 8 * torch.pi**2 * sigma**2
    return bfactor