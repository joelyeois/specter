import torch

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))

def kgrid(sh, delta):
    """
    Constructs |k| grid. Can be 2D or 3D.
    
    Parameters
    ----------
    sh : tuple
        Shape of grid.
    delta : float or tuple
        Pixel size. If float, assumes same pixel size for all dimensions.
    
    Returns
    -------
    k : tensor
        |k| grid of shape sh.
    """
    if len(sh) == 2:
        ny, nx = sh
        if isinstance(delta, int):
            dy = dx = delta
        else:
            dy, dx = delta
        kx = torch.fft.fftshift(torch.fft.fftfreq(nx, dx))
        ky = torch.fft.fftshift(torch.fft.fftfreq(ny, dy))
        kyy, kxx = torch.meshgrid(kx, ky, indexing="ij")
        k2 = kxx**2 + kyy**2
        k = torch.sqrt(k2)
    elif len(sh) == 3:
        nz, ny, nx = sh
        if len(delta) == 3:
            dz, dy, dx = delta
        else:
            dz = dy = dx = delta
        kx = torch.fft.fftshift(torch.fft.fftfreq(nx, dx))
        ky = torch.fft.fftshift(torch.fft.fftfreq(ny, dy))
        kz = torch.fft.fftshift(torch.fft.fftfreq(nz, dz))
        kxx, kyy, kzz = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = kxx**2 + kyy**2 + kzz**2
        k = torch.sqrt(k2)
    return k


def fresnel(wavelength, kgrid, distance,):
    """
    Fresnel propagator in Fourier space.
    
    Parameters
    ----------
    wavelength : float
        Wavelength.
    kgrid : tensor
        Grid of |k| in Fourier domain. Must be 1/[wavelength unit].
    distance : float
        Distance to propagate. Must be [wavelength unit].
    
    Returns
    -------
    propagator : tensor
        Propagator of same shape as kgrid.
    """
    propagator = torch.exp(1j * torch.pi * wavelength * k**2 * distance)
    return propagator

def angularspectrum(wavelength, k, distance,):
    """
    Angular spectrum propagator in Fourier space. Refer to Goodman's Introduction
    to Fourier Optics, Chapter 3, Eq 3-74.
    
    Parameters
    ----------
    wavelength : float
        Wavelength.
    kgrid : tensor
        Grid of |k| in Fourier domain. Must be 1/[wavelength unit].
    distance : float
        Distance to propagate. Must be [wavelength unit].
    
    Returns
    -------
    propagator : tensor
        Propagator of same shape as kgrid.
    """
    k_aperture = kgrid**2 < 1/wavelength
    propagator = torch.exp(1j * 2 * np.pi / wavelength * distance * torch.sqrt(1 - wavelength**2 * k**2)) * k_aperture
    return propagator

def propagate(wave, wavelength, kgrid, distance, method='fresnel'):
    if method == 'fresnel':
        prop = fresnel(wavelength, kgrid, distance)
    elif method == 'angularspectrm':
        prop = angularspectrm(wavelength, kgrid, distance)
    p_wave = ifft2(fft2(wave) * prop)
    return p_wave