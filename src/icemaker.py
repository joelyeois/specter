import numpy as np
import pdbtools
import potential
import torch
from scipy.interpolate import CubicSpline
from skimage.feature import peak_local_max
from tqdm import tqdm
from fft_tools import fftconvolve

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / A3]

fftn = lambda array: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(array)))
ifftn = lambda array: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(array)))


class Icemaker:
    """Creates ice with water rings. Slow."""

    def __init__(self, dx=0.5, n=200):
        self.mdsim_dx = dx
        self.mdsim_n = n
        self.mdsim_dk = 1 / self.mdsim_n / self.mdsim_dx

    def get_mdsim(self, filepath, trim_size=100):
        self.get_mdsim_file(filepath)
        mdsim_ice_deltas = []

        x, y, z, X, Y, Z = potential.coordinate_grid_3d(
            (self.mdsim_n, self.mdsim_n, self.mdsim_n),
            (self.mdsim_dx, self.mdsim_dx, self.mdsim_dx),
        )

        for frame in tqdm(self.mdsim_frame_indexes[10:]):
            coordstart = frame + 9
            coords = self.get_coordinates_from_frame(coordstart)

            # recenters coordinates onto origin (0,0,0)
            center = pdbtools.center_of_particle(coords)
            centered_coords = coords - center.reshape(1, -1)
            centered_coords = self.trim_coordinates(
                centered_coords, trim_size=trim_size
            )

            mdsim_ice_delta = torch.zeros(self.mdsim_n, self.mdsim_n, self.mdsim_n)
            for cc in centered_coords:
                xi, yi, zi = potential.nearest_index(x, y, z, cc[0], cc[1], cc[2])
                mdsim_ice_delta[zi, yi, xi] = 1

            mdsim_ice_deltas.append(mdsim_ice_delta)

        self.mdsim_ice_deltas = torch.stack(mdsim_ice_deltas)

    def get_mdsim_file(self, filepath):
        with open(filepath) as f:
            self.lines = f.readlines()

        self.mdsim_frame_indexes = [
            i for i, x in tqdm(enumerate(self.lines)) if x == "ITEM: TIMESTEP\n"
        ]

    def get_coordinates_from_frame(
        self, start_line_number, lines=None, no_atoms=128000
    ):
        if lines is None:
            lines = self.lines

        coords = torch.zeros(no_atoms, 3)
        for i, s in enumerate(lines[start_line_number : start_line_number + no_atoms]):
            id, typ, x, y, z = s.split()
            coords[i, 0] = float(x)
            coords[i, 1] = float(y)
            coords[i, 2] = float(z)
        return coords

    def trim_coordinates(self, coords, trim_size=100):
        trimmed_coords = []
        for co in coords:
            if not (
                torch.abs(co[0]) > trim_size // 2
                or torch.abs(co[1]) > trim_size // 2
                or torch.abs(co[2]) > trim_size // 2
            ):
                trimmed_coords.append(co)
        trimmed_coords = torch.stack(trimmed_coords)
        return trimmed_coords

    def get_mdsim_averaged_f_kernel(self, filepath, source='torch'):
        if source == 'dump':
            self.get_mdsim(filepath, trim_size=100)
            self.mdsim_ice_deltas_f = []
            for mdsim_ice_delta in tqdm(self.mdsim_ice_deltas):
                self.mdsim_ice_deltas_f.append(fftn(mdsim_ice_delta))
            self.mdsim_ice_deltas_f = torch.stack(self.mdsim_ice_deltas_f)
            self.mdsim_ice_deltas_f = torch.mean(torch.abs(self.mdsim_ice_deltas_f), dim=0)
        
        elif source == 'torch':
            self.mdsim_ice_deltas_f = torch.load(filepath)

    def create_initial_ice_volume(self, n, dx):
        dv = dx**3
        nv = n**3
        total_vol = nv * dv  # A^3
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * total_vol)

        ice_idx = np.random.choice(n**3, self.n_ice_molecules, replace=False)
        ice_vol_init = torch.zeros(n**3)
        ice_vol_init[ice_idx] = 1
        ice_vol_init = ice_vol_init.reshape(n, n, n)
        return ice_vol_init

    def interpolate_mdsim_f_kernel(self, n, dx):
        self.interp_n = n
        self.interp_dx = dx
        self.interp_dk = 1 / n / dx

        # compute 3D radial average of mdsim data
        self.mdsim_f_radial_avg = radial_profile_3d(self.mdsim_ice_deltas_f)
        self.mdsim_radial_k = torch.arange(len(self.mdsim_f_radial_avg)) * self.mdsim_dk

        # create interpolation grid
        kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
        ky = kx.clone()
        kz = kx.clone()
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        K = torch.sqrt(KX**2 + KY**2 + KZ**2)

        # interpolate, exclude DC
        spline = CubicSpline(self.mdsim_radial_k[1:], self.mdsim_f_radial_avg[1:])
        interp = torch.from_numpy(spline(K.ravel()))

        # replace DC value
        self.interp_f_kernel = interp.reshape(n, n, n)
        self.interp_f_kernel[n // 2, n // 2, n // 2] = self.mdsim_ice_deltas_f[
            self.mdsim_n // 2, self.mdsim_n // 2, self.mdsim_n // 2
        ]

        # compute 3D radial average of interp data
        self.interp_f_radial_avg = radial_profile_3d(self.interp_f_kernel)
        self.interp_radial_k = (
            torch.arange(len(self.interp_f_radial_avg)) * self.interp_dk
        )

    def generate_ice(self, n=None, dx=None, niter=5, min_distance=3):
        if n is None:
            n = self.interp_n
        if dx is None:
            dx = self.interp_dx

        self.ice_vol_init = self.create_initial_ice_volume(n=n, dx=dx)
        self.current_ice_vol = self.ice_vol_init.clone()
        self.niter = niter
        self.min_distance = min_distance

        self.frob_norm = []
        self.n_extra_atoms = []

        for _ in tqdm(range(niter)):
            prev_ice_vol = self.current_ice_vol.clone()
            ice_vol_f = fftn(self.current_ice_vol)

            # amplitude multiplication
            ice_vol_f *= self.interp_f_kernel

            # amplitude replacement
            # ice_vol_f = self.interp_f_kernel * torch.exp(1j * torch.angle(ice_vol_f))

            new_ice = torch.abs(ifftn(ice_vol_f))
            peaks = peak_local_max(
                new_ice.numpy(),
                num_peaks=self.n_ice_molecules,
                min_distance=min_distance,
                exclude_border=False,
            )

            ice_vol = torch.zeros(n, n, n)
            for peak in peaks:
                ice_vol[*peak] = 1

            # add more ice if needed
            if len(peaks) < self.n_ice_molecules:
                n_extra = self.n_ice_molecules - len(peaks)
                self.n_extra_atoms.append(n_extra)
                zero_idx = (ice_vol == 0).nonzero()
                idx = np.random.choice(len(zero_idx), n_extra, replace=False)
                for i in idx:
                    ice_vol[*zero_idx[i]] = 1

            self.frob_norm.append(torch.mean(torch.abs(self.current_ice_vol - ice_vol) ** 2))
            self.current_ice_vol = ice_vol
        self.frob_norm = torch.tensor(self.frob_norm)
        self.n_extra_atoms = torch.tensor(self.n_extra_atoms)

class NaiveIcemaker:
    """Creates ice through random choice. Fast."""

    def __init__(self, dx, n):
        self.dx = dx
        self.n = n
        self.dv = dx**3
        self.nv = n**3
        self.total_vol = self.nv * self.dv  # A^3
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.total_vol)
        self.ice_kernel = self.create_ice_kernel()

    def create_initial_ice_volume(self):
        #slowest, without duplicates
        # ice_idx = np.random.choice(self.n**3, self.n_ice_molecules, replace=False)

        #second fastest, without duplicates
        # ice_idx = torch.randperm(self.n**3)
        # ice_idx = ice_idx[:self.n_ice_molecules]
        
        #fastest, with duplicates
        ice_idx = torch.randint(0, self.n**3, (self.n_ice_molecules,))
        
        ice_vol_init = torch.zeros(self.n**3)
        ice_vol_init[ice_idx] = 1
        ice_vol_init = ice_vol_init.reshape(self.n, self.n, self.n)
        return ice_vol_init

    def create_ice_kernel(self, sn=28):
        #sample a 28x28 grid to represent kernel first.
        #4xbin down to 7x7, centerd on atom origin
        sx = (torch.arange(sn) - (sn - 1) / 2) * self.dx/4
        sZ, sY, sX = torch.meshgrid(sx, sx, sx, indexing="ij")
        sR = torch.sqrt(sX**2 + sY**2 + sZ**2)

        #see cryosim for details.
        a0 = 0.529  # Bohr radius, [Angstrom]
        e = 14.4  # electron charge, [V-Angstrom]
        c1 = 2 * (torch.pi**2) * a0 * e
        c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

        #P params for Oxygen. See Kirkland Appendix C.
        P = torch.tensor([[3.39969204e-001, 3.81570280e-001, 3.07570172e-001, 3.81571436e-001],
                          [1.30369072e-001, 1.91919745e+001, 8.83326058e-002, 7.60635525e-001],
                          [1.96586700e-001, 2.07401094e+000, 9.96220028e-004, 3.03266869e-002]])
        P = P.T
        # tile scattering factors to match r_xy grid
        P = P[:, :, None, None, None].expand((4, 3) + sR.shape)

        s1 = c1 * torch.sum(
            P[0] / sR * torch.exp(-2 * torch.pi * sR * torch.sqrt(P[1])), 0
        )
        s2 = c2 * torch.sum(
            P[2] * P[3] ** (-3 / 2) * torch.exp(-(torch.pi**2) * (sR**2) / P[3]), 0
        )
        pot = s1 + s2

        avgpool3d = torch.nn.AvgPool3d(4, stride=4)
        return avgpool3d(pot[None, None]).squeeze() * self.dx

    def generate_random_icecube(self, batchsize=1):
        if batchsize == 1:
            self.icedeltas = self.create_initial_ice_volume()
            self.icecube = fftconvolve(self.icedeltas, self.ice_kernel, mode='same')
            return self.icecube
        else:
            icecubes = torch.zeros(batchsize, self.n, self.n, self.n)
            for i in range(batchsize):
                self.icedeltas = self.create_initial_ice_volume()
                self.icecube = fftconvolve(self.icedeltas, self.ice_kernel, mode='same')
                icecubes[i] = self.icecube
            return icecubes

def radial_profile_3d(data, center=None, return_r=False):
    m, n, o = np.shape(data)
    if center is None:
        center = 0, 0, 0
    x = np.arange(n) - (n) // 2 + center[0]
    y = np.arange(m) - (m) // 2 + center[1]
    z = np.arange(o) - (m) // 2 + center[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    r = r.round().astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    if return_r:
        return r, radialprofile
    else:
        return radialprofile