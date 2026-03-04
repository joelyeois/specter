import lightning as L
import torch
import torch.nn.functional as F
from rich.progress import track
from torchinterp1d import interp1d

from . import pdbtools, potential
from .array_utils import grid_3d, radial_grid_3d, radial_profile_3d, real_to_kgrid_3d
from .atom import kirkland_atomic_potential_3d, lobato_atomic_potential_3d
from .fft_tools import fft3, fftconvolve

avogadro = 6.02214076e23
density_of_amorphous_ice = 0.94  # [g/cm3]
molar_mass_of_water = 18.01528  # [g/mol]
ndensity_of_amorphous_ice = (
    density_of_amorphous_ice * avogadro / molar_mass_of_water * 1e-24
)  # [particles / A3]


def rfftn(array):
    return torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(array, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2),
    )


def torch_peak_local_max(image, min_distance=1, num_peaks=None):
    """
    Find local maxima in batched 3D images and return fixed number of peaks per batch.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (B, D, H, W)
    min_distance : int
        Minimum separation between peaks (voxels)
    num_peaks : int
        Number of peaks to return per batch (must be <= total peaks in each batch)

    Returns
    -------
    peaks : torch.LongTensor, shape (B, num_peaks, 3)
        Peak coordinates (z, y, x) for each batch.
    """
    B, D, H, W = image.shape
    x = image.unsqueeze(1)  # (B, 1, D, H, W)
    k = 2 * min_distance + 1
    pooled = F.max_pool3d(x, kernel_size=k, stride=1, padding=min_distance)
    mask = (x == pooled).squeeze(1)  # (B, D, H, W)

    # Flatten spatial dims
    flat_mask = mask.view(B, -1)
    flat_image = image.view(B, -1)

    # Mask non-maxima
    flat_image_masked = flat_image.clone()
    flat_image_masked[~flat_mask] = -float("inf")

    if num_peaks is None:
        num_peaks = flat_mask.sum(dim=1).min().item()  # take min available peaks

    # Top-k per batch
    topk_vals, topk_idx = flat_image_masked.topk(num_peaks, dim=1)

    # Convert flat indices back to 3D coords
    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_ = topk_idx % W

    peaks = torch.stack([z, y, x_], dim=2)  # (B, num_peaks, 3)
    return peaks


class Icemaker(L.LightningModule):
    """
    Generates 3D ice volumes with water-like molecular structure based on
    molecular dynamics simulations. Provides methods to load simulation data,
    compute radial averages, and iteratively generate ice volumes that match
    a target Fourier amplitude kernel.
    """

    def __init__(
        self,
        dx=0.5,
        n=200,
        nz=None,
        chunk_size=None,
        progressbars=True,
        parameterization="kirkland",
    ):
        """
        Initialize the Icemaker.

        Parameters
        ----------
        dx : float
            Voxel size in Angstroms.
        n : int
            Number of voxels in x and y dimensions.
        nz : float
            Ice thickness in angstroms.
        device : str or torch.device
            Device for tensor operations (default: 'cuda').
        """
        super().__init__()

        # load 3D radial average of mdsim data
        self.saved_data_path = "../ice-data/mdsim_f_radial_avg_400x400x400_0.25A.pt"
        self.mdsim_dx = 0.25
        self.mdsim_n = 400
        self.mdsim_dk = 1 / self.mdsim_n / self.mdsim_dx
        self.get_mdsim_f_radial_avg(self.saved_data_path)
        self.chunk_size = chunk_size
        self.progressbars = progressbars
        self.parameterization = parameterization

        # self.ice_thickness = ice_thickness
        # if ice_thickness is None or ice_thickness < n * dx:
        #     self.nz = n
        #     if ice_thickness is not None and ice_thickness < n * dx:
        #         print(
        #             "Ice thickness smaller than particle size. Using minimum thickness."
        #         )
        #     self.ice_thickness = n * dx
        # else:
        #     self.nz = int(ice_thickness // dx)
        if nz is None:
            self.nz = n
        else:
            self.nz = nz
        self.dx = dx
        self.dk = 1 / n / dx
        self.n = n
        self.dv = dx**3
        self.nv = n**2 * self.nz
        self.v = self.dv * self.nv
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.v)

        # create k-space coordinates grid
        kx = torch.fft.fftshift(torch.fft.fftfreq(n, dx))
        ky = kx
        kz = torch.fft.fftshift(torch.fft.fftfreq(self.nz, dx))
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        self.register_buffer("K", torch.sqrt(KX**2 + KY**2 + KZ**2))

        # pre-compute ice kernel for algorithm
        self.interpolate_mdsim_f_kernel()

        self.register_buffer("ice_kernel", self.create_ice_kernel())

    def get_mdsim(self, filepath, trim_size=100, startframe=10, endframe=101):
        """
        Load MD simulation dump and convert atomic coordinates into voxel grid.

        Parameters
        ----------
        filepath : str
            Path to the MD simulation dump file.
        trim_size : int
            Maximum half-size of the cube to retain around particle center.
        startframe : int
            Frame index to start processing.
        endframe : int
            Frame index to stop processing.
        """
        self.get_mdsim_file(filepath)
        mdsim_ice_deltas = []

        x, y, z, X, Y, Z = grid_3d(self.mdsim_n, self.mdsim_dx)

        self.mdsim_ice_coordinates = []
        for frame in track(
            self.mdsim_frame_indexes[startframe:endframe],
            disable=not (self.progressbars),
        ):
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
            self.mdsim_ice_coordinates.append(centered_coords)
        self.mdsim_ice_deltas = torch.stack(mdsim_ice_deltas)

    def get_mdsim_file(self, filepath):
        """
        Reads MD simulation dump file into memory and finds timestep indices.

        Parameters
        ----------
        filepath : str
            Path to the MD simulation dump file.
        """
        with open(filepath) as f:
            self.lines = f.readlines()

        self.mdsim_frame_indexes = [
            i for i, x in track(enumerate(self.lines)) if x == "ITEM: TIMESTEP\n"
        ]

    def get_coordinates_from_frame(
        self, start_line_number, lines=None, no_atoms=128000
    ):
        """
        Parse atom coordinates from a given frame in MD dump.

        Parameters
        ----------
        start_line_number : int
            Line number where atom coordinates start.
        lines : list[str] or None
            Pre-loaded file lines. If None, uses `self.lines`.
        no_atoms : int
            Number of atoms to read.

        Returns
        -------
        coords : torch.Tensor, shape (no_atoms, 3)
            Atomic coordinates (x, y, z) for the frame.
        """
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
        """
        Trim coordinates to a cube of given size centered at origin.

        Parameters
        ----------
        coords : torch.Tensor, shape (N, 3)
            Coordinates to trim.
        trim_size : float
            Side length of the cube to retain (Angstroms).

        Returns
        -------
        trimmed_coords : torch.Tensor, shape (M, 3)
            Coordinates within the cube.
        """
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

    def get_mdsim_averaged_f_kernel(self, filepath, source="torch"):
        """
        Compute or load the 3D Fourier amplitude of the ice volume.

        Parameters
        ----------
        filepath : str
            Path to load precomputed Fourier amplitude tensor.
        source : str, {'dump', 'torch'}
            If 'dump', compute FFT from MD simulation dump.
            If 'torch', load from file.
        """
        if source == "dump":
            self.get_mdsim(filepath, trim_size=100)
            self.mdsim_ice_deltas_f = []
            for mdsim_ice_delta in track(
                self.mdsim_ice_deltas, disable=not (self.progressbars)
            ):
                self.mdsim_ice_deltas_f.append(fft3(mdsim_ice_delta))
            self.mdsim_ice_deltas_f = torch.stack(self.mdsim_ice_deltas_f)
            self.mdsim_ice_deltas_f = torch.mean(
                torch.abs(self.mdsim_ice_deltas_f), dim=0
            )

        elif source == "torch":
            self.mdsim_ice_deltas_f = torch.load(filepath).to(self.device)

    def get_mdsim_f_radial_avg(self, saved_data_path=None):
        """
        Compute or load radial average of MD simulation Fourier amplitudes.

        Parameters
        ----------
        saved_data_path : str or None
            Optional path to precomputed radial average. If None, compute from
            `self.mdsim_ice_deltas_f`.
        """
        if saved_data_path is not None:
            mdsim_f_radial_avg = torch.load(saved_data_path)
            self.register_buffer("mdsim_f_radial_avg", mdsim_f_radial_avg)
        else:
            # compute 3D radial average of mdsim data
            self.mdsim_f_radial_avg = radial_profile_3d(self.mdsim_ice_deltas_f)
        mdsim_radial_k = torch.arange(len(self.mdsim_f_radial_avg)) * self.mdsim_dk
        self.register_buffer("mdsim_radial_k", mdsim_radial_k)

    def create_initial_ice_volume(self, batchsize=1):
        """
        Create a random initial 3D ice volume with specified number of molecules.

        Parameters
        ----------
        batchsize : int
            Number of ice volumes to generate.
        device : str
            Default to 'cpu' since this is a fast function. Saves GPU memory.

        Returns
        -------
        ice_vol_init : torch.Tensor, shape (batchsize, nz, n, n) or (nz, n, n)
            Binary tensor with 1 where ice molecules are placed.
        """

        # Preallocate batch tensor
        ice_vol_init = torch.zeros(
            batchsize, self.nz * self.n * self.n, device=self.device
        )

        # Randomly select indices for each batch volume
        idx = torch.randint(
            0, self.nz * self.n * self.n, (batchsize, self.n_ice_molecules)
        )

        # Scatter 1s at chosen indices
        batch_indices = (
            torch.arange(batchsize).unsqueeze(1).expand(-1, self.n_ice_molecules)
        )
        ice_vol_init[batch_indices, idx] = 1.0

        # Reshape to (B, nz, n, n)
        ice_vol_init = ice_vol_init.view(batchsize, self.nz, self.n, self.n)
        return ice_vol_init

    def interpolate_mdsim_f_kernel(self):
        """
        Generate a 3D Fourier amplitude kernel for ice generation by
        interpolating MD simulation radial averages.

        Returns
        -------
        None
            Updates `self.interp_radial_k` and `self.interp_f_radial_avg`.
        """

        # interpolate, exclude DC
        interp = interp1d(
            self.mdsim_radial_k[1:], self.mdsim_f_radial_avg[1:], self.K.ravel()
        )

        # replace DC value
        interp_f_kernel = interp.reshape(self.nz, self.n, self.n)
        interp_f_kernel[self.nz // 2, self.n // 2, self.n // 2] = self.n_ice_molecules
        self.register_buffer("interp_f_kernel", interp_f_kernel)

        # register half kernel for rfftn
        self.register_buffer(
            "interp_f_halfkernel",
            torch.flip(interp_f_kernel[:, :, : self.n // 2 + 1], dims=[2]),
        )

        # compute 3D radial average of interp data
        self.register_buffer("interp_f_radial_avg", radial_profile_3d(interp_f_kernel))
        self.register_buffer(
            "interp_radial_k", torch.arange(len(self.interp_f_radial_avg)) * self.dk
        )

    def generate_ice_deltas(
        self, niter=5, min_distance=1.9, add_extra_molecules=True, batchsize=1
    ):
        """
        Iteratively generate ice volume using Fourier amplitude kernel.

        Parameters
        ----------
        niter : int
            Maximum number of iterations.
        min_distance : float
            Minimum separation between molecules (Angstroms).
        add_extra_molecules : bool
            If True, randomly add extra molecules to satisfy density.
        batchsize : int
            Number of ice volumes to generate.

        Returns
        -------
        None
            Updates `self.current_ice_vol` and `self.ice_coordinates`.
        """

        self.batchsize = batchsize
        self.register_buffer("current_icedeltas", self.ice_vol_init.clone())
        self.niter = niter
        self.min_distance = min_distance

        self.frob_norm = []
        self.n_extra_atoms = []

        for i in track(
            range(niter),
            description="Running ice algorithm",
            transient=True,
            disable=not (self.progressbars),
        ):
            ice_vol_f = rfftn(self.current_icedeltas)

            # amplitude multiplication
            ice_vol_f *= self.interp_f_halfkernel.unsqueeze(0)

            new_ice = torch.abs(self.irfftn(ice_vol_f))
            peaks = torch_peak_local_max(
                new_ice,
                num_peaks=self.n_ice_molecules,
                min_distance=int(min_distance / self.dx),
            )

            # ice_vol shape: (B, nz, n, n)
            self.register_buffer(
                "ice_vol",
                torch.zeros(batchsize, self.nz, self.n, self.n, device=self.device),
            )
            num_peaks = peaks.shape[1]  # must be fixed per batch

            # batch indices
            self.register_buffer(
                "batch_idx", torch.arange(batchsize).view(-1, 1).expand(-1, num_peaks)
            )

            # unpack coordinates
            z_idx = peaks[:, :, 0]
            y_idx = peaks[:, :, 1]
            x_idx = peaks[:, :, 2]

            # set ice voxels
            self.ice_vol[
                self.batch_idx.flatten(),
                z_idx.flatten(),
                y_idx.flatten(),
                x_idx.flatten(),
            ] = 1

            # ice_vol[peaks[:, 0], peaks[:, 1], peaks[:, 2]] = 1

            ## Add extra molecules to satisfy density. But this leads to bad results.
            # if add_extra_molecules:
            #     if len(peaks) < self.n_ice_molecules:
            #         n_extra = self.n_ice_molecules - len(peaks)
            #         self.n_extra_atoms.append(n_extra)

            #         # Find all empty locations
            #         zero_idx = (ice_vol == 0).nonzero(as_tuple=False)

            #         # Randomly choose n_extra of them
            #         perm = torch.randperm(zero_idx.shape[0], device=ice_vol.device)
            #         chosen = zero_idx[perm[:n_extra]]

            #         # Mark them as filled
            #         ice_vol[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1

            mse = F.mse_loss(self.current_icedeltas.cpu(), self.ice_vol.cpu())
            self.frob_norm.append(mse)
            self.current_icedeltas = self.ice_vol
            if i > 1 and torch.isclose(self.frob_norm[-1], self.frob_norm[-2]):
                break
        self.n_peaks = peaks.shape[1]
        self.ice_coordinates = peaks.cpu()
        self.frob_norm = torch.tensor(self.frob_norm)
        self.n_extra_atoms = torch.tensor(self.n_extra_atoms)

    def create_ice_kernel(self):
        # create super-sampled (ss) coordinate system
        ssn, ssdx, ssf = potential.compute_supersampling_parameters(self.dx)
        # set original convention to torch to avoid singularity at origin.
        sR = radial_grid_3d(ssn, ssdx, convention="torch")

        # for binning super-sampled grids to main volume grid.
        avgpool3d = torch.nn.AvgPool3d(ssf, stride=ssf)

        if self.parameterization == "kirkland":
            pot = kirkland_atomic_potential_3d(8, sR)
        elif self.parameterization == "lobato":
            pot = lobato_atomic_potential_3d(8, sR)
        elif self.parameterization == "shryov":
            # from params_cat.json, 'O(HH)'
            params = torch.tensor(
                [
                    [0.3131, 0.8722],
                    [0.8102, 4.9669],
                    [0.9812, 14.1666],
                    [-0.5997, 64.1638],
                    [-0.1519, 121.3711],
                ]
            )
            # Separate columns: a_i, b_i
            a = (
                params[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # shape (3,1,1,1)
            b = params[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            k_xyz = real_to_kgrid_3d(sR)
            k2 = k_xyz**2
            k2 = k2.unsqueeze(0)  # shape (1, Nx, Ny, Nz)

            s1_f = torch.sum(a * torch.exp(-b * k2 / 4), 0)
            dkx = k_xyz[1, 0, 0] - k_xyz[0, 0, 0]
            dky = k_xyz[0, 1, 0] - k_xyz[0, 0, 0]
            dkz = k_xyz[0, 0, 1] - k_xyz[0, 0, 0]
            pot = -torch.abs(fft3(s1_f)) * dkx * dky * dkz  # need to negate

        return avgpool3d(pot[None, None]).squeeze() * self.dx

    def generate_ice(self, batchsize=1):
        # initialize
        ice_vol_init = self.create_initial_ice_volume(batchsize=batchsize)
        self.register_buffer("ice_vol_init", ice_vol_init)

        # run algorithms
        self.generate_ice_deltas(batchsize=batchsize)

        # convolve with ice kernel
        self.register_buffer("icecubes", torch.zeros_like(self.current_icedeltas))
        # for i in tqdm(range(batchsize), desc='Computing batches of icecubes', leave=False):
        for i in track(
            range(batchsize),
            description="Computing batches of icecubes",
            transient=True,
            disable=not (self.progressbars),
        ):
            self.icecube = fftconvolve(
                self.current_icedeltas[i], self.ice_kernel, mode="same"
            )
            self.icecubes[i] = self.icecube
        return self.icecubes

    def irfftn(self, array):
        return torch.fft.fftshift(
            torch.fft.irfftn(
                torch.fft.ifftshift(array, dim=(-3, -2)),
                s=(self.nz, self.n, self.n),
                dim=(-3, -2, -1),
            ),
            dim=(-3, -2, -1),
        )

    def generate_big_ice(self, shape):
        B, nz, ny, nx = shape
        num_z = int(torch.ceil(torch.as_tensor(nz) / self.nz))
        num_y = int(torch.ceil(torch.as_tensor(ny) / self.n))
        num_x = int(torch.ceil(torch.as_tensor(nx) / self.n))
        N = B * num_z * num_y * num_x
        num_blocks_per_B = num_z * num_y * num_x

        # generate N ice cubes
        if self.chunk_size is None:
            ices = self.generate_ice(N)

            # assemble ice
            big_ice = torch.empty(B, num_z * self.nz, num_y * self.n, num_x * self.n)
            idx = 0
            for ib in range(B):
                for iz in range(num_z):
                    for iy in range(num_y):
                        for ix in range(num_x):
                            big_ice[
                                ib,
                                iz * self.nz : (iz + 1) * self.nz,
                                iy * self.n : (iy + 1) * self.n,
                                ix * self.n : (ix + 1) * self.n,
                            ] = ices[idx]
                            idx += 1
            # trim
            return big_ice[:B, :nz, :ny, :nx]
        else:
            # generate batch of ice positions
            big_ice = torch.empty(B, num_z * self.nz, num_y * self.n, num_x * self.n)

            idx = 0
            # for start in tqdm(range(0, N, self.chunk_size), desc='Generate ice positions', leave=False):
            for start in track(
                range(0, N, self.chunk_size),
                description="Generate ice positions",
                transient=True,
                disable=not (self.progressbars),
            ):
                end = min(start + self.chunk_size, N)
                batchsize = end - start

                # initialize & generate
                ice_vol_init = self.create_initial_ice_volume(batchsize=batchsize)
                self.register_buffer("ice_vol_init", ice_vol_init)
                self.generate_ice_deltas(batchsize=batchsize)

                # get current batch (move to CPU if needed)
                batch_icedeltas = (
                    self.current_icedeltas.cpu()
                )  # shape (batchsize, self.nz, self.n, self.n)

                # directly insert into big_ice
                for b in range(batchsize):
                    global_idx = start + b

                    ib = global_idx // num_blocks_per_B
                    local_idx = global_idx % num_blocks_per_B

                    iz = local_idx // (num_y * num_x)
                    iy = (local_idx % (num_y * num_x)) // num_x
                    ix = local_idx % num_x

                    big_ice[
                        ib,
                        iz * self.nz : (iz + 1) * self.nz,
                        iy * self.n : (iy + 1) * self.n,
                        ix * self.n : (ix + 1) * self.n,
                    ] = batch_icedeltas[b]

                    idx += 1

            # resolve boundary conflicts where ice are too near
            min_distance_px = int(self.min_distance / self.dx)
            for ib in range(B):
                big_ice[ib] = clean_block_boundaries(
                    big_ice[ib], (self.nz, self.n, self.n), min_distance_px
                )

            # perform batchwise fft
            for ib in range(B):
                # for iz in tqdm(range(num_z), desc='Ice convolution', leave=False):
                for iz in track(
                    range(num_z),
                    description="Ice convolution",
                    transient=True,
                    disable=not (self.progressbars),
                ):
                    for iy in range(num_y):
                        for ix in range(num_x):
                            big_ice[
                                ib,
                                iz * self.nz : (iz + 1) * self.nz,
                                iy * self.n : (iy + 1) * self.n,
                                ix * self.n : (ix + 1) * self.n,
                            ] = fftconvolve(
                                big_ice[
                                    ib,
                                    iz * self.nz : (iz + 1) * self.nz,
                                    iy * self.n : (iy + 1) * self.n,
                                    ix * self.n : (ix + 1) * self.n,
                                ].to(self.ice_kernel.device),
                                self.ice_kernel,
                                mode="same",
                            ).cpu()
            return big_ice[:B, :nz, :ny, :nx]


class NaiveIcemaker(L.LightningModule):
    def __init__(self, dx, n, nz=None, progressbars=True):
        """
        Creates ice through random choice. Given a volume, we calculate the number
        of ice molecules that should populate the volume based on the density of
        amorphous ice. Random choice is then used to determine the position of
        ice molecules, which are then dressed with the scattering kernel of ice.

        Parameters
        ----------
        dx : float
            Pixel size in angstroms.
        n: int
            Number of pixels in xy-axis. Assumes a square field-of-view.
        nz: float
            Specifices the thickness of ice in Angstroms. Typically 100–1000 A. Must
            be same or larger than FOV of the particle.
        """
        super().__init__()

        self.dx = dx
        self.n = n

        # self.ice_thickness = ice_thickness
        # if ice_thickness is None or ice_thickness < n * dx:
        #     self.nz = n
        #     if ice_thickness is not None and ice_thickness < n * dx:
        #         print(
        #             "Ice thickness smaller than particle size. Using minimum thickness."
        #         )
        #     self.ice_thickness = n * dx
        # else:
        #     self.nz = int(ice_thickness // dx)
        if nz is None:
            self.nz = n
        else:
            self.nz = nz

        self.dv = dx**3  # voxel volume
        self.nv = n**2 * self.nz  # number of voxels
        self.total_vol = self.nv * self.dv  # total volume
        self.n_ice_molecules = int(ndensity_of_amorphous_ice * self.total_vol)
        self.register_buffer("ice_kernel", self.create_ice_kernel())

        self.progressbars = progressbars

    def create_initial_ice_volume(self):
        # slowest, without duplicates
        # ice_idx = np.random.choice(self.n**3, self.n_ice_molecules, replace=False)

        # second fastest, without duplicates
        # ice_idx = torch.randperm(self.n**3)
        # ice_idx = ice_idx[:self.n_ice_molecules]

        # fastest, with duplicates
        ice_idx = torch.randint(0, self.nv, (self.n_ice_molecules,))

        ice_vol_init = torch.zeros(self.nv, device=self.device)
        ice_vol_init[ice_idx] = 1
        ice_vol_init = ice_vol_init.reshape(self.nz, self.n, self.n)  # z, y, x
        return ice_vol_init

    def create_ice_kernel(self, sn=28):
        # sample a 28x28 grid to represent kernel first.
        # 4xbin down to 7x7, centerd on atom origin
        sx = (torch.arange(sn) - (sn - 1) / 2) * self.dx / 4
        sZ, sY, sX = torch.meshgrid(sx, sx, sx, indexing="ij")
        sR = torch.sqrt(sX**2 + sY**2 + sZ**2)

        # see cryosim for details.
        a0 = 0.529  # Bohr radius, [Angstrom]
        e = 14.4  # electron charge, [V-Angstrom]
        c1 = 2 * (torch.pi**2) * a0 * e
        c2 = 2 * (torch.pi ** (5 / 2)) * a0 * e

        # P params for Oxygen. See Kirkland Appendix C.
        P = torch.tensor(
            [
                [3.39969204e-001, 3.81570280e-001, 3.07570172e-001, 3.81571436e-001],
                [1.30369072e-001, 1.91919745e001, 8.83326058e-002, 7.60635525e-001],
                [1.96586700e-001, 2.07401094e000, 9.96220028e-004, 3.03266869e-002],
            ]
        )
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

    def generate_ice(self, batchsize=1):
        icecubes = torch.zeros(batchsize, self.nz, self.n, self.n, device=self.device)
        for i in range(batchsize):
            self.icedeltas = self.create_initial_ice_volume()
            self.icecube = fftconvolve(self.icedeltas, self.ice_kernel, mode="same")
            icecubes[i] = self.icecube
        return icecubes


def remove_deltas_based_on_density(slab, expected_number=None, dx=None):
    if expected_number is None:
        if dx is None:
            raise ValueError("dx must be specified.")
        else:
            expected_number = int(slab.numel() * dx**3 * ndensity_of_amorphous_ice)

    # Step 1: Get indices of all 1s
    ones_indices = (slab == 1).nonzero(as_tuple=False)  # shape: [num_ones, 3]
    current_number = len(ones_indices)
    # Step 2: Randomly pick N indices
    if current_number > expected_number:
        N = current_number - expected_number
        selected_idx = ones_indices[torch.randperm(len(ones_indices))[:N]]

        # Step 3: Set selected positions to 0
        slab[selected_idx[:, 0], selected_idx[:, 1], selected_idx[:, 2]] = 0
        return slab
    else:
        return slab


def clean_block_boundaries(
    bigblock: torch.Tensor, shape: tuple, min_dist: int
) -> torch.Tensor:
    D, H, W = bigblock.shape
    d, h, w = shape  # block sizes

    nD = D // d
    nH = H // h
    nW = W // w

    n_density_bigblock = bigblock.sum() / bigblock.numel()

    # Depth boundaries
    for bd in range(1, nD):
        start = bd * d - min_dist * 2
        end = bd * d + min_dist * 2
        start = max(start, 0)
        end = min(end, D)
        slab = bigblock[start:end, :, :]
        bigblock[start:end, :, :] = remove_deltas_based_on_density(
            slab, expected_number=int(n_density_bigblock * slab.numel())
        )

    # Height boundaries
    for bh in range(1, nH):
        start = bh * h - min_dist * 2
        end = bh * h + min_dist * 2
        start = max(start, 0)
        end = min(end, H)
        slab = bigblock[:, start:end, :]
        bigblock[:, start:end, :] = remove_deltas_based_on_density(
            slab, expected_number=int(n_density_bigblock * slab.numel())
        )

    # Width boundaries
    for bw in range(1, nW):
        start = bw * w - min_dist * 2
        end = bw * w + min_dist * 2
        start = max(start, 0)
        end = min(end, W)
        slab = bigblock[:, :, start:end]
        bigblock[:, :, start:end] = remove_deltas_based_on_density(
            slab, expected_number=int(n_density_bigblock * slab.numel())
        )

    return bigblock
