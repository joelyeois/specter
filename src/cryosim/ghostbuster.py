import lightning as L
import torch
import torch.nn as nn
from .imagegenerator import ImageGenerator
from .symmetries import get_rotation_matrices, apply_symmetry
from .fft_tools import fft3, ifft3
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
)
import torch.nn.functional as F


class Ghostbuster(L.LightningModule):
    def __init__(
        self,
        V,
        voxel_size,
        quaternions,
        translations,
        ctf_params,
        energy,
        dose_per_angstrom,
        anisomag=None,
        alpha=0.0,
        defocus_offset=torch.tensor(0.0),
        scattering_model="multislice",
        aberration_model="holography",
        klim=None,
        sparsity=None,
        lr=None,
        lr_R=None,
        lr_T=None,
        lr_D=None,
        lr_decay=0.1,
        scheduler="LambdaLR",
        kmask=None,
        flipcurvature=False,
        fsc_ref=None,
        fsc_mask=None,
        rotate_mode="real",
        symmetry=None,
        symmetry_batchsize=None,
        symmetry_mode="fourier",
        use_cpu_for_symmetry=False,
    ):
        super().__init__()

        # Always use manual optimization to handle masking and multiple optimizers consistently
        self.automatic_optimization = False
        if all(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]):
            print("Non-optimization mode.")
        elif sum(l_rate is None for l_rate in [lr, lr_R, lr_T, lr_D]) == 3:
            print("Single parameter optimization mode.")
        else:
            print("Multi-parameter optimization mode.")

        # optimization parameters
        self.sparsity = sparsity
        self.lr = lr
        self.lr_R = lr_R
        self.lr_T = lr_T
        self.lr_D = lr_D
        self.lr_decay = lr_decay
        self.log_lrs = []
        self.log_total_loss = []
        self.log_sparsity_loss = []
        self.log_norm_loss = []
        self.scheduler = scheduler

        # symmetry parameters
        self.symmetry = symmetry
        self.symmetry_batchsize = symmetry_batchsize
        self.symmetry_mode = symmetry_mode
        if symmetry is not None:
            sym_rot_matrices = get_rotation_matrices(symmetry)
            self.register_buffer("sym_rot_matrices", sym_rot_matrices)
        self.use_cpu_for_symmetry = use_cpu_for_symmetry

        # masks
        self.register_buffer("kmask", kmask)

        # fsc
        if fsc_mask is None:
            fsc_mask = 1
        self.fsc_mask = fsc_mask
        self.fsc_ref = fsc_ref

        # model parameters
        self.dose_per_angstrom = dose_per_angstrom
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.energy = energy
        self.rotate_mode = rotate_mode
        if lr is None:
            self.register_buffer("V", V)
        else:
            self.V = nn.Parameter(V)

        # ctf
        self.ctf_params = {}
        for k, v in ctf_params.items():
            self.register_buffer(k, v)
            self.ctf_params[k] = getattr(self, k)
        if lr_D is None:
            self.register_buffer("defocus_offset", defocus_offset)
        else:
            self.defocus_offset = nn.Parameter(defocus_offset)

        # rotations
        if lr_R is None:
            self.register_buffer("rotations", quaternions)
        else:
            self.rotations = nn.Parameter(quaternions)

        # translations
        if lr_T is None:
            self.register_buffer("translations", translations)
        else:
            self.translations = nn.Parameter(translations)

        # anisomag
        if anisomag is None:
            self.anisomag = anisomag
        else:
            self.register_buffer("anisomag", anisomag)

        # imaging models
        self.flip_curvature = flipcurvature
        self.scattering_model = scattering_model
        self.aberration_model = aberration_model

        # initialize imagegenerator
        self.imagegenerator = ImageGenerator(
            self.V,
            self.voxel_size,
            self.rotations,
            self.translations,
            self.ctf_params,
            self.energy,
            self.dose_per_angstrom,
            ice_model=None,
            scattering_model=self.scattering_model,
            aberration_model=self.aberration_model,
            noise_model=None,
            klim=klim,
            flip_curvature=self.flip_curvature,
            alpha=self.alpha,
        )

    def forward(self, idx):
        image = self.imagegenerator(idx)
        return image

    def symmetrize(self):
        self.V.data = apply_symmetry(
            self.V.data,
            self.sym_rot_matrices,
            batchsize=self.symmetry_batchsize,
            method=self.symmetry_mode,
        )

    def reciprocal_lr_scheduler(self, *args):
        return 1 / (1 + self.lr_decay * self.global_step**0.5)

    def configure_optimizers(self):
        # new. Single parameter optimization only
        if self.lr is not None:
            optimizerV = AdamW([self.V], lr=self.lr)
            # optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9)
            # optimizer = NAdam(self.parameters(), lr=self.lr)
        if self.lr_R is not None:
            optimizerR = AdamW([self.rotations], lr=self.lr_R)
            # optimizerR = Adam([self.rotations], lr=self.lr_R)
            # optimizer = LBFGS([self.rotations], lr=self.lr_R)
        if self.lr_T is not None:
            optimizerT = AdamW([self.translations], lr=self.lr_T)
        if self.lr_D is not None:
            optimizerD = AdamW([self.defocus_offset], lr=self.lr_D)

        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.niter//2, eta_min=1e-6)
        lr_schedulers = []
        if self.lr is not None:
            if self.scheduler == "CosineAnnealingWarmRestarts":
                lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizerV,
                    self.num_training_steps_per_epoch(),
                    eta_min=1e-6,
                    T_mult=2,
                )
            elif self.scheduler == "MultiplicativeLR":
                lr_scheduler = ExponentialLR(optimizerV, 0.999)
            elif self.scheduler == "LambdaLR":
                lr_scheduler = LambdaLR(optimizerV, self.reciprocal_lr_scheduler)
            lr_schedulers.append(lr_scheduler)

        # new. multi-parameter optimization.
        opts = []
        if self.lr is not None:
            opts.append(optimizerV)
        if self.lr_R is not None:
            # opts.append({"optimizerR": optimizerR})
            opts.append(optimizerR)
        if self.lr_T is not None:
            # opts.append({"optimizerT": optimizerT})
            opts.append(optimizerT)
        if self.lr_D is not None:
            # opts.append({"optimizerD": optimizerD})
            opts.append(optimizerD)
        return opts, lr_schedulers

    def _common_step(self, batch, batch_idx):
        # Link parameters to simulator to ensure autograd graph connectivity
        if hasattr(self.imagegenerator, "V"):
            self.imagegenerator.V = self.V
        if hasattr(self.imagegenerator, "quaternions"):
            self.imagegenerator.quaternions = self.rotations

        images, idx = batch
        out = self.forward(idx)

        # mseloss
        loss = F.mse_loss(images, out)
        # loss = F.l1_loss(images, out)
        self.log_norm_loss.append(loss.detach().cpu())

        # sparsity loss
        if self.sparsity is not None:
            sparsity_loss = self.sparsity * torch.mean(torch.abs(self.V))
            loss = loss + sparsity_loss
            self.log_sparsity_loss.append(sparsity_loss.detach().cpu())

        # total loss
        self.log_total_loss.append(loss.detach().cpu())

        return loss, out, images

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()

        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        loss, out, y1 = self._common_step(batch, batch_idx)
        # self.log('train_loss', loss)
        self.log_dict(
            {"train_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Zero gradients
        for opt in opts:
            opt.zero_grad()

        # Backward
        self.manual_backward(loss)

        # Step optimizers
        for opt in opts:
            opt.step()

        # manual optimization
        if not self.automatic_optimization:
            sch = self.lr_schedulers()
            if sch:
                if not isinstance(sch, (list, tuple)):
                    sch = [sch]
                for s in sch:
                    s.step()
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # log lr
        if self.lr is not None:
            self.log_lrs.append(
                self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[
                    0
                ]["lr"]
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.kmask is not None:
            self.V.data = torch.real(ifft3(fft3(self.V.data) * self.kmask))

    def on_train_epoch_end(self):
        # enforce symmetry
        if self.symmetry is not None:
            self.V.data = apply_symmetry(
                self.V.data,
                self.sym_rot_matrices,
                batchsize=self.symmetry_batchsize,
                method=self.symmetry_mode,
            )

    def num_training_steps_per_epoch(self) -> int:
        """Get number of training steps per epoch"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches

        return num_steps
