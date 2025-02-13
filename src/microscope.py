import lightning as L
import torch
import numpy as np

class TransferFunction(L.LightningModule):
    def __init__(self, freqs, energy=200, alpha=0.0, usectf=False):
        super().__init__()
        self.register_buffer("freqs", freqs)
        self.wavelength = energy_to_wavelength(energy)
        self.alpha = torch.tensor(alpha)
        self.usectf = usectf

    def aberration(self, cs, dfu, dfv, dfang):
        w = self.wavelength
        y = self.freqs[..., 0]
        x = self.freqs[..., 1]
        ang = torch.arctan2(y, x).unsqueeze(0)  # returns radians, not degrees?
        k2 = (x**2 + y**2).unsqueeze(0)
        dfu = dfu.unsqueeze(1).unsqueeze(2)
        dfv = dfv.unsqueeze(1).unsqueeze(2)
        dfang = dfang.unsqueeze(1).unsqueeze(2)
        cs = cs.unsqueeze(1).unsqueeze(2)
        df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
        gamma = torch.pi * w * k2 * (0.5 * cs * w**2 * k2 - df)
        return gamma

    def transfer(self, cs, dfu, dfv, dfang):
        gamma = self.aberration(cs, dfu, dfv, dfang)
        if self.usectf:
            return torch.sqrt(1 - self.alpha**2) * torch.sin(
                gamma
            ) - self.alpha * torch.cos(gamma)
        else:
            return torch.exp(-1j * gamma)

    def forward(self, exitwave, cs, dfu, dfv, dfang):
        f = self.transfer(cs, dfu, dfv, dfang)
        return ifft2(fft2(exitwave) * f)