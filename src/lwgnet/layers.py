#!/usr/bin/env python3
import torch
from torch import nn
from torch.fft import fftshift, ifftshift


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.fill_(1.0 + 1.0j)


class forwardLayerWF(nn.Module):
    def __init__(self, opts, CTF):

        super().__init__()
        self.register_buffer("P", opts["P"])
        #self.N = opts["N"]
        self.N_obj = opts["N_obj"]
        self.register_buffer("mu", torch.tensor(0.01))

    def forward(self, x, I, Ns):
        b, ch, h, w = I.shape
        P = self.P.repeat(b, 1, 1)
        Ns = Ns.squeeze()
        cen0 = tuple(n // 2 for n in self.N_obj)
        grad_xk = torch.zeros(
            (b, ch, self.N_obj[0], self.N_obj[1]),
            dtype=torch.cfloat,
            device=x.device,
            requires_grad=False,
        )
        psi_k = torch.zeros((b, h, w), requires_grad=False)
        xFT = torch.zeros(x.shape, requires_grad=False)
        Z = torch.zeros((xFT.shape), device=x.device, requires_grad=False)

        #Nh, Nv = self.N

        # Estimate forward measurement loss and gradient
        for k in range(ch):
            xFT = fftshift(torch.fft.fft2(x), (1, 2))
            # cenx = cen0[0] + Ns[k, 0]
            # ceny = cen0[1] + Ns[k, 1]
            # kxl = torch.round(cenx - self.N[0])
            # kxh = torch.round(cenx + self.N[0])
            # kyl = torch.round(ceny - self.N[1])
            # kyh = torch.round(ceny + self.N[1])
            psi_k = xFT * P[k]

            inv_psik = torch.fft.ifft2(ifftshift(psi_k, (1, 2))) * ((h * w) / (self.N_obj[0] * self.N_obj[1]))
            I_estim = torch.abs(inv_psik) ** 2
            diff = diff = (I_estim - I[:, k, :, :]) * inv_psik

            Z = torch.zeros((xFT.shape), dtype=torch.cfloat, device=x.device, requires_grad=False)
            Z = (fftshift(torch.fft.fft2(diff), (1, 2))) * P[k]
            invZ = torch.fft.ifft2(ifftshift(Z, (1, 2))) * ((self.N_obj[0] * self.N_obj[1]) / (h * w))
            grad_xk[:, k, :, :] = invZ
            x = x - self.mu * invZ

        return grad_xk
