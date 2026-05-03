import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FISTA_NET, ISTA_NET, FBPCONVNET


class ProximalMappingNetwork(nn.Module):
    def __init__(self, n_filters: int = 32):
        super().__init__()
        Nf = n_filters
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  Nf, 3, padding=1, bias=False),
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(Nf, Nf, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(Nf, 1,  3, padding=1, bias=False),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, r: torch.Tensor, theta: torch.Tensor):
        z        = self.encode(r)
        z_thresh = F.relu(z - theta) - F.relu(-z - theta)
        x_out    = F.relu(self.decode(z_thresh) + r)
        return x_out, z_thresh


class FISTANet(nn.Module):
    def __init__(
        self,
        A_matrix:   torch.Tensor,
        n_stages:   int = FISTA_NET["n_stages"],
        n_filters:  int = FISTA_NET["n_filters"],
        image_size: int = 64,
    ):
        super().__init__()
        self.Ns       = n_stages
        self.img_size = image_size

        self.register_buffer("A",       A_matrix.float())
        self.register_buffer("W_tilde", self._compute_W_tilde(A_matrix))

        self.prox_net = ProximalMappingNetwork(n_filters)

        self.w1 = nn.Parameter(torch.tensor(FISTA_NET["init_w1"]))
        self.c1 = nn.Parameter(torch.tensor(FISTA_NET["init_c1"]))
        self.w2 = nn.Parameter(torch.tensor(FISTA_NET["init_w2"]))
        self.c2 = nn.Parameter(torch.tensor(FISTA_NET["init_c2"]))
        self.w3 = nn.Parameter(torch.tensor(FISTA_NET["init_w3"]))
        self.c3 = nn.Parameter(torch.tensor(FISTA_NET["init_c3"]))

    @staticmethod
    def _compute_W_tilde(A: torch.Tensor) -> torch.Tensor:
        return A.float().T

    def _mu(self, k: int) -> torch.Tensor:
        return F.softplus(-F.softplus(-self.w1) * k + self.c1)

    def _theta(self, k: int) -> torch.Tensor:
        return F.softplus(-F.softplus(-self.w2) * k + self.c2)

    def _rho(self, k: int) -> torch.Tensor:
        w3  = F.softplus(self.w3)
        num = F.softplus(w3 * k + self.c3) - F.softplus(w3 + self.c3)
        return (num / F.softplus(w3 * k + self.c3).clamp(1e-8)).clamp(0, 1)

    def forward(self, b: torch.Tensor, x0: torch.Tensor):
        B, _, H, W = x0.shape
        N          = H * W
        x_prev     = x0
        y          = x0
        ints       = []

        for k in range(1, self.Ns + 1):
            mu    = self._mu(k)
            theta = self._theta(k)
            rho   = self._rho(k)
            res   = (self.A @ y.view(B, N).T).T - b
            r     = y - mu * (self.W_tilde @ res.T).T.view(B, 1, H, W)
            x, z  = self.prox_net(r, theta)
            y     = x + rho * (x - x_prev)
            x_prev = x
            ints.append((x, z))

        return x, ints

    def get_learned_params(self):
        mus, thetas, rhos = [], [], []
        with torch.no_grad():
            for k in range(1, self.Ns + 1):
                mus.append(self._mu(k).item())
                thetas.append(self._theta(k).item())
                rhos.append(self._rho(k).item())
        return mus, thetas, rhos

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ISTANet(nn.Module):
    def __init__(
        self,
        A_matrix:   torch.Tensor,
        n_stages:   int = ISTA_NET["n_stages"],
        n_filters:  int = ISTA_NET["n_filters"],
        image_size: int = 64,
    ):
        super().__init__()
        self.Ns       = n_stages
        self.img_size = image_size
        self.register_buffer("A", A_matrix.float())
        self.prox_net = ProximalMappingNetwork(n_filters)
        self.mus    = nn.ParameterList(
            [nn.Parameter(torch.tensor(ISTA_NET["init_mu"]))    for _ in range(n_stages)])
        self.thetas = nn.ParameterList(
            [nn.Parameter(torch.tensor(ISTA_NET["init_theta"])) for _ in range(n_stages)])

    def forward(self, b: torch.Tensor, x0: torch.Tensor):
        B, _, H, W = x0.shape
        N          = H * W
        x          = x0
        ints       = []
        for k in range(self.Ns):
            r    = x - self.mus[k] * (
                (self.A.T @ ((self.A @ x.view(B, N).T).T - b).T).T
            ).view(B, 1, H, W)
            x, z = self.prox_net(r, self.thetas[k])
            ints.append((x, z))
        return x, ints

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _UNetBlock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.b(x)


class FBPConvNet(nn.Module):
    def __init__(self, base_ch: int = FBPCONVNET["base_ch"]):
        super().__init__()
        b          = base_ch
        self.pool  = nn.MaxPool2d(2)
        self.e1    = _UNetBlock(1,     b)
        self.e2    = _UNetBlock(b,     b * 2)
        self.e3    = _UNetBlock(b * 2, b * 4)
        self.e4    = _UNetBlock(b * 4, b * 8)
        self.bn    = _UNetBlock(b * 8, b * 16)
        self.u4    = nn.ConvTranspose2d(b * 16, b * 8,  2, stride=2)
        self.d4    = _UNetBlock(b * 16, b * 8)
        self.u3    = nn.ConvTranspose2d(b * 8,  b * 4,  2, stride=2)
        self.d3    = _UNetBlock(b * 8,  b * 4)
        self.u2    = nn.ConvTranspose2d(b * 4,  b * 2,  2, stride=2)
        self.d2    = _UNetBlock(b * 4,  b * 2)
        self.u1    = nn.ConvTranspose2d(b * 2,  b,      2, stride=2)
        self.d1    = _UNetBlock(b * 2,  b)
        self.out   = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d  = self.bn(self.pool(e4))
        d  = self.d4(torch.cat([self.u4(d), e4], dim=1))
        d  = self.d3(torch.cat([self.u3(d), e3], dim=1))
        d  = self.d2(torch.cat([self.u2(d), e2], dim=1))
        d  = self.d1(torch.cat([self.u1(d), e1], dim=1))
        return self.out(d) + x

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
