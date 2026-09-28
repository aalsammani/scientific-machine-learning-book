"""One-dimensional Fourier neural operator for the linear heat semigroup on a periodic domain
(Chapter 9, sections 9.3-9.5).

Operator: u0 -> u(., T) for u_t = nu u_xx on [0, 2 pi) with periodic boundary conditions,
nu = 0.05, T = 1. The reference is exact (each Fourier mode k decays by exp(-nu k^2 T)), so the
measured error contains no reference discretization error.

Input distribution for training: u0 = sum_{m=1..8} Re(c_m e^{i m x}) with c_m = (a_m + i b_m)/m,
a_m, b_m ~ N(0, 1) i.i.d. (in the code the decay factor is written 1/(k+1) with the zero-based index
k = m - 1, so mode m is scaled by 1/m); mode 0 is zero.

Architecture: pointwise lifting to width channels, two Fourier layers (spectral convolution keeping
the lowest `modes` rfft coefficients, plus a pointwise linear path, tanh), a pointwise tanh
projection layer and a pointwise output layer; no coordinate input channel; float64. The spectral
weights are drawn from seeded torch.Generator objects (seed + 1, seed + 2); the pointwise
(Conv1d) layers keep PyTorch's default initialization, which draws from the global random stream.
Because the scripts never call torch.manual_seed, those layers depend on how many models were
created earlier in the same process: rerunning a seed in a different order or process gives a
statistically equivalent but not identical model (see code/chapter09/README.md).
"""
import torch, numpy as np
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

NU, T = 0.05, 1.0
MAX_MODE_TRAIN = 8  # training input distribution: random coefficients on modes 1..8

def sample_u0_coeffs(n_samples, max_mode, rng, amp=1.0):
    # complex Fourier coefficients for a real periodic signal, modes 1..max_mode (mode 0 fixed to 0)
    c = np.zeros((n_samples, max_mode), dtype=np.complex128)
    re = rng.normal(0, amp, (n_samples, max_mode))
    im = rng.normal(0, amp, (n_samples, max_mode))
    c = re + 1j*im
    # decay envelope: index k = 0..max_mode-1 holds mode m = k + 1, which is scaled by 1/m
    decay = np.array([1.0/ (k+1)**1.0 for k in range(max_mode)])
    c = c * decay[None,:]
    return c  # shape (n_samples, max_mode), modes 1..max_mode

def eval_on_grid(coeffs, n_grid):
    # build u0 on a grid of size n_grid from coefficients for modes 1..max_mode, periodic on [0,2pi)
    x = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    max_mode = coeffs.shape[-1]
    u0 = np.zeros((coeffs.shape[0], n_grid))
    for k in range(1, max_mode+1):
        ck = coeffs[:, k-1]
        u0 += (ck.real[:,None]*np.cos(k*x)[None,:] - ck.imag[:,None]*np.sin(k*x)[None,:])
    return u0, x

def exact_propagate(coeffs, n_grid, T, nu=NU):
    x = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    max_mode = coeffs.shape[-1]
    u = np.zeros((coeffs.shape[0], n_grid))
    for k in range(1, max_mode+1):
        ck = coeffs[:, k-1] * np.exp(-nu*k**2*T)
        u += (ck.real[:,None]*np.cos(k*x)[None,:] - ck.imag[:,None]*np.sin(k*x)[None,:])
    return u

class SpectralConv1d(torch.nn.Module):
    def __init__(self, in_ch, out_ch, modes, seed):
        super().__init__()
        self.modes = modes
        g = torch.Generator().manual_seed(seed)
        scale = 1.0/(in_ch*out_ch)
        self.weight_real = torch.nn.Parameter(scale*torch.randn(in_ch, out_ch, modes, generator=g))
        self.weight_imag = torch.nn.Parameter(scale*torch.randn(in_ch, out_ch, modes, generator=g))

    def forward(self, x):
        # x: (batch, in_ch, n_grid)
        n_grid = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_ch, n_grid//2+1)
        out_ft = torch.zeros(x.shape[0], self.weight_real.shape[1], x_ft.shape[-1], dtype=torch.complex128)
        m = min(self.modes, x_ft.shape[-1])
        w = torch.complex(self.weight_real[:, :, :m], self.weight_imag[:, :, :m])
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], w)
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)

class FNO1d(torch.nn.Module):
    def __init__(self, modes, width, seed):
        super().__init__()
        self.lift = torch.nn.Conv1d(1, width, 1)
        self.spec1 = SpectralConv1d(width, width, modes, seed+1)
        self.lin1 = torch.nn.Conv1d(width, width, 1)
        self.spec2 = SpectralConv1d(width, width, modes, seed+2)
        self.lin2 = torch.nn.Conv1d(width, width, 1)
        self.proj1 = torch.nn.Conv1d(width, width, 1)
        self.proj2 = torch.nn.Conv1d(width, 1, 1)
        # The Conv1d layers keep PyTorch's default initialization (global random stream, float64).

    def forward(self, u0):
        # u0: (batch, n_grid)
        x = u0.unsqueeze(1)  # (batch,1,n_grid)
        x = self.lift(x)
        x = torch.tanh(self.spec1(x) + self.lin1(x))
        x = torch.tanh(self.spec2(x) + self.lin2(x))
        x = torch.tanh(self.proj1(x))
        x = self.proj2(x)
        return x.squeeze(1)

