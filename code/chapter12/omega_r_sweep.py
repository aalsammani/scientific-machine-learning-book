"""The flexible-network trap in joint state-parameter estimation (Chapter 12, section 12.5,
Theorem 12.2; Figure 12.1).

Joint estimation of the diffusivity nu (true value 0.5) and a network u_theta for the heat equation
of Running Example A on (0, 1) x (0, T] with T = 0.2, from 40 noisy observations
(x ~ U(0, 1), t ~ U(0.01, T), noise standard deviation 0.01; numpy default_rng(7)). Loss:
omega_r (L_residual + L_boundary) + L_data, with the residual weight omega_r swept over
{0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10}. Network 3 x 32 tanh (LeCun normal, seed 0), n_r = 600 residual
and n_b = 50 boundary points per side (torch.Generator(1000)); log nu initialized at log 0.2; Adam
(2000 iterations, learning rate 2e-3), float64, one thread. The recovered nu, the final data loss
and the final residual loss are recorded for each omega_r (the boundary loss is not stored).

Usage:
    python code/chapter12/omega_r_sweep.py
Output: results/chapter12/omega_r_sweep.json
Runtime: several minutes on one core. Stochastic but seeded.
"""
import numpy as np, torch, json, time
from pathlib import Path
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

NU_TRUE, T = 0.5, 0.2
def exact(x, t, nu=NU_TRUE):
    return np.exp(-nu*np.pi**2*t)*np.sin(np.pi*x) + 0.5*np.exp(-nu*25*np.pi**2*t)*np.sin(5*np.pi*x)

class MLP(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = [2,32,32,32,1]
        self.linears = torch.nn.ModuleList()
        for din,dout in zip(layers[:-1], layers[1:]):
            lin = torch.nn.Linear(din,dout)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(dout,din,generator=g)*(1.0/din)**0.5)
                lin.bias.zero_()
            self.linears.append(lin)
    def forward(self, x, t):
        h = torch.stack([x,t],-1)
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h).squeeze(-1)

def residual(model, x, t, nu):
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    u = model(x,t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut - nu*uxx

N_R, N_B = 600, 50
ADAM_ITERS = 2000

# sparse noisy observations of u used as the data-misfit term (inverse problem: recover nu from data)
rng = np.random.default_rng(7)
n_obs = 40
x_obs = rng.uniform(0,1,n_obs); t_obs = rng.uniform(0.01,T,n_obs)
sigma_obs = 0.01
y_obs = exact(x_obs, t_obs) + rng.normal(0, sigma_obs, n_obs)
x_obs_t, t_obs_t, y_obs_t = torch.tensor(x_obs), torch.tensor(t_obs), torch.tensor(y_obs)

def run(omega_r, seed=0):
    g = torch.Generator().manual_seed(seed+1000)
    xr = torch.rand(N_R, generator=g); tr = torch.rand(N_R, generator=g)*T
    tb = torch.rand(N_B, generator=g)*T
    model = MLP(seed)
    log_nu = torch.nn.Parameter(torch.log(torch.tensor(0.2)))  # deliberately wrong initial guess
    opt = torch.optim.Adam(list(model.parameters())+[log_nu], lr=2e-3)
    for it in range(ADAM_ITERS):
        opt.zero_grad()
        nu = torch.exp(log_nu)
        lr = (residual(model,xr,tr,nu)**2).mean()
        lb = (model(torch.zeros(N_B),tb)**2).mean() + (model(torch.ones(N_B),tb)**2).mean()
        ldata = ((model(x_obs_t,t_obs_t)-y_obs_t)**2).mean()
        loss = omega_r*(lr+lb) + ldata
        loss.backward(); opt.step()
    return float(torch.exp(log_nu).detach()), float(ldata.detach()), float(lr.detach())

omega_values = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
results = []
for om in omega_values:
    nu_hat, ldata, lres = run(om)
    results.append(dict(omega_r=om, nu_hat=nu_hat, data_loss=ldata, residual_loss=lres))
    print(f"omega_r={om}: nu_hat={nu_hat:.4f} (true {NU_TRUE}), data_loss={ldata:.2e}, residual_loss={lres:.2e}")

json.dump(dict(nu_true=NU_TRUE, sigma_obs=sigma_obs, n_obs=n_obs, results=results),
          open(Path(__file__).resolve().parents[2] / "results" / "chapter12" / "omega_r_sweep.json", "w"), indent=1)
