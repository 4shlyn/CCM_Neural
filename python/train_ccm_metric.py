# train_ccm_metric.py
# Neural CCM training for cart-pendulum with sin/cos embedding:
#   x̃ = [x, xdot, s, c, thetadot]
#
# Enforces contraction on closed-loop differential dynamics:
#   F = sym(dotM + (A + B K)^T M + M (A + B K) + 2 lam M)  <  0
#

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

class Params:
    def __init__(self):
        self.M = 0.6
        self.m = 0.2
        self.l = 0.3
        self.I = 0.006
        self.b = 0.05
        self.c = 0.002
        self.g = 9.81
p = Params()

# ----------------------------
# Dynamics in embedded coords
# ----------------------------
def f_and_B(x, p: Params):
    # x: (batch,5) = [X, dX, s, c, dth]
    X, dX, s, c, dth = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]  # noqa
    M, m, l, I, g, b, cdamp = p.M, p.m, p.l, p.I, p.g, p.b, p.c

    D = (M+m)*(I+m*l*l) - (m*l*c)**2
    a1 = (I + m*l*l) / D
    b1 = -(m*l*c) / D

    ddX0 = ((I+m*l*l)*(-b*dX + m*l*s*dth*dth) + (m*l*c)*(m*g*l*s - cdamp*dth)) / D
    ddth0 = ((m*l*c)*(b*dX - m*l*s*dth*dth) - (M+m)*(m*g*l*s - cdamp*dth)) / D

    ds = c*dth
    dc = -s*dth

    f = torch.stack([dX, ddX0, ds, dc, ddth0], dim=1)

    B = torch.zeros(x.shape[0], 5, 1, device=x.device, dtype=x.dtype)
    B[:,1,0] = a1
    B[:,4,0] = b1
    return f, B

def jacobian_f(x, p: Params):
    x = x.clone().requires_grad_(True)
    f, _ = f_and_B(x, p)
    batch = x.shape[0]
    A = torch.zeros(batch, 5, 5, device=x.device, dtype=x.dtype)
    for j in range(5):
        g = torch.autograd.grad(f[:, j].sum(), x, create_graph=True, retain_graph=True)[0]
        A[:, j, :] = g
    return A.transpose(1,2)

def dMdt(x, M, xdot):
    batch = x.shape[0]
    n = M.shape[1]
    dotM = torch.zeros(batch, n, n, device=x.device, dtype=x.dtype)
    for j in range(n):
        for k in range(n):
            g = torch.autograd.grad(M[:, j, k].sum(), x, create_graph=True, retain_graph=True)[0]
            dotM[:, j, k] = (g * xdot).sum(dim=1)
    return dotM

def sym(S):
    return 0.5*(S + S.transpose(1,2))

def max_eig_sym(S):
    return torch.linalg.eigvalsh(S)[:, -1]

# Metric and gain networks
def vec_to_lower_tri_5(v):
    batch = v.shape[0]
    L = torch.zeros(batch, 5, 5, device=v.device, dtype=v.dtype)
    idx = 0
    for r in range(5):
        for c in range(r+1):
            L[:, r, c] = v[:, idx]; idx += 1
    return L

class MetricNet(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 15)
        )
    def forward(self, x):
        return vec_to_lower_tri_5(self.net(x))

class GainNet(nn.Module):
    # outputs k(x): (batch,1,5)
    def __init__(self, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 5)
        )
    def forward(self, x):
        k = self.net(x)
        return k.unsqueeze(1)

def metric_from_L(L, eps):
    batch = L.shape[0]
    I = torch.eye(5, device=L.device, dtype=L.dtype).unsqueeze(0).repeat(batch,1,1)
    return L.transpose(1,2) @ L + eps*I


# sampling.
x_mean = torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0], device=DEVICE)
x_std  = torch.tensor([1.0, 2.0, 0.7, 0.7, 6.0], device=DEVICE)

def normalize(x): return (x - x_mean) / x_std

def sample_batch(N):
    X  = torch.empty(N, device=DEVICE).uniform_(-0.8, 0.8)
    dX = torch.empty(N, device=DEVICE).uniform_(-2.0, 2.0)
    th = math.pi + torch.empty(N, device=DEVICE).uniform_(-0.7, 0.7)
    dth= torch.empty(N, device=DEVICE).uniform_(-6.0, 6.0)
    s  = torch.sin(th)
    c  = torch.cos(th)
    return torch.stack([X, dX, s, c, dth], dim=1)

# training.
metric_net = MetricNet().to(DEVICE)
gain_net   = GainNet().to(DEVICE)

eps = 5e-4
lam = 2.0

opt = optim.Adam(list(metric_net.parameters()) + list(gain_net.parameters()), lr=2e-3)

steps = 12000
batch_size = 256

for step in range(steps):
    x = sample_batch(batch_size).requires_grad_(True)
    xn = normalize(x)

    L = metric_net(xn)
    M = metric_from_L(L, eps=eps)

    f, B = f_and_B(x, p)
    A = jacobian_f(x, p)

    # Learned differential feedback gain
    K = gain_net(xn) # (batch,1,5)
    Acl = A + (B @ K) # (batch,5,5)

    # dotM along closed-loop flow
    u_local = (K @ (x - x_mean).unsqueeze(-1)).squeeze(-1).squeeze(1)  # scalar per batch (rough)
    xdot = f + B.squeeze(-1) * u_local.unsqueeze(-1)
    dotM = dMdt(x, M, xdot)

    F = sym(dotM + Acl.transpose(1,2) @ M + M @ Acl + 2.0*lam*M)

    viol = torch.relu(max_eig_sym(F))
    loss_ccm = (viol**2).mean()

    # regularize to avoid insane gains
    loss_gain = 1e-4 * (K**2).mean()
    loss_reg  = 1e-4 * (M**2).mean()

    loss = loss_ccm + loss_gain + loss_reg

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 400 == 0:
        print(f"step {step:5d} | loss={loss.item():.3e} | viol_mean={viol.mean().item():.3e} | viol_max={viol.max().item():.3e}")

# export to matlab for sim + control loop
def to_np(t): return t.detach().cpu().numpy()

def export_linear_layers(seq):
    layers = [m for m in seq if isinstance(m, nn.Linear)]
    out = {}
    for i, layer in enumerate(layers):
        out[f"W{i+1}"] = to_np(layer.weight)
        out[f"b{i+1}"] = to_np(layer.bias)
    return out

state = {}

# metric net weights
metric_layers = export_linear_layers(metric_net.net)
for k,v in metric_layers.items():
    state["M_"+k] = v

# gain net weights
gain_layers = export_linear_layers(gain_net.net)
for k,v in gain_layers.items():
    state["K_"+k] = v

state["x_mean"] = to_np(x_mean)
state["x_std"]  = to_np(x_std)
state["eps"]    = np.array([eps])
state["lam"]    = np.array([lam])

savemat("metric_net.mat", state)
print("Saved metric_net.mat (with M_*, K_* weights)")
