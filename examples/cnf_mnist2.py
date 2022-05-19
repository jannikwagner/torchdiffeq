import torch.nn as nn
import torch

from torchdiffeq import odeint_adjoint as odeint

class Conv2dConcat(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, residual=True, nonlinearity="relu"):
        super().__init__()
        self.residual = residual and in_dim == out_dim
        self.nonlinearity = NONLINEARITIES[nonlinearity]
        self._layer = nn.Conv2d(in_dim+1, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, t, x):
        # print(x.shape)
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        # print(ttx.shape)
        y = self._layer(ttx)
        # print(y.shape)
        if self.residual:
            y = y + x
        y = self.nonlinearity(y)
        # print(y)
        return y

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU()
}

class ODENet(nn.Module):
    def __init__(self, hidden_dims, input_shape, nonlinearity="relu"):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_shape[0]] + hidden_dims, hidden_dims + [input_shape[0]]):
            layers.append(Conv2dConcat(in_dim, out_dim, nonlinearity=nonlinearity))
        self.layers = nn.ModuleList(layers)
    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x

class StackedODENet(nn.Module):
    def __init__(self, n_stack, hidden_dims, input_shape, nonlinearity="relu", residual=True):
        super().__init__()
        self.residual = True
        layers = []
        for _ in range(n_stack):
            layers.append(ODENet(hidden_dims, input_shape, nonlinearity))
        self.layers = nn.ModuleList(layers)
    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x

def divergence_bf(f, z):
    f_flat = f.view(f.shape[0], -1)
    sum_diag = 0.
    for i in range(f_flat.shape[1]):
        grad = torch.autograd.grad(f_flat[:, i].sum(), z, create_graph=True)[0]
        sum_diag += grad.view(grad.shape[0], -1).contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, z):
    e = sample_gaussian_like(z)
    e_dfdz = torch.autograd.grad(f, z, e, create_graph=True)[0]
    e_dfdz_e = e_dfdz * e
    approx_tr_dzdx = e_dfdz_e.view(z.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

def sample_gaussian_like(y):
    return torch.randn_like(y)

class ODEFunc(nn.Module):
    def __init__(self, ode_net, approximate_trace=True):
        super().__init__()
        self.ode_net = ode_net
        self.divergence_fn = divergence_approx if approximate_trace else divergence_bf
    def forward(self, t, states):
        z, log_pz = states if isinstance(states, tuple) else (states, None)
        t = torch.tensor(t).type_as(z)
        batchsize = z.shape[0]
        if log_pz is not None:
            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                t.requires_grad_(True)
                dx = self.ode_net(t, z)
                # Hack for 2D data to use brute force divergence computation.
                if not self.training:
                    divergence = divergence_bf(dx, z).view(batchsize, 1)
                else:
                    divergence = self.divergence_fn(dx, z).view(batchsize, 1)
            return tuple([dx, -divergence])
        else:
            dx = self.ode_net(t, z)
            return dx

        
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

import math
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


class CNF(nn.Module):
    def __init__(self, ode_func, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.ode_func = ode_func
        self.T = T
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_atol = atol
        self.test_rtol = rtol
    def forward(self, y, get_log_px=True, integration_times=None, reverse=False):
        # reverse=False: x=z_0 to z_T

        if get_log_px:
            _log_pz = torch.zeros(y.shape[0], 1).to(y)

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.T]).to(y)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Add regularization states.

        z_t = odeint(
            self.ode_func,
            (y, _log_pz) if get_log_px else y,
            integration_times.to(y),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )
        if get_log_px:
            z_t, dlog_pz = z_t
            z = z_t[-1] if not reverse else z_t[0]
            log_pz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
            log_px = log_pz - dlog_pz if not reverse else log_pz + dlog_pz

        if len(integration_times) == 2:
            z_t = z_t[1]
            if get_log_px:
                log_px = log_px[1]
        
        if get_log_px:
            return z_t, log_px
        else:
            return z_t
