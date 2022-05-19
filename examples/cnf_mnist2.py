from curses import nonl
from turtle import forward
import torch.nn as nn
import torch

class Conv2dConcat(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, residual=True, nonlinearity="relu"):
        super().__init__()
        self.residual = residual and in_dim == out_dim
        self.nonlinearity = NONLINEARITIES[nonlinearity]
        self._layer = nn.Conv2d(in_dim+1, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, t, x):
        print(x.shape)
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        print(ttx.shape)
        y = self._layer(ttx)
        print(y.shape)
        if self.residual:
            y = y + x
        y = self.nonlinearity(y)
        print(y)
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

class ODEFunc(nn.Module):
    def __init__(self, ode_net, approximate_trace=True):
        super().__init__()
        self.ode_net = ode_net
        self.approximate_trace = approximate_trace
    def forward(self, t, states):
        x = states[0]
        log_px = states[1]
        
