import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
here, I will experiment!
"""

import dataclasses
@dataclasses.dataclass
class Config:
    network: str = "odenet"
    tol: float = 1e-3
    adjoint: bool = False
    downsampling_method: str = "conv"
    nepochs: int = 10
    data_aug: bool = True
    lr: float = 0.1
    batch_size: int = 138
    test_batch_size : int = 1000

    save: str = './experiment1'
    debug: bool = False
    gpu: int = 0
    data_path: str = "./data/mnist"

args = Config()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class Linear_CNF_HN_func(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width, yolo_dim):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width, yolo_dim)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width, yolo_dim):
        super().__init__()

        blocksize = width * in_out_dim * yolo_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3 * blocksize + width*yolo_dim)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize
        self.yolo_dim = yolo_dim

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = torch.tanh(self.fc3(params))
        params = self.fc4(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, self.yolo_dim)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, self.yolo_dim, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, self.yolo_dim, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, self.yolo_dim)
        return [W, B, U]

################################
import copy

def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU()
}

class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, strides, nonlinearity="softplus", num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        assert len(strides) == len(hidden_dims) + 1
        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = ConcatConv2d(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

def sample_gaussian_like(y):
    return torch.randn_like(y)

class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        self.T = T

        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.T]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def create_model(args, data_shape):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=data_shape,
            strides=strides,
            nonlinearity=args.nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            solver=args.solver,
        )
        return cnf
    model = build_cnf()
    return model


import torchvision.transforms as tforms
import torchvision.datasets as dset
def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x

def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape

def get_train_loader(train_set):
    current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    return train_loader

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim

import math
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

from torchvision.utils import save_image
import dataclasses
@dataclasses.dataclass
class Args:
    dims = "8,32,32,8"
    strides = "2,2,1,-2,-2"
    nonlinearity = "tanh"

    divergence_fn = "approximate"
    residual = True
    
    time_length = 1.0
    solver = "dopri5"

    data = "mnist"
    imagesize = None
    test_batch_size = 200
    batch_size = 200

    add_noise = True
    resume = None

    lr = 10**-3
    weight_decay = 0.0001

    begin_epoch = 1
    num_epochs = 100
    warmup_iters = 1000

    max_grad_norm = 1e10

    log_freq = 10

import time

if __name__ == "__main__":
    args = Args()
    # train_set, test_loader, data_shape = get_dataset(args)
    # model = create_model(args, data_shape)
    # print(model)
    # print(model(train_set[0][0][None,...]))

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    model = create_model(args, data_shape)


    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    steps_meter = RunningAverageMeter(0.97)
    grad_meter = RunningAverageMeter(0.97)
    tt_meter = RunningAverageMeter(0.97)

    best_loss = float("inf")
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set)
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            # cast data and move to device
            x = cvt(x)
            # compute loss
            loss = compute_bits_per_dim(x, model)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            grad_meter.update(grad_norm)

            if (itr) % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Grad Norm {:.4f}({:.4f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, grad_meter.val, grad_meter.avg
                    )
                )
                print(log_message)

            itr += 1

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                losses = []
                for (x, y) in test_loader:
                    if not args.conv:
                        x = x.view(x.shape[0], -1)
                    x = cvt(x)
                    loss = compute_bits_per_dim(x, model)
                    losses.append(loss)

                loss = np.mean(losses)
                if loss < best_loss:
                    best_loss = loss
                    makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)
