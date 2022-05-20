import torch.nn as nn
import torch
import torch.optim as optim
import time
import numpy as np
import os
from torchvision.utils import save_image

class Conv2dConcat(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, residual=True, nonlinearity="relu"):
        super().__init__()
        self.residual = residual and in_dim == out_dim
        self._layer = nn.Conv2d(in_dim+1, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.nonlinearity = NONLINEARITIES[nonlinearity]
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
    "elu": nn.ELU(),
    "none": nn.Identity(),
    None: nn.Identity()
}

class ODENet(nn.Module):
    def __init__(self, hidden_dims, input_shape, nonlinearity="relu"):
        super().__init__()
        layers = []
        for l, (in_dim, out_dim) in enumerate(zip([input_shape[0]] + hidden_dims, hidden_dims + [input_shape[0]])):
            _non_lin = nonlinearity if l < len(hidden_dims) else None
            layers.append(Conv2dConcat(in_dim, out_dim, nonlinearity=_non_lin))
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
        self.num_calls = 0
    def forward(self, t, states):
        self.num_calls += 1
        z, log_pz = states if isinstance(states, tuple) else (states, None)
        t = torch.tensor(t).type_as(z)
        batchsize = z.shape[0]
        if log_pz is not None:
            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                t.requires_grad_(True)
                dx = self.ode_net(t, z)
                
                # if not self.training:
                #     divergence = divergence_bf(dx, z).view(batchsize, 1)
                # else:
                divergence = self.divergence_fn(dx, z).view(batchsize, 1)
            return tuple([dx, -divergence])
        else:
            dx = self.ode_net(t, z)
            return dx
    def before_int(self):
        self.num_calls = 0

        
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

import math
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


class CNF(nn.Module):
    def __init__(self, ode_func:ODEFunc, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5, aug_dim=0):
        super().__init__()
        self.ode_func = ode_func
        self.T = T
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_atol = atol
        self.test_rtol = rtol
        self.aug_dim = aug_dim
    def forward(self, y, get_log_px=True, integration_times=None, reverse=False):
        # reverse=False: x=z_0 to z_T

        if get_log_px:
            _log_pz = torch.zeros(y.shape[0], 1).to(y)

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.T]).to(y)
        if reverse:
            integration_times = _flip(integration_times, 0)
        
        # augmented NODE
        y_shape = list(y.shape)
        y_shape[1] = self.aug_dim
        tt = torch.zeros(y_shape).to(y)
        y_aug = torch.cat([tt, y], 1)
        
        self.ode_func.before_int()

        z_t = odeint(
            self.ode_func,
            (y_aug, _log_pz) if get_log_px else y_aug,
            integration_times.to(y_aug),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )
        if get_log_px:
            z_t, dlog_pz = z_t
        
        # delete augmentation
        if self.aug_dim != 0:
            z_t = z_t[:, :, self.aug_dim:]

        if get_log_px:
            z = z_t[-1] if not reverse else z_t[0]
            log_pz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
            log_px = log_pz - dlog_pz if not reverse else log_pz + dlog_pz

        if len(integration_times) == 2:  # only return end
            z_t = z_t[1]
            if get_log_px:
                log_px = log_px[1]
        
        if get_log_px:
            return z_t, log_px
        else:
            return z_t



def create_model(args, data_shape):
    aug_data_shape = list(data_shape)
    aug_data_shape[0] += args.aug_dim

    def build_cnf():
        ode_net = ODENet(
            hidden_dims=args.hidden_dims,
            input_shape=aug_data_shape,
            nonlinearity=args.nonlinearity,
        )
        ode_func = ODEFunc(
            ode_net=ode_net,
            approximate_trace=args.approximate_trace,
        )
        cnf = CNF(
            ode_func=ode_func,
            T=args.time_length,
            solver=args.solver,
            aug_dim=args.aug_dim,
            atol=args.tol,
            rtol=args.tol
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

def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def compute_bits_per_dim(x, model: CNF):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, log_px = model(x, get_log_px=True)  # run model forward

    log_px_per_dim = torch.sum(log_px) / x.nelement()  # averaged over batches
    bits_per_dim = -(log_px_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim

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

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

import dataclasses
@dataclasses.dataclass
class Args:
    hidden_dims = []
    # hidden_dims = [4]
    nonlinearity = "tanh"
    aug_dim = 0

    approximate_trace = True
    residual = True
    
    time_length = 1.0
    solver = "dopri5"
    tol = 1e-2

    data = "mnist"
    imagesize = None
    test_batch_size = 128
    batch_size = 128

    add_noise = True
    resume = None

    lr = 10**-3
    weight_decay = 0.0001

    begin_epoch = 1
    num_epochs = 100
    warmup_iters = 1000

    max_grad_norm = 1e10

    log_freq = 1
    val_freq = 1
    save = "experiment2"

    adjoint = True

    data_parallel = False

if __name__ == "__main__":

    args = Args()
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

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
    print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    # visualize samples
    fixed_z = cvt(torch.randn(100, *data_shape))

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    grad_meter = RunningAverageMeter(0.97)
    calls_meter = RunningAverageMeter(0.97)

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
            calls_meter.update(model.ode_func.num_calls)

            if (itr) % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Grad Norm {:.4f}({:.4f}) | Calls {:.4f}({:.4f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, grad_meter.val, grad_meter.avg, calls_meter.val, calls_meter.avg
                    )
                )
                print(log_message)

            itr += 1
            break

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            print("validating")
            with torch.no_grad():
                start = time.time()
                losses = []
                for (x, y) in test_loader:
                    x = cvt(x)
                    loss = compute_bits_per_dim(x, model)
                    losses.append(loss.item())

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
        print("sample")
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, get_log_px=False, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)
