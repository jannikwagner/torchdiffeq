import numpy as np
import torch

def test_ode_net():
    from cnf_mnist2 import ODENet
    input_shape = (3,28,28)
    hidden_dims = [4,8,8,4] 
    ode_net = ODENet(hidden_dims, input_shape)
    print(ode_net)

    x = torch.as_tensor(np.random.rand(2, *input_shape), dtype=torch.float32)
    print(x.shape)
    y = ode_net(0,x)
    print(y.shape)
    # print(y)

def test_ode_func():
    print("test_ode_func")
    from cnf_mnist2 import ODENet, ODEFunc
    input_shape = (3,28,28)
    hidden_dims = [4,8,8,4] 
    ode_net = ODENet(hidden_dims, input_shape)
    ode_func = ODEFunc(ode_net, False)
    print(ode_func)

    x = torch.as_tensor(np.random.rand(2, *input_shape), dtype=torch.float32)
    log_px = None

    print(x.shape)
    dx, dlog_px = ode_func(0,(x, log_px))
    print(dx.shape)
    print(dlog_px)
    # print(y)
def test_cnf():
    print("test_ode_func")
    from cnf_mnist2 import ODENet, ODEFunc, CNF
    input_shape = (3,4,4)
    hidden_dims = [4,4] 
    ode_net = ODENet(hidden_dims, input_shape)
    ode_func = ODEFunc(ode_net, True)
    cnf = CNF(ode_func, atol=0.001, rtol=0.001)
    print(cnf)

    x = torch.as_tensor(np.random.rand(2, *input_shape), dtype=torch.float32)

    print(x.shape)
    with torch.no_grad():
        pass
    z, log_px = cnf(x, True)
    print(z.shape)
    x_2, log_px_2 = cnf(z, True, reverse=True)
    print(x_2.shape)
    z_2 = cnf(x, False)
    print(z_2.shape)
    x_3 = cnf(z, False, reverse=True)
    print(x_3.shape)
    print(log_px, log_px_2)
    # print(x-x_2)
    # print(x-x_3)
    # print(z-z_2)
    print(x_2-x_3)
    # print(y)

if __name__ == "__main__":
    test_ode_net()
    test_ode_func()
    test_cnf()