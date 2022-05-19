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

if __name__ == "__main__":
    test_ode_net()