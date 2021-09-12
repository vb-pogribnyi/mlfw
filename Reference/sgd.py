import torch

class MyModel(torch.nn.Module):
    def __init__(self, ch_in, ch_out, win):
        super(MyModel, self).__init__()
        self.conv = torch.nn.Conv1d(ch_in, ch_out, win)

    def forward(self, x):
        return self.conv(x)


def run_test(ch_in, ch_out, win, lr, input, target, weight, bias):
    mse = torch.nn.MSELoss()
    model = MyModel(ch_in, ch_out, win)
    model.conv.weight.data = torch.tensor(weight).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = torch.tensor(bias).reshape(model.conv.bias.data.shape)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    net_input = torch.tensor(input).float().reshape(1, ch_in, -1)
    net_target = torch.tensor(target).float().reshape(1, ch_out, -1)

    opt.zero_grad()
    net_output = model(net_input)
    loss = mse(net_output, net_target)
    loss.backward()
    print('Grads:', model.conv.weight.grad.detach().numpy().reshape(-1), model.conv.bias.grad.detach().numpy().reshape(-1))
    print('Before:', model.conv.weight.detach().numpy().reshape(-1), model.conv.bias.detach().numpy().reshape(-1))
    opt.step()
    print('After:', model.conv.weight.detach().numpy().reshape(-1), model.conv.bias.detach().numpy().reshape(-1))

def sgd_test_1():
    run_test(
        ch_in=1,
        ch_out=1,
        win=1,
        lr=0.1,
        input=[1],
        target=[0.8],
        weight=[1.],
        bias=[0.]
    )


def sgd_test_2():
    run_test(
        ch_in=1,
        ch_out=2,
        win=3,
        lr=0.1,
        input=[1, 0.6, 0.9],
        target=[0.8, 0.3],
        weight=[1., 0.2, 0.1, 0.3, 0.4, 1.],
        bias=[0., 0.6]
    )


def sgd_test_3():
    run_test(
        ch_in=2,
        ch_out=1,
        win=3,
        lr=0.01,
        input=[1, 0.6, 0.9, 0.3, 0.2, 0.4],
        target=[0.3],
        weight=[1., 0.2, 0.1, 0.3, 0.4, 1.],
        bias=[0.6]
    )


# sgd_test_1()
# sgd_test_2()
sgd_test_3()
