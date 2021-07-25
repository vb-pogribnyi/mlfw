import torch


class Model(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(ch_in, ch_out, 3)

    def forward(self, x):
        return self.conv(x)


def test1():
    print('Test 1')
    model = Model(2, 2)
    model.conv.weight.data = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape([1, 2, 5])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


def test2():
    print('Test 2')
    model = Model(1, 1)
    model.conv.weight.data = torch.tensor([0.1, 0.2, 0.3]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 2, 3, 4, 5, 6]).reshape([2, 1, 3])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


def test3():
    print('Test 3')
    model = Model(2, 1)
    model.conv.weight.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape([2, 2, 3])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


def test4():
    print('Test 4')
    model = Model(1, 2)
    model.conv.weight.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 2, 3, 4, 5, 6]).reshape([2, 1, 3])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


def testN():
    print('Test N')
    model = Model(3, 2)
    model.conv.weight.data = torch.tensor([-0.1, 0.2, -0.1, -0.2, 0.4, -0.2, -0.3, 0.6, -0.3, -0.4, 0.8, -0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]).reshape([2, 3, 5])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    test4()
    # testN()