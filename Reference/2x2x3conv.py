import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(2, 2, 3)

    def forward(self, x):
        return self.conv(x)


def test1():
    print('Test 1')
    model = Model()
    model.conv.weight.data = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape([1, 2, 5])
    out = model(data.float())
    print(out.shape, out)
    print('Output:', out.reshape([-1]).detach().numpy())


if __name__ == '__main__':
    test1()