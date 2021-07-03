import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


def test1():
    print('Test 1')
    model = Model()
    model.conv.weight.data = model.conv.weight.data * 0 + 1
    model.conv.bias.data = model.conv.bias.data * 0
    data = torch.tensor([1]).reshape([1, 1, -1])
    out = model(data.float())
    print('Output:', out.reshape([-1]).detach().numpy())


if __name__ == '__main__':
    test1()