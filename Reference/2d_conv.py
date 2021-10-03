import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, ch_in, ch_out, window_w=3, window_h=3):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(ch_in, ch_out, (window_w, window_h))

    def forward(self, x):
        return self.conv(x)


def runTestForward(ch_in, ch_out, window_w, window_h, weight, bias, input):
    model = Model(ch_in, ch_out, window_w, window_h)
    model.conv.weight.data = torch.tensor(weight).reshape(model.conv.weight.data.shape).float()
    model.conv.bias.data = torch.tensor(bias).reshape(model.conv.bias.data.shape).float()
    model_input = torch.tensor(input).float()
    out = model(model_input)
    print(out.reshape(-1).detach().numpy())


def testForward1():
    runTestForward(
        1, 1, 1, 1,
        [1],
        [1],
        np.array([[[[1]]]])
    )


if __name__ == '__main__':
    testForward1()
