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


def testForward2():
    runTestForward(
        1, 1, 3, 3,
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1],
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((1, 1, 3, 3))
    )


def testForward3():
    runTestForward(
        1, 1, 3, 3,
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1],
        np.array([
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1
        ]).reshape((1, 1, 4, 4))
    )


def testForward4():
    runTestForward(
        1, 1, 3, 3,
        [0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1],
        [1],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2
        ]).reshape((1, 1, 4, 4))
    )


def testForward5():
    runTestForward(
        1, 2, 3, 3,
        [
            0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1,
            0.8, 0.6, 0.2, 0.6, 1.2, 0.1, 0.9, 0.6, 0.4
        ],
        [0.2, 0.5],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2
        ]).reshape((1, 1, 4, 4))
    )


def testForward6():
    runTestForward(
        2, 1, 3, 3,
        [
            0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1,
            0.8, 0.6, 0.2, 0.6, 1.2, 0.1, 0.9, 0.6, 0.4
        ],
        [0.2],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2,

            1.8, 0.2, 0.4, 0.1,
            0.9, 0.2, 0.4, 0.5,
            0.1, 0.5, 0.3, 0.3,
            1.2, 0.8, 0.9, 0.1
        ]).reshape((1, 2, 4, 4))
    )


def testForward7():
    runTestForward(
        2, 2, 3, 3,
        [
            0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1,
            0.8, 0.6, 0.2, 0.6, 1.2, 0.1, 0.9, 0.6, 0.4,
            0.6, 0.3, 0.2, 0.6, 1.0, 0.8, 0.2, 0.4, 0.1,
            0.4, 0.9, 0.2, 0.6, 1.8, 0.4, 0.9, 0.6, 0.5
        ],
        [0.2, 0.5],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2,

            1.8, 0.2, 0.4, 0.1,
            0.9, 0.2, 0.4, 0.5,
            0.1, 0.5, 0.3, 0.3,
            1.2, 0.8, 0.9, 0.1
        ]).reshape((1, 2, 4, 4))
    )


def testForward8():
    runTestForward(
        2, 3, 3, 3,
        [
            0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1,
            0.8, 0.6, 0.2, 0.6, 1.2, 0.1, 0.9, 0.6, 0.4,
            0.6, 0.3, 0.2, 0.6, 1.0, 0.8, 0.2, 0.4, 0.1,
            0.4, 0.9, 0.2, 0.6, 1.8, 0.4, 0.9, 0.6, 0.5,
            0.1, 0.3, 0.5, 0.6, 1.7, 0.8, 0.1, 0.4, 0.3,
            0.6, 0.2, 0.5, 0.7, 0.3, 0.4, 0.3, 0.6, 0.5
        ],
        [0.2, 0.5, 0.3],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2,

            1.8, 0.2, 0.4, 0.1,
            0.9, 0.2, 0.4, 0.5,
            0.1, 0.5, 0.3, 0.3,
            1.2, 0.8, 0.9, 0.1
        ]).reshape((1, 2, 4, 4))
    )


def testForward9():
    runTestForward(
        2, 3, 3, 3,
        [
            0.1, 0.3, 0.2, 0.6, 1.1, 0.8, 0.5, 0.4, 0.1,
            0.8, 0.6, 0.2, 0.6, 1.2, 0.1, 0.9, 0.6, 0.4,
            0.6, 0.3, 0.2, 0.6, 1.0, 0.8, 0.2, 0.4, 0.1,
            0.4, 0.9, 0.2, 0.6, 1.8, 0.4, 0.9, 0.6, 0.5,
            0.1, 0.3, 0.5, 0.6, 1.7, 0.8, 0.1, 0.4, 0.3,
            0.6, 0.2, 0.5, 0.7, 0.3, 0.4, 0.3, 0.6, 0.5
        ],
        [0.2, 0.5, 0.3],
        np.array([
            1.6, 0.3, 0.5, 0.8,
            0.2, 0.2, 0.1, 0.5,
            0.1, 0.2, 1.3, 0.5,
            1.1, 0.5, 0.3, 0.2,

            1.8, 0.2, 0.4, 0.1,
            0.9, 0.2, 0.4, 0.5,
            0.1, 0.5, 0.3, 0.3,
            1.2, 0.8, 0.9, 0.1,

            0.2, 1.3, 0.5, 0.8,
            0.2, 0.2, 0.8, 0.5,
            0.1, 0.4, 1.2, 0.8,
            1.1, 0.5, 0.3, 0.1,

            1.8, 0.2, 0.4, 0.1,
            0.9, 0.3, 0.6, 0.5,
            0.3, 0.1, 0.9, 0.3,
            1.2, 0.1, 0.2, 0.7
        ]).reshape((2, 2, 4, 4))
    )


if __name__ == '__main__':
    # testForward1()
    # testForward2()
    # testForward3()
    # testForward4()
    # testForward5()
    # testForward6()
    # testForward7()
    # testForward8()
    testForward9()
