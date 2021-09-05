import torch


class Model(torch.nn.Module):
    def __init__(self, ch_in, ch_out, window=3):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(ch_in, ch_out, window)

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


def testGrad1():
    print('Test grad 1')
    model = Model(1, 1, 1)
    model.conv.weight.data = torch.tensor([0.2]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1]).reshape([1, 1, 1])
    out = model(data.float())
    target = torch.tensor([0.7]).reshape(1, 1, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad2():
    print('Test grad 2')
    model = Model(1, 2, 1)
    model.conv.weight.data = torch.tensor([-0.2, 0.]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1]).reshape([1, 1, 1])
    out = model(data.float())
    target = torch.tensor([1.2, 1.2]).reshape(1, 2, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad3():
    print('Test grad 3')
    model = Model(1, 1, 1)
    model.conv.weight.data = torch.tensor([0.2]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 0.6]).reshape([2, 1, 1])
    out = model(data.float())
    target = torch.tensor([0.7, 0.62]).reshape(2, 1, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad4():
    print('Test grad 4')
    model = Model(2, 1, 1)
    model.conv.weight.data = torch.tensor([0.2, 0.1]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([1, 0.6]).reshape([1, 2, 1])
    out = model(data.float())
    target = torch.tensor([0.7]).reshape(1, 1, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad5():
    print('Test grad 5')
    model = Model(1, 2, 1)
    model.conv.weight.data = torch.tensor([0.2, 0.1]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([0.6]).reshape([1, 1, 1])
    out = model(data.float())
    target = torch.tensor([0.7, 0.2]).reshape(1, 2, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad6():
    print('Test grad 6')
    model = Model(2, 2, 1)
    model.conv.weight.data = torch.tensor([0.2, 0.1, 0.3, 0.4]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([0.6, 0.2]).reshape([1, 2, 1])
    out = model(data.float())
    target = torch.tensor([0.7, 0.5]).reshape(1, 2, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad7():
    print('Test grad 7')
    model = Model(1, 1, 3)
    model.conv.weight.data = torch.tensor([0.2, 0.1, 0.3]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([0.6, 0.2, 0.8]).reshape([1, 1, 3])
    out = model(data.float())
    target = torch.tensor([0.7]).reshape(1, 1, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad8():
    print('Test grad 8')
    model = Model(2, 2, 3)
    model.conv.weight.data = torch.tensor([0.2, 0.1, 0.3, 0.6, 0.4, 0.8, 0.7, 0.5, 0.4, 0.2, 0.9, 0.8]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([0.6, 0.2, 0.8, 0.3, 0.5, 0.4]).reshape([1, 2, 3])
    out = model(data.float())
    target = torch.tensor([0.7, 0.4]).reshape(1, 2, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad9():
    print('Test grad 9')
    model = Model(2, 2, 3)
    model.conv.weight.data = torch.tensor([0.2, 0.1, 0.3, 0.6, 0.4, 0.8, 0.7, 0.5, 0.4, 0.2, 0.9, 0.8]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([
        0.6, 0.2, 0.8, 0.3, 0.5, 0.4,
        0.2, 0.4, 0.8, 0.4, 0.2, 0.7
    ]).reshape([2, 2, 3])
    out = model(data.float())
    target = torch.tensor([0.7, 0.4, 0.6, 0.8]).reshape(2, 2, 1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    print('Output:', out.reshape([-1]).detach().numpy())


def testGrad10():
    print('Test grad 10')
    model = Model(2, 2, 3)
    model.conv.weight.data = torch.tensor([0.2, 0.1, 0.3, 0.6, 0.4, 0.8, 0.7, 0.5, 0.4, 0.2, 0.9, 0.8]).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + 1
    data = torch.tensor([
        0.6, 0.5, 0.2, 0.1, 0.8, 0.3, 0.5, 0.4,
        0.2, 0.4, 0.8, 0.6, 0.4, 0.3, 0.2, 0.7
    ]).reshape([2, 2, 4])
    out = model(data.float())
    target = torch.tensor([0.7, 0.5, 0.4, 0.8, 0.6, 0.3, 0.8, 0.1]).reshape(2, 2, 2)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model.conv.weight.grad, model.conv.bias.grad)
    for v in model.conv.weight.grad.reshape(-1):
        print(v.item())
    print('W grad', model.conv.weight.grad.reshape(-1))
    print('Output:', out.reshape([-1]).detach().numpy())
    print('Loss back:', 2 * (out - target).reshape([-1]).detach().numpy())


def runTestSens(test_idx, ch_in, ch_out, window, weight, bias, input, output, n_examples):
    print('Test sensitivity {}'.format(test_idx))
    model = Model(ch_in, ch_out, window)
    model_input = torch.nn.Linear(len(input), len(input))
    model_input.weight.data = torch.eye(len(input)).float() * torch.tensor(input).unsqueeze(-1).repeat(1, len(input))
    model_input.bias.data = model_input.bias.data * 0

    model.conv.weight.data = torch.tensor(weight).reshape(model.conv.weight.data.shape)
    model.conv.bias.data = model.conv.bias.data * 0 + torch.tensor(bias).reshape(model.conv.bias.data.shape)
    data = torch.ones_like(torch.tensor(input))
    model_in = model_input(data.float()).reshape(n_examples, ch_in, -1)
    out = model(model_in)
    target = torch.tensor(output).reshape(n_examples, ch_out, -1)
    loss = torch.mean(torch.square(target - out))
    loss.backward()
    print(out.shape, model_input.bias.grad)
    for v in model_input.weight.grad[:, 0].reshape(-1):
        print(v.item())
    print('Output:', out.reshape([-1]).detach().numpy())
    print('Loss back:', 2 * (out - target).reshape([-1]).detach().numpy())
    print('')

def testSens1():
    runTestSens(
        1,              # test idx
        1, 1, 1,        # model shape
        [0.2],          # weight
        [1],            # bias
        [0.1],           # input
        [0.7],           # output
        1           # n_examples
    )

def testSens1_2():
    runTestSens(
        1,              # test idx
        1, 1, 1,        # model shape
        [0.2],          # weight
        [0.4],            # bias
        [0.1],           # input
        [0.7],           # output
        1           # n_examples
    )

def testSens1_3():
    runTestSens(
        1,              # test idx
        1, 1, 1,        # model shape
        [0.2],          # weight
        [1],            # bias
        [0.1, 0.2],           # input
        [0.7, 0.5],           # output
        2           # n_examples
    )

def testSens2():
    runTestSens(
        1,              # test idx
        1, 1, 1,        # model shape
        [0.4],          # weight
        [1],            # bias
        [0.1],           # input
        [0.7],           # output
        1           # n_examples
    )

def testSens3():
    runTestSens(
        1,              # test idx
        1, 1, 3,        # model shape
        [0.4, 0.6, 0.3],          # weight
        [1],            # bias
        [0.1, 0.5, 0.2],           # input
        [0.7],           # output
        1           # n_examples
    )

def testSens4():
    runTestSens(
        1,              # test idx
        1, 1, 3,        # model shape
        [0.4, 0.6, 0.3],          # weight
        [1],            # bias
        [0.1, 0.5, 0.2, 0.3],           # input
        [0.7, 0.7],           # output
        1           # n_examples
    )

def testSens5():
    runTestSens(
        1,              # test idx
        2, 1, 3,        # model shape
        [0.4, 0.6, 0.3, 0.1, 0.4, 0.2],          # weight
        [0.6],            # bias
        [0.1, 0.5, 0.2, 0.3, 0.4, 0.8, 0.5, 0.2],           # input
        [0.7, 0.5],           # output
        1           # n_examples
    )

def testSens6():
    runTestSens(
        1,              # test idx
        2, 2, 3,        # model shape
        [0.4, 0.6, 0.3, 0.1, 0.4, 0.2, 0.5, 0.6, 0.4, 0.1, 0.7, 0.3],          # weight
        # [0.4, 0.6, 0.3, 0.5, 0.6, 0.4, 0.1, 0.4, 0.2, 0.1, 0.7, 0.3],          # weight
        [1, 0.6],            # bias
        [0.1, 0.5, 0.2, 0.3, 0.4, 0.8, 0.5, 0.2, 0.5, 0.2, 0.7, 0.6, 0.3, 0.4, 0.9, 0.2],           # input
        [0.7, 0.5, 0.6, 0.2, 0.1, 0.3, 0.6, 0.1],           # output
        2           # n_examples
    )

if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # testN()
    # testGrad1()
    # testGrad2()
    # testGrad3()
    # testGrad4()
    # testGrad5()
    # testGrad6()
    # testGrad7()
    # testGrad8()
    # testGrad9()
    # testGrad10()
    # testSens1()
    # testSens1_2()
    # testSens1_3()
    # testSens2()
    # testSens3()
    # testSens4()
    # testSens5()
    testSens6()
