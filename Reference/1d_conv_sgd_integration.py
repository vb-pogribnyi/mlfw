import numpy as np
import torch
import matplotlib.pyplot as plt

N_EXAMPLES = 13
N_POINTS = 100
TEST_NAME = 'conv1d_sgd'
np.random.seed(42)

def get_example():
    x = np.linspace(1, N_POINTS / 10, N_POINTS)
    r = np.random.random() * 0.8 + 0.6
    noise = np.random.normal(0, 0.3, len(x))
    y = np.sin(x / r) + noise
    target = np.cos(x / r)

    return y, target

def print_c(arr, name):
    idx = 0
    idx_step = 10
    print('float {}[] ='.format(name), '{')
    while idx < len(arr):
        print(', '.join(arr[idx:idx + idx_step].astype(str)), end='')
        idx += idx_step
        if idx < len(arr):
            print(',')
        else:
            print('')
    print('};')

data = []
lbls = []
for i in range(N_EXAMPLES):
    ex = get_example()
    data.append(ex[0])
    lbls.append(ex[1][1:-1])
    # plt.plot(ex[0])
    # plt.plot(ex[1])
    # plt.show()
data = np.array(data)
lbls = np.array(lbls)
print_c(data.copy().reshape(-1), '{}_input'.format(TEST_NAME))
print_c(lbls.copy().reshape(-1), '{}_target'.format(TEST_NAME))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)

        self.conv1.weight.data = torch.tensor([-0.07021185, -0.54910874, 0.5624425]).reshape(self.conv1.weight.data.shape)
        self.conv1.bias.data = torch.tensor([-0.42805082]).reshape(self.conv1.bias.data.shape)

    def forward(self, x):
        return self.conv1(x)

model = Model()
data = torch.Tensor(data).unsqueeze(1)
lbls = torch.Tensor(lbls).unsqueeze(1)
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
mse = torch.nn.MSELoss()
print_c(model(data).detach().numpy().reshape(-1), '{}_out'.format(TEST_NAME))
print_c(model.conv1.weight.data.numpy().reshape(-1), '{}_weight_start'.format(TEST_NAME))
print_c(model.conv1.bias.data.numpy().reshape(-1), '{}_bias_start'.format(TEST_NAME))
for epoch in range(10):
    opt.zero_grad()
    out = model(data)
    loss = mse(out, lbls)
    # if epoch % 100 == 0:
    #     print(epoch, loss)
    loss.backward()
    opt.step()
    # print_c(out.detach().numpy().reshape(-1), '{}_out{}'.format(TEST_NAME, epoch))
    print_c(loss.detach().numpy().reshape(-1), '{}_loss{}'.format(TEST_NAME, epoch))
    print_c(model.conv1.weight.data.numpy().reshape(-1), '{}_weight{}'.format(TEST_NAME, epoch))
    print_c(model.conv1.bias.data.numpy().reshape(-1), '{}_bias{}'.format(TEST_NAME, epoch))
plt.plot(data.detach().numpy()[0, 0])
plt.plot(lbls.detach().numpy()[0, 0])
plt.plot(model(data).detach().numpy()[0, 0])
plt.show()

