import torch
import numpy as np

mse = torch.nn.MSELoss()

def run_forward(input, target, in_shape):
    input = torch.tensor(input).float().reshape(in_shape)
    target = torch.tensor(target).float().reshape(in_shape)
    out = mse(input, target)

    return out.item()

def forward_test_1():
    loss = run_forward(
        [0],
        [1],
        (1, 1, 1)
    )
    print(loss)

def forward_test_5():
    loss = run_forward(
        [-1, 4],
        [0, 1],
        (1, 2, 1)
    )
    print(loss)

def forward_test_6():
    loss = run_forward(
        [-1, 4, 0, 3],
        [0, 1, 2, 1],
        (2, 2, 1)
    )
    print(loss)

# forward_test_1()
forward_test_5()
# forward_test_6()
