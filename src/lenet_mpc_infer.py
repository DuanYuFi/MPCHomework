import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from aby3protocol import Aby3Protocol, Matrix

cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, "lenet_outdir")


# 卷积操作
# TODO: Need Check
def conv2d(
    X: Matrix,
    W: Matrix,
    b: Matrix,
    stride: int = 1,
    padding: int = 0,
    protocol: Aby3Protocol = None,
) -> Matrix:
    raise NotImplementedError()  # TODO

    (n_H_prev, n_W_prev) = X.dimensions()
    (f, f, n_C_prev, n_C) = W.dimensions()
    n_H = int((n_H_prev - f + 2 * padding) / stride) + 1
    n_W = int((n_W_prev - f + 2 * padding) / stride) + 1
    Z = Matrix(n_H, n_W * n_C)

    X_pad = np.pad(
        np.array(X.data).reshape(n_H_prev, n_W_prev, n_C_prev),
        ((padding, padding), (padding, padding), (0, 0)),
        "constant",
    )
    X_pad = Matrix(
        n_H_prev + 2 * padding,
        n_W_prev + 2 * padding * n_C_prev,
        X_pad.flatten().tolist(),
    )

    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                X_slice = X_pad[vert_start:vert_end, horiz_start:horiz_end]
                Z[h, w * n_C + c] = protocol.mat_mul_ss(X_slice, W).data[0] + b.data[c]

    return Z


# 平均池化操作
def avg_pool(
    X: Matrix, f: int = 2, stride: int = 2, protocol: Aby3Protocol = None
) -> Matrix:
    raise NotImplementedError()  # TODO

    (n_H_prev, n_W_prev) = X.dimensions()
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    Z = Matrix(n_H, n_W)

    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride
            vert_end = vert_start + f
            horiz_start = w * stride
            horiz_end = horiz_start + f

            sum_val = protocol.input_share([0], 0)
            for i in range(vert_start, vert_end):
                for j in range(horiz_start, horiz_end):
                    sum_val = protocol.add_ss(
                        sum_val, protocol.input_share([X[i, j]], 0)
                    )

            Z[h, w] = sum_val[0] / (f * f)

    return Z


# ReLU 激活函数
def relu(X: Matrix, protocol: Aby3Protocol = None) -> Matrix:
    (n_H, n_W) = X.dimensions()
    zero_matrix = Matrix(n_H, n_W, data=None)
    X_relu = protocol.mat_max_sp(X, zero_matrix)
    return X_relu


# 扁平化
def flatten(X: Matrix) -> Matrix:
    (n_H, n_W) = X.dimensions()
    return Matrix(1, n_H * n_W, X.data)


# 全连接层
def dense(X: Matrix, W: Matrix, b: Matrix, protocol: Aby3Protocol = None) -> Matrix:
    X_flat = flatten(X)
    Z = protocol.mat_mul_ss(X_flat, W)
    Z = protocol.mat_add_sp(Z, b)
    return Z


# LeNet-5 前向传播
def lenet_forward(X: Matrix, params: dict, protocol: Aby3Protocol) -> Matrix:
    # 卷积层 1 -> ReLU -> 池化层 1
    X = conv2d(
        X,
        params["conv1.weight"],
        params["conv1.bias"],
        stride=1,
        padding=2,
        protocol=protocol,
    )
    X = relu(X, protocol)
    X = avg_pool(X, f=2, stride=2, protocol=protocol)

    # 卷积层 2 -> ReLU -> 池化层 2
    X = conv2d(
        X,
        params["conv2.weight"],
        params["conv2.bias"],
        stride=1,
        padding=0,
        protocol=protocol,
    )
    X = relu(X, protocol)
    X = avg_pool(X, f=2, stride=2, protocol=protocol)

    # 扁平化
    X = flatten(X)

    # 全连接层 1 -> ReLU
    X = dense(X, params["fc1.weight"], params["fc1.bias"], protocol)
    X = relu(X, protocol)

    # 全连接层 2 -> ReLU
    X = dense(X, params["fc2.weight"], params["fc2.bias"], protocol)
    X = relu(X, protocol)

    # 输出层
    X = dense(X, params["fc3.weight"], params["fc3.bias"], protocol)

    return X


def infer(inputs):
    player_id = int(sys.argv[1])
    protocol = Aby3Protocol(player_id)

    # 从文件中读取参数和输入数据
    """
    ====== secret_params ======
    {
        'conv1.bias': torch.Size([6]),
        'conv1.weight': torch.Size([6, 1, 5, 5]),
        'conv2.bias': torch.Size([16]),
        'conv2.weight': torch.Size([16, 6, 5, 5]),
        'fc1.bias': torch.Size([120]),
        'fc1.weight': torch.Size([120, 400]),
        'fc2.bias': torch.Size([84]),
        'fc2.weight': torch.Size([84, 120]),
        'fc3.bias': torch.Size([10]),
        'fc3.weight': torch.Size([10, 84])
    }
    """
    # player0 和 player1 分别持有
    if protocol.player_id == 0:
        secret_params, input_data = (
            torch.load(os.path.join(out_dir, f"lenet5_params_0.pth")),
            None,
        )
    elif protocol.player_id == 1:
        secret_params, input_data = (
            torch.load(os.path.join(out_dir, f"lenet5_params_1.pth")),
            None,
        )
    elif protocol.player_id == 2:
        valid_params = torch.load(os.path.join(out_dir, f"lenet5_params_0.pth"))
        # load valid params, but we do not need the value, just replace weights with zeros
        # since player2 do not know any about model params
        secret_params = {k: torch.zeros_like(v) for k, v in valid_params}
        input_data = inputs
    else:
        assert False, "unreachable"

    params_shared = {}
    for k, v in secret_params.items():
        params_shared0 = protocol.input_share(v.numpy().flatten().tolist(), 0)
        params_shared1 = protocol.input_share(v.numpy().flatten().tolist(), 1)
        # calc average
        params_sum = protocol.mat_add_ss(params_shared0, params_shared1)
        params_shared[k] = protocol.mat_div_sp(params_sum, 2)

    X_shared = protocol.input_share(input_data, 2)

    # 模型前向传播推理
    outputs = lenet_forward(X_shared, params_shared, protocol)
    # 通过协议将推理结果揭露
    revealed_outputs = protocol.reveal(outputs)
    protocol.disconnect()

    # 显示结果
    return revealed_outputs


def test_accuracy():
    # prepare test dataset
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=os.path.join(out_dir, "MNIST_test_datasets"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    for inputs, labels in tqdm(test_loader, desc="Testing"):
        outputs = infer(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
    )


if __name__ == "__main__":
    test_accuracy()
