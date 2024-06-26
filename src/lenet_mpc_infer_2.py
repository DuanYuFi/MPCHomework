import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F

from aby3protocol import Aby3Protocol, Matrix

cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, "..", "output")


class MatrixND(np.ndarray):
    def __new__(cls, dims, data=None):
        if data is None:
            data = np.zeros(dims, dtype=object)
        obj = np.asarray(data).reshape(dims).view(cls)
        return obj

    def to_mpc_matrix(self):
        dims = self.shape
        assert len(dims) == 2
        flat_data = self.flatten()
        return Matrix(dims[0], dims[1], flat_data)

    @classmethod
    def from_mpc_matrix(cls, matrix: Matrix) -> "MatrixND":
        dim = matrix.dimensions()
        data = np.array(matrix.data).reshape(dim)
        return cls(dim, data)


# 卷积操作
# TODO: Need Check
def conv2d(
    X: MatrixND,
    W: MatrixND,
    b: MatrixND,
    stride: int = 1,
    padding: int = 0,
    protocol: Aby3Protocol = None,
) -> MatrixND:
    
    (N, C_in, H_in, W_in) = X.shape 
    (C_out, C_in_k, K_H, K_W) = W.shape 
    
    assert C_in == C_in_k, "Channels of input and kernel are not equal!"
    
    H_out = int((H_in - K_H + 2 * padding) / stride) + 1
    W_out = int((W_in - K_W + 2 * padding) / stride) + 1
    Z = np.zeros((N, C_out, H_out, W_out), dtype=object)
    vis = np.zeros((N, C_out, H_out, W_out), dtype=object)
    
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # 提取输入张量中的子矩阵
                    sub_matrix = X[n, :, h:h + K_H, w:w + K_W].reshape(C_in * K_H, K_W)
                    # 将卷积核reshape成合适的形状
                    kernel_matrix = W[c_out].reshape(C_in * K_H, K_W)
                    # 使用mat_mul进行矩阵乘法
                    Z[n, c_out, h, w] = np.sum(protocol.mat_mul_ss(sub_matrix.to_mpc_matrix(), kernel_matrix.T.to_mpc_matrix()))                    # for c_in in range(C_in):
                    #     for kh in range(K_H):
                    #         for kw in range(K_W):
                    #             mul_x = protocol.mul_ss([X[n, c_in, h + kh, w + kw]], [W[c_out, c_in, kh, kw]])
                    #             if vis[n, c_out, h, w] == 0:
                    #                 vis[n, c_out, h, w] = 1 
                    #                 Z[n, c_out, h, w] = protocol.add_sp(mul_x, [0])
                    #             else:
                    #                 Z[n, c_out, h, w] = protocol.add_ss(mul_x, Z[n, c_out, h, w])
                    
        
    return MatrixND(Z.shape, Z)


# 平均池化操作
def avg_pool(
    X: MatrixND, f: int = 2, stride: int = 2, protocol: Aby3Protocol = None
) -> MatrixND:
    (n_C_prev, n_H_prev, n_W_prev) = X.shape[1:]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    Z = np.zeros((X.shape[0], n_C_prev, n_H, n_W), dtype=object)

    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride
            horiz_start = w * stride

            # 将 sum_vals 初始值设置为池化窗口中的第一个加数
            sum_vals = X[:, :, vert_start, horiz_start]

            for i in range(f):
                for j in range(f):
                    if i == 0 and j == 0:
                        continue  # 第一个值已经设置为 sum_vals 的初始值
                    X_slice = X[:, :, vert_start + i, horiz_start + j]
                    sum_list = protocol.add_ss(
                        sum_vals.flatten().tolist(), X_slice.flatten().tolist()
                    )
                    sum_vals = np.array(sum_list).reshape(sum_vals.shape)

            div_list = protocol.div_sp(sum_vals.flatten().tolist(), f * f)
            div_vals = np.array(div_list).reshape(sum_vals.shape)
            Z[:, :, h, w] = div_vals

    return MatrixND(Z.shape, Z)


# ReLU 激活函数
def relu(X: MatrixND, protocol: Aby3Protocol = None) -> MatrixND:
    zero_matrix = np.zeros(X.shape, dtype=object)
    relu_output = protocol.max_sp(X.flatten().tolist(), zero_matrix.flatten().tolist())

    return MatrixND(X.shape, relu_output)


# 扁平化，保留 batchsize 的维度
def flatten(X: MatrixND) -> MatrixND:
    return MatrixND((X.shape[0], -1), X.flatten())


# 全连接层
# TODO: 代码写完但还没调试
def dense(
    X: MatrixND, W: MatrixND, b: MatrixND, protocol: Aby3Protocol = None
) -> MatrixND:
    Z = protocol.mat_mul_ss(X.to_mpc_matrix(), W.T.to_mpc_matrix())
    batch_size = X.shape[0]
    b_reshaped = np.tile(b, (batch_size, 1))
    Z = protocol.mat_add_ss(Z, MatrixND(b_reshaped.shape, b_reshaped).to_mpc_matrix())
    Z_dims = (batch_size, W.shape[0])
    return MatrixND(Z_dims, Z.data)


# LeNet-5 前向传播
def lenet_forward(X: MatrixND, params: dict, protocol: Aby3Protocol) -> MatrixND:
    # 卷积层 1 -> ReLU -> 池化层 1
    X = conv2d(
        X,
        params["conv1.weight"],  # torch.Size([6, 1, 5, 5]),
        params["conv1.bias"],  # torch.Size([6]),
        stride=1,
        padding=0,
        protocol=protocol,
    )
    X = relu(X, protocol)  # conv0 output shape:  (1, 6, 28, 28)
    X = avg_pool(
        X, f=2, stride=2, protocol=protocol
    )  # pool0 output shape:  (1, 6, 14, 14)

    # 卷积层 2 -> ReLU -> 池化层 2
    X = conv2d(
        X,
        params["conv2.weight"],  # torch.Size([16, 6, 5, 5]),
        params["conv2.bias"],  # torch.Size([16]),
        stride=1,
        padding=0,
        protocol=protocol,
    )  # conv1 output shape:  (1, 16, 10, 10)
    X = relu(X, protocol)
    X = avg_pool(
        X, f=2, stride=2, protocol=protocol
    )  # pool1 output shape:  (1, 16, 5, 5)

    # 扁平化
    X = flatten(X)

    # 全连接层 1 -> ReLU
    X = dense(
        X,
        params["fc1.weight"],  # torch.Size([120, 400]),
        params["fc1.bias"],  # torch.Size([120]),
        protocol,
    )  # dense0 output shape:         (1, 120)
    X = relu(X, protocol)

    # 全连接层 2 -> ReLU
    X = dense(
        X,
        params["fc2.weight"],  # torch.Size([84, 120]),
        params["fc2.bias"],  # torch.Size([84]),
        protocol,
    )
    X = relu(X, protocol)  # dense1 output shape:         (1, 84)

    # 输出层
    X = dense(
        X,
        params["fc3.weight"],  # torch.Size([10, 84])
        params["fc3.bias"],  # torch.Size([10]),
        protocol,
    )  # dense2 output shape:         (1, 10)

    return X


def infer(inputs):
    player_id = 2
    protocol = Aby3Protocol(player_id)

    # 从文件中读取参数和输入数据
    # player0 和 player1 分别持有
    valid_params = torch.load(os.path.join(out_dir, f"lenet5_params.pth"))
    if protocol.player_id == 0:
        secret_params, input_data = valid_params, torch.zeros_like(inputs)
    elif protocol.player_id == 1:
        # load valid params, but we do not need the value, just replace weights with zeros
        # since player2 do not know any about model params
        secret_params = {k: torch.zeros_like(v) for k, v in valid_params.items()}
        input_data = inputs
    elif protocol.player_id == 2:
        # load valid params, but we do not need the value, just replace weights with zeros
        # since player2 do not know any about model params
        secret_params = {k: torch.zeros_like(v) for k, v in valid_params.items()}
        input_data = torch.zeros_like(inputs)
    else:
        assert False, "unreachable"

    # noleak
    del valid_params

    params_shared = {
        k: MatrixND(v.shape, protocol.input_share(v.numpy().flatten().tolist(), 0))
        for k, v in secret_params.items()
    }
    X_shared = MatrixND(
        input_data.shape, protocol.input_share(input_data.flatten().tolist(), 1)
    )

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

    for i, (inputs, labels) in enumerate(test_loader):
        outputs = infer(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tqdm.write(
            f"Testing [{i+1}/{len(test_loader)}]: Accuracy => {correct} / {total} = {100 * correct / total:.4f}"
        )


if __name__ == "__main__":
    test_accuracy()
