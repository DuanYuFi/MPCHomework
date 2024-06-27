import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        flat_data = list(self.flatten())
        return Matrix(dims[0], dims[1], flat_data)

    @classmethod
    def from_mpc_matrix(cls, matrix: Matrix) -> "MatrixND":
        dim = matrix.dimensions()
        data = np.array(matrix.data).reshape(dim)
        return cls(dim, data)


# 卷积操作
def conv2d(
    X: MatrixND,
    W: MatrixND,
    b: MatrixND,
    stride: int = 1,
    padding: int = 0,
    protocol: Aby3Protocol = None,
    progress: Progress = None,
    parent_task_id=None,
) -> MatrixND:

    (N, C_in, H_in, W_in) = X.shape
    (C_out, C_in_k, K_H, K_W) = W.shape

    assert C_in == C_in_k, "Channels of input and kernel are not equal!"

    H_out = int((H_in - K_H + 2 * padding) / stride) + 1
    W_out = int((W_in - K_W + 2 * padding) / stride) + 1
    Z = np.zeros((N, C_out, H_out, W_out), dtype=object)

    for n in range(N):
        child_task_id = progress.add_task(
            f"[cyan]Processing batch {n + 1}/{N}",
            total=C_out * H_out * W_out,
            parent=parent_task_id,
        )
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    sub_matrix = X[n, :, h : h + K_H, w : w + K_W].reshape(
                        C_in * K_H, K_W
                    )
                    kernel_matrix = W[c_out].reshape(C_in * K_H, K_W)
                    mat_res = protocol.mat_mul(
                        sub_matrix.to_mpc_matrix(), kernel_matrix.T.to_mpc_matrix()
                    )
                    Z[n, c_out, h, w] = [mat_res.data[0]]
                    for x in mat_res.data[1:]:
                        Z[n, c_out, h, w] = protocol.add(Z[n, c_out, h, w], [x])
                    Z[n, c_out, h, w] = Z[n, c_out, h, w][0]
                    progress.advance(child_task_id)
        progress.remove_task(child_task_id)

    X = MatrixND(Z.shape, Z)
    X_shared = MatrixND(
        X.shape, protocol.reveal(X.flatten().tolist())
    )
    print("Output from conv2d: ")
    print(X_shared)
    return X


# 平均池化操作
def avg_pool(
    X: MatrixND,
    f: int = 2,
    stride: int = 2,
    protocol: Aby3Protocol = None,
    progress: Progress = None,
    parent_task_id=None,
) -> MatrixND:
    (n_C_prev, n_H_prev, n_W_prev) = X.shape[1:]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    Z = np.zeros((X.shape[0], n_C_prev, n_H, n_W), dtype=object)

    child_task_id = progress.add_task(
        f"[cyan]Pooling...", total=n_H * n_W, parent=parent_task_id
    )
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
                    sum_list = protocol.add(
                        sum_vals.flatten().tolist(), X_slice.flatten().tolist()
                    )
                    sum_vals = np.array(sum_list).reshape(sum_vals.shape)

            div_list = protocol.div_sp(sum_vals.flatten().tolist(), f * f)
            div_vals = np.array(div_list).reshape(sum_vals.shape)
            Z[:, :, h, w] = div_vals
            progress.advance(child_task_id)
    progress.remove_task(child_task_id)

    X = MatrixND(Z.shape, Z)
    X_shared = MatrixND(
        X.shape, protocol.reveal(X.flatten().tolist())
    )
    print("Output from avg_pool: ")
    print(X_shared)
    return X


# ReLU 激活函数
def relu(X: MatrixND, protocol: Aby3Protocol = None) -> MatrixND:
    zero_matrix = np.zeros(X.shape, dtype=object)
    relu_output = protocol.max_sp(X.flatten().tolist(), zero_matrix.flatten().tolist())

    X = MatrixND(X.shape, relu_output)
    X_shared = MatrixND(
        X.shape, protocol.reveal(X.flatten().tolist())
    )
    print("Output from relu: ")
    print(X_shared)
    return X


# 扁平化，保留 batchsize 的维度
def flatten(X: MatrixND) -> MatrixND:
    return MatrixND((X.shape[0], -1), X.flatten())


# 全连接层
def dense(
    X: MatrixND, W: MatrixND, b: MatrixND, protocol: Aby3Protocol = None
) -> MatrixND:
    Z = protocol.mat_mul(X.to_mpc_matrix(), W.T.to_mpc_matrix())
    batch_size = X.shape[0]
    b_reshaped = np.tile(b, (batch_size, 1))
    Z = protocol.mat_add(Z, MatrixND(b_reshaped.shape, b_reshaped).to_mpc_matrix())
    Z_dims = (batch_size, W.shape[0])

    X = MatrixND(Z_dims, Z.data)
    X_shared = MatrixND(
        X.shape, protocol.reveal(X.flatten().tolist())
    )
    print("Output from dense: ")
    print(X_shared)
    return X


# LeNet-5 前向传播
def lenet_forward(X: MatrixND, params: dict, protocol: Aby3Protocol) -> MatrixND:
    with Progress() as progress:
        # 创建父任务
        parent_task_id = progress.add_task("[green]LeNet Forward Pass", total=7)

        X_shared = MatrixND(
            X.shape, protocol.reveal(X.flatten().tolist())
        )
        print("Source: ")
        print(X_shared)

        # 卷积层 1
        progress.update(
            parent_task_id, description=f"[green]Conv Layer 1 # input shape: {X.shape}"
        )
        X = conv2d(
            X,
            params["conv1.weight"],  # torch.Size([6, 1, 5, 5]),
            params["conv1.bias"],  # torch.Size([6]),
            stride=1,
            padding=0,
            protocol=protocol,
            progress=progress,
            parent_task_id=parent_task_id,
        )
        progress.advance(parent_task_id)

        # ReLU 层 1
        progress.update(
            parent_task_id, description=f"[green]ReLU Layer 1 # input shape: {X.shape}"
        )
        X = relu(X, protocol)
        progress.advance(parent_task_id)

        # 平均池化层 1
        progress.update(
            parent_task_id,
            description=f"[green]Avg Pool Layer 1 # input shape: {X.shape}",
        )
        X = avg_pool(
            X,
            f=2,
            stride=2,
            protocol=protocol,
            progress=progress,
            parent_task_id=parent_task_id,
        )  # pool0 output shape:  (1, 6, 14, 14)
        progress.advance(parent_task_id)

        # 卷积层 2
        progress.update(
            parent_task_id, description=f"[green]Conv Layer 2 # input shape: {X.shape}"
        )
        X = conv2d(
            X,
            params["conv2.weight"],  # torch.Size([16, 6, 5, 5]),
            params["conv2.bias"],  # torch.Size([16]),
            stride=1,
            padding=0,
            protocol=protocol,
            progress=progress,
            parent_task_id=parent_task_id,
        )  # conv1 output shape:  (1, 16, 10, 10)
        progress.advance(parent_task_id)

        # ReLU 层 2
        progress.update(
            parent_task_id, description=f"[green]ReLU Layer 2 # input shape: {X.shape}"
        )
        X = relu(X, protocol)
        progress.advance(parent_task_id)

        # 平均池化层 2
        progress.update(
            parent_task_id,
            description=f"[green]Avg Pool Layer 2 # input shape: {X.shape}",
        )
        X = avg_pool(
            X,
            f=2,
            stride=2,
            protocol=protocol,
            progress=progress,
            parent_task_id=parent_task_id,
        )  # pool1 output shape:  (1, 16, 5, 5)
        progress.advance(parent_task_id)

        # 扁平化层
        progress.update(
            parent_task_id, description=f"[green]Flatten Layer # input shape: {X.shape}"
        )
        X = flatten(X)
        progress.advance(parent_task_id)

        # 全连接层 1
        progress.update(
            parent_task_id, description=f"[green]FC Layer 1 # input shape: {X.shape}"
        )
        X = dense(
            X,
            params["fc1.weight"],  # torch.Size([120, 400]),
            params["fc1.bias"],  # torch.Size([120]),
            protocol,
        )  # dense0 output shape:         (1, 120)
        progress.advance(parent_task_id)

        # ReLU 层 3
        progress.update(
            parent_task_id, description=f"[green]ReLU Layer 3 # input shape: {X.shape}"
        )
        X = relu(X, protocol)
        progress.advance(parent_task_id)

        # 全连接层 2
        progress.update(
            parent_task_id, description=f"[green]FC Layer 2 # input shape: {X.shape}"
        )
        X = dense(
            X,
            params["fc2.weight"],  # torch.Size([84, 120]),
            params["fc2.bias"],  # torch.Size([84]),
            protocol,
        )
        progress.advance(parent_task_id)

        # ReLU 层 4
        progress.update(
            parent_task_id, description=f"[green]ReLU Layer 4 # input shape: {X.shape}"
        )
        X = relu(X, protocol)
        progress.advance(parent_task_id)

        # 输出层
        progress.update(
            parent_task_id, description=f"[green]FC Layer 3 # input shape: {X.shape}"
        )
        X = dense(
            X,
            params["fc3.weight"],  # torch.Size([10, 84])
            params["fc3.bias"],  # torch.Size([10]),
            protocol,
        )  # dense2 output shape:         (1, 10)
        progress.advance(parent_task_id)
        progress.update(parent_task_id, description=f"[green]Output shape: {X.shape}")

    return X


def infer(inputs):
    match = re.search(r"_(\d+)", sys.argv[0])
    if match:
        player_id = int(match.group(1))
        print(f"Using player-{player_id} in {sys.argv[0]}")
    else:
        player_id = int(sys.argv[1])
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
    revealed_outputs = MatrixND(
        outputs.shape, protocol.reveal(outputs.flatten().tolist())
    )
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(test_loader):
        padded_inputs = F.pad(inputs, (2, 2, 2, 2))
        outputs = infer(padded_inputs)
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        print(
            f"Testing [{i+1}/{len(test_loader)}]: Accuracy => {correct} / {total} = {100 * correct / total:.4f}"
        )


if __name__ == "__main__":
    test_accuracy()
