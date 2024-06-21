import numpy as np
import torch

from aby3protocol import RSS3PC, Aby3Protocol, Matrix


# 卷积操作
def conv2d(
    X: Matrix,
    W: Matrix,
    b: Matrix,
    stride: int = 1,
    padding: int = 0,
    protocol: Aby3Protocol = None,
) -> Matrix:
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


# Sigmoid 激活函数
def sigmoid(X: Matrix, protocol: Aby3Protocol = None) -> Matrix:
    (n_H, n_W) = X.dimensions()
    X_sigmoid = Matrix(n_H, n_W)
    for i in range(n_H):
        for j in range(n_W):
            X_sigmoid[i, j] = 1 / (1 + np.exp(-X[i, j]))
    return X_sigmoid


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
    # 卷积层 1 -> Sigmoid -> 池化层 1
    X = conv2d(
        X,
        params["conv1_weight"],
        params["conv1_bias"],
        stride=1,
        padding=2,
        protocol=protocol,
    )
    X = sigmoid(X, protocol)
    X = avg_pool(X, f=2, stride=2, protocol=protocol)

    # 卷积层 2 -> Sigmoid -> 池化层 2
    X = conv2d(
        X,
        params["conv2_weight"],
        params["conv2_bias"],
        stride=1,
        padding=0,
        protocol=protocol,
    )
    X = sigmoid(X, protocol)
    X = avg_pool(X, f=2, stride=2, protocol=protocol)

    # 扁平化
    X = flatten(X)

    # 全连接层 1 -> Sigmoid
    X = dense(X, params["fc1_weight"], params["fc1_bias"], protocol)
    X = sigmoid(X, protocol)

    # 全连接层 2 -> Sigmoid
    X = dense(X, params["fc2_weight"], params["fc2_bias"], protocol)
    X = sigmoid(X, protocol)

    # 输出层
    X = dense(X, params["fc3_weight"], params["fc3_bias"], protocol)

    return X


def secret_share_data(data: list, protocol: Aby3Protocol) -> list:
    """
    将数据秘密共享到协议中

    参数:
    data (list): 输入数据
    protocol (Aby3Protocol): MPC协议实例

    返回:
    list: 秘密共享的数据
    """
    return protocol.input_share(data, protocol.player_id)


def infer(X: Matrix, params: dict, protocol: Aby3Protocol) -> list:
    """
    使用秘密共享的LeNet-5模型进行推理

    参数:
    X (Matrix): 输入数据
    params (dict): 模型参数
    protocol (Aby3Protocol): MPC协议实例

    返回:
    list: 推理结果
    """
    # 前向传播推理
    output = lenet_forward(X, params, protocol)
    # 通过协议将推理结果揭露
    revealed_output = protocol.reveal(output)
    return revealed_output


def load_params(file_path: str):
    """
    从文件加载模型参数

    参数:
    file_path (str): 文件路径

    返回:
    dict: 模型参数
    """
    params = torch.load(file_path)
    return {
        "conv1_weight": params["conv1.weight"].numpy().flatten().tolist(),
        "conv1_bias": params["conv1.bias"].numpy().flatten().tolist(),
        "conv2_weight": params["conv2.weight"].numpy().flatten().tolist(),
        "conv2_bias": params["conv2.bias"].numpy().flatten().tolist(),
        "fc1_weight": params["fc1.weight"].numpy().flatten().tolist(),
        "fc1_bias": params["fc1.bias"].numpy().flatten().tolist(),
        "fc2_weight": params["fc2.weight"].numpy().flatten().tolist(),
        "fc2_bias": params["fc2.bias"].numpy().flatten().tolist(),
        "fc3_weight": params["fc3.weight"].numpy().flatten().tolist(),
        "fc3_bias": params["fc3.bias"].numpy().flatten().tolist(),
    }


if __name__ == "__main__":
    import sys

    player_id = int(sys.argv[1])
    protocol = Aby3Protocol(player_id)

    # 从文件中读取参数和输入数据
    params = load_params("lenet5_params.pth")

    if player_id in [0, 1]:
        params_shared = {
            "conv1_weight": secret_share_data(params["conv1_weight"], protocol),
            "conv1_bias": secret_share_data(params["conv1_bias"], protocol),
            "conv2_weight": secret_share_data(params["conv2_weight"], protocol),
            "conv2_bias": secret_share_data(params["conv2_bias"], protocol),
            "fc1_weight": secret_share_data(params["fc1_weight"], protocol),
            "fc1_bias": secret_share_data(params["fc1_bias"], protocol),
            "fc2_weight": secret_share_data(params["fc2_weight"], protocol),
            "fc2_bias": secret_share_data(params["fc2_bias"], protocol),
            "fc3_weight": secret_share_data(params["fc3_weight"], protocol),
            "fc3_bias": secret_share_data(params["fc3_bias"], protocol),
        }
    else:
        input_data = np.random.randint(0, 256, (32, 32, 1)).flatten().tolist()
        X_shared = secret_share_data(input_data, protocol)

    if player_id == 2:
        X_shared = secret_share_data(input_data, protocol)
    else:
        X_shared = Matrix(
            32, 32, [0] * (32 * 32)
        )  # 对于持有模型参数的参与者，输入数据为零

    # 模型推理
    output = infer(X_shared, params_shared, protocol)

    # 显示结果
    if player_id == 2:
        print(f"推理结果: {output}")

    protocol.disconnect()
