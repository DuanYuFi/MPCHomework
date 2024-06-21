import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=33024)

matrics = [
    [151, 164, 248, 137, 187, 121, 74, 75, 136, 239, 22, 141, 165, 188, 241, 26],
    [102, 208, 169, 9, 228, 119, 2, 56, 224, 233, 177, 206, 80, 29, 74, 237],
    [40, 180, 184, 157, 116, 127, 215, 8, 142, 251, 82, 84, 209, 94, 235, 44],
]

datas = [0] * 3

for i in range(3):
    datas[i] = protocol.input_share(matrics[i], i)

datas = [Matrix(4, 4, data) for data in datas]

result = protocol.mat_mul_ss(datas[0], datas[1])
print(result.dimensions())
result = protocol.mat_mul_ss(result, datas[2])
print(result.dimensions())

result = result.data

print(len(result))

result = protocol.reveal(result, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()
