from aby3protocol import *
import sys

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=33444)

datas = [0] * 3
for i in range(3):
    datas[i] = protocol.input_share([[123], [234], [345]][i], i)

result = protocol.mul_ss(datas[0], datas[1])
result = protocol.mul_ss(result, datas[2])

result = protocol.reveal(result, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()