import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]

input_f = [[2.1, 3.1, 4.1, 5.1], [6.1, 7.1, 8.1, 9.1], [10.1, 11.1, 12.1, 13.1]]

share_i = [0] * 3
for i in range(3):
    share_i[i] = protocol.input_share(input_i[i], i)

share_f = [0] * 3
for i in range(3):
    share_f[i] = protocol.input_share(input_f[i], i)

print("input ok")

# reveal

reveal_i = [0] * 3
for i in range(3):
    reveal_i[i] = protocol.reveal(share_i[i], to=0)

if player_id == 0:
    print(reveal_i)

# print(share_f[i][0].decimal)

reveal_f = [0] * 3
for i in range(3):
    reveal_f[i] = protocol.reveal(share_f[i], to=0)

if player_id == 0:
    print(reveal_f)

protocol.disconnect()
