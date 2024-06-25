import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [2, -3, 4, -5]

input_f = [2.7182818284590452353, -3.1415926535897932384, 25.132741228718, -1e-6]

share_i = protocol.input_share(input_i, 0)
share_f = protocol.input_share(input_f, 0)

# print("input ok")

# reveal

reveal_i = protocol.reveal(share_i)

print(reveal_i)

# print(share_f[i][0].decimal)

reveal_f = protocol.reveal(share_f)

print(reveal_f)

protocol.disconnect()
