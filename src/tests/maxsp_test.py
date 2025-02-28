import os
import random
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_f = Matrix(2, 2, [8.71828, 1.12345, -2.675654, -9.75645])

a = protocol.input_share(input_f, 0)
result = protocol.max_sp(a, 0)

result = protocol.reveal(result)

print(result)

protocol.disconnect()
