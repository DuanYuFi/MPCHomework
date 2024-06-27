import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [2, -3, 4, -5] * 1024
input_f = [8.71828, 1.12345, -2.675654, -9.75645] * 1024

a = protocol.input_share(input_i, 0)
b = protocol.input_share(input_f, 0)

c = protocol.compare(a, b)

result = protocol.reveal(c)

print(result)

protocol.disconnect()
