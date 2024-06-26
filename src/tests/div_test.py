import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [6.1, -7.2, 8.3, -9.4]
input_f = 8.71828

a = protocol.input_share(input_i, 0)
b = input_f

c = protocol.div_goldschmidt_general_sp(a, b)

result = protocol.reveal(c)

print(result)

protocol.disconnect()
