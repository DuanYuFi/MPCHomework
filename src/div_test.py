import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_f = Matrix(2, 2, [8.71828, 1.12345, -2.675654, -9.75645])
input_i = 3

a = protocol.input_share(input_f, 0)
result = protocol.mat_div_sp(a, input_i)

result = protocol.reveal(result)

print(result)

protocol.disconnect()
