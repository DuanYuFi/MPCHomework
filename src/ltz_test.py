import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_f = [8.71828, 1.12345, -2.675654, -9.75645]

a = protocol.input_share(input_f, 0)

ltz = protocol.ltz(a)
result = protocol.reveal(ltz)

print(result)

protocol.disconnect()