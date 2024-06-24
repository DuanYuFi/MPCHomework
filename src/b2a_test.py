import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input = [123, 234, 345]

a = protocol.input_share(input, 0)
b = protocol.bit_decomposition(a)

result = protocol.reveal(b, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()