
import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [2, -3, 4, -5]
input_f = [8.71828, 1.12345, -2.675654, -9.75645]

a = protocol.input_share(input_i, 0)
b = protocol.input_share(input_f, 0)
c = protocol.sub(a, b)

decompositions = protocol.bit_decomposition(c)
msb = []

for each in decompositions:
    msb.append(RSS3PC((each[0] >> 63) & 1, (each[1] >> 63) & 1))

msb = protocol.bit_injection(msb)

result = protocol.reveal(msb, to=0, sign=False)

if player_id == 0:
    print(result)

protocol.disconnect()