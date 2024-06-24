import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

a = protocol.input_share([123, 234, 345], 0, True)
b = protocol.input_share([456, 567, 678], 0, True)

c = protocol.xor_gate(a, b)
d = protocol.and_gate(a, b)
e = protocol.or_gate(a, b)

c = protocol.reveal(c, to=0)
d = protocol.reveal(d, to=0)
e = protocol.reveal(e, to=0)

if player_id == 0:
    print(c)
    print(d)
    print(e)

protocol.disconnect()