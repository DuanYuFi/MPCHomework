import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

from aby3protocol import *
from OT import OT3

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

ot3 = OT3(2, 1, 0, player_id, protocol)

if player_id == 2:
    m1 = [123, 234, 345]
    m2 = [-123, -234, -345]
    ot3.send(m1, m2)

elif player_id == 0:
    m = [0, 1, 0]
    ot3.choice(m)

elif player_id == 1:
    recv = ot3.receive()
    print(recv)

protocol.disconnect()
