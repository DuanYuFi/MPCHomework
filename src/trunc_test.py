import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333, debug=True)

if player_id == 0:
    data = protocol.input_share([186, 216, 183, 184], 0)
else:
    data = protocol.input_share([0] * 4, 0)

truncated = protocol.shift_right(data, 5)
result = protocol.reveal(truncated, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()
