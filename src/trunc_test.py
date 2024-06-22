import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333, debug=True)

if player_id == 0:
    data = protocol.input_share([18664, 21612, 18356, 18412], 0)
else:
    data = protocol.input_share([0] * 4, 0)

truncated = protocol.shift_right(data, 16)
result = protocol.reveal(truncated, to=0)

if player_id == 0:
    print(result)
    print([each >> 16 for each in [18664, 21612, 18356, 18412]])

protocol.disconnect()
