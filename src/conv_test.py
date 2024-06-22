import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_f = [2.7182818284590452353, 3.1415926535897932384, 25.132741228718, 1e-6]
share_f = protocol.input_share(input_f, 0)

result = protocol.reveal(share_f, to=0)

if player_id == 0:
    print(result)

share_i = protocol.f2i(share_f)
result = protocol.reveal(share_i, to=0)

if player_id == 0:
    print(result)

share_f_again = protocol.i2f(share_i)
result = protocol.reveal(share_f_again, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()