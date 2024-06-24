import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

a = [2.1]
b = [8.71828]

fix_a = [round(2.1 * 2 ** 20)]
fix_b = [round(8.71828 * 2 ** 20)]

share_a = protocol.input_share(fix_a, 0)
share_b = protocol.input_share(fix_b, 0)

# rev_a = protocol.reveal(share_a, to=0)
# rev_b = protocol.reveal(share_b, to=0)

# if player_id == 0:
#     print(rev_a)
#     print(rev_b)

tmp = protocol.mul_ss(share_a, share_b)
tmp = protocol.shift_right(tmp, 20)
# tmp = protocol.reduce(tmp, 64)

result = protocol.reveal(tmp, to=0)

if player_id == 0:
    result = [each / 2 ** 20 for each in result]
    print(result)

protocol.disconnect()