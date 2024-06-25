import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input = [
    -4.67372434614279,
    91.7779206453715,
    -36.5414413166793,
    91.0786324224954,
    -50.0387628740142,
    39.8330898243399,
    -34.8634377329487,
    99.9197975054495,
    65.4276636868028,
    13.9880638134373,
]

# input = [int(each) for each in input]
# input = [round(each * 2 ** 20) for each in input]
print(input)

a = protocol.input_share(input, 0)

CONST_0 = protocol.zero_shares(len(input))
CONST_1 = protocol.input_share([1] * len(input), 1)

a = protocol.add(a, CONST_0)

# for i in range(len(a)):
#     a[i].set_decimal(0)

a = protocol.mul(a, CONST_1)

# a = protocol.add(a, CONST_0)
# a = protocol.mul(a, CONST_1)
# b = protocol.bit_decomposition(a)

result = protocol.reveal(a)
# result = [each / 2 ** 20 for each in result]

# if player_id == 0:
print(result)

# input2 = [123, 234, 345]
# input3 = [12, 34, 56]

# a = protocol.input_share(input2, 0)
# b = protocol.input_share(input3, 0)

# c = protocol.sub(a, b)
# d = protocol.bit_decomposition(c)

# result = protocol.reveal(d, to=0)

# if player_id == 0:
#     print(result)

# input_i = [2, -3, 4, -5]
# input_f = [8.71828, 1.12345, -2.675654, -9.75645]

# a = protocol.input_share(input_i, 0)
# b = protocol.input_share(input_f, 0)

# c = protocol.sub(a, b)
# d = protocol.bit_decomposition(c)

# result = protocol.reveal(d, to=0)

# if player_id == 0:
#     print(result)

protocol.disconnect()
