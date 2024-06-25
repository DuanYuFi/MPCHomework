import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

numbers = [16337017718880406793, 8819928064602880786, 11736542363935815776]

shares = []
for i in range(3):
    shares.append(protocol.input_share([numbers[i]], i, True)[0])


def full_adder(a, b, c):
    sum = protocol.xor_gate(protocol.xor_gate(a, b), c)
    carry = protocol.or_gate(
        protocol.and_gate(a, b), protocol.and_gate(c, protocol.xor_gate(a, b))
    )
    return sum, carry


def add(a, b, bit_length):
    sum = []
    carry = [RSS3PC(0, 0, modular=64)]
    for i in range(bit_length):
        sum_bit, carry = full_adder([a[i]], [b[i]], carry)
        sum.append(sum_bit[0])
    # sum.append(carry[0])
    return sum


a, b, c = shares

a_bits = []
b_bits = []
c_bits = []

for i in range(64):
    a_bits.append(RSS3PC(a[0] >> i & 1, a[1] >> i & 1, modular=64))
    b_bits.append(RSS3PC(b[0] >> i & 1, b[1] >> i & 1, modular=64))
    c_bits.append(RSS3PC(c[0] >> i & 1, c[1] >> i & 1, modular=64))

result = add(a_bits, b_bits, 64)
result = add(result, c_bits, 64)

result = protocol.reveal(result, to=0)

if player_id == 0:
    print(result)

protocol.disconnect()
