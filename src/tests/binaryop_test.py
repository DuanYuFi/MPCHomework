import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [2, -3, 4, -5]
input_f = [8.71828, 1.12345, -2.675654, -9.75645]


def run_test(test_name, lhs, rhs):
    print("Running test:", test_name)

    share1 = protocol.input_share(lhs, 0)
    share2 = protocol.input_share(rhs, 0)

    result = getattr(protocol, test_name)(share1, share2)
    result = protocol.reveal(result, to=0)

    correct_result = getattr(protocol, test_name + "_pp")(lhs, rhs)

    if player_id == 0:
        print(result)
        print(correct_result)


run_test("add", input_i, input_f)
run_test("mul", input_i, input_f)
run_test("sub", input_i, input_f)
run_test("mat_mul", Matrix(2, 2, input_i), Matrix(2, 2, input_f))

protocol.disconnect()
