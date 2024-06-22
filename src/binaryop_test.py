import sys

from aby3protocol import *

player_id = int(sys.argv[1])
protocol = Aby3Protocol(player_id, port_base=23333)

input_i = [2, -3, 4, -5]
input_f = [2.7182818284590452353, -3.1415926535897932384, 25.132741228718, -1e-6]

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

protocol.disconnect()