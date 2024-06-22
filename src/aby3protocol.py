import random

from network import Player


def arithmetic_shift_right(x, offset, nbits=64):
    if x >> (nbits - 1):
        return (x >> offset) + ((2 ** nbits - 1) << (nbits - offset))
    else:
        return x >> offset

class RSS3PC:
    data: list
    decimal: int

    def __init__(self, s1, s2, decimal=0):
        self.data = [s1, s2]
        self.decimal = decimal

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __str__(self):
        return f"({self.data[0]}, {self.data[1]})"

    def __repr__(self):
        return f"({self.data[0]}, {self.data[1]})"

    def set_decimal(self, decimal):
        self.decimal = decimal


class Matrix:
    data: list
    nrows: int
    ncols: int

    def __init__(self, n, m, data=None):
        if data is None:
            data = [0] * (n * m)
        else:
            self.data = data
        self.nrows = n
        self.ncols = m

    def row(self, index):
        if index >= self.nrows:
            raise IndexError("Index out of range")
        return self.data[index * self.ncols : (index + 1) * self.ncols]

    def col(self, index):
        if index >= self.ncols:
            raise IndexError("Index out of range")
        return [self.data[i * self.ncols + index] for i in range(self.nrows)]

    def dimensions(self):
        return (self.nrows, self.ncols)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            i, j = index
            return self.data[i * self.ncols + j]
        return self.row(index)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            assert len(index) == 2, "Index must be a tuple of length 2"
            i, j = index
            self.data[i * self.ncols + j] = value
        else:
            self.data[index * self.ncols : (index + 1) * self.ncols] = value

    def transpose(self):
        data = []
        for j in range(self.ncols):
            data.extend(self.col(j))
        return Matrix(self.ncols, self.nrows, data)


class Aby3Protocol:
    """
    ABY3 protocol
    """

    player: Player
    PRNGs: list
    modular: int

    def __init__(
        self, player_id, modular_bit=64, demical_bit=32, port_base=None, debug=False
    ):
        self.player = Player(player_id, 3, port_base=port_base, debug=debug)
        seed = random.getrandbits(32)
        self.PRNGs = [None, None]
        self.PRNGs[1] = random.Random(seed)
        seed = self.player.pass_around(seed, 1)
        self.PRNGs[0] = random.Random(seed)

        self.modular = 1 << modular_bit
        self.player_id = player_id
        self.demical = demical_bit
        self.modular_bit = modular_bit

    def disconnect(self):
        self.player.disconnect()

    @staticmethod
    def get_type(shares):
        """
        return the types of input

        Args:
            shares: list of RSS3PC or int or float

        Returns:
            tuple: (bool, bool)
                bool: True if shares is secret, False if shares is public.
                bool: True if shares is fixed point, False if shares is integer.
        """

        if isinstance(shares, Matrix):
            shares = shares[0]

        if isinstance(shares[0], RSS3PC):
            if shares[0].decimal > 0:
                return True, True
            else:
                return True, False
        elif isinstance(shares[0], int):
            return False, False
        elif isinstance(shares[0], float):
            return False, True
        else:
            raise ValueError(f"Invalid type {type(shares[0])} for shares")

    def input_share(self, public: list, owner: int):
        if all(isinstance(each_public, int) for each_public in public):
            return self.input_share_i(public, owner)
        elif any(isinstance(each_public, float) for each_public in public):
            return self.input_share_f(public, owner)
        else:
            raise ValueError(f"Invalid type {type(public[0])} for public value")

    def input_share_i(self, public: int, owner: int):
        ret = [RSS3PC(0, 0) for _ in range(len(public))]

        if owner == self.player.player_id:
            public = [each if each >= 0 else each + self.modular for each in public]
            send_buffer = []
            for i in range(len(public)):
                r = self.PRNGs[0].randrange(self.modular)
                ret[i][0] = r
                ret[i][1] = (public[i] - r) % self.modular
                send_buffer.append(ret[i][1])
            self.player.send(send_buffer, 1)

        elif owner == (self.player.player_id - 1) % 3:
            recv_data = self.player.recv(-1)
            for i in range(len(public)):
                ret[i][0] = recv_data[i]
                ret[i][1] = 0

        else:
            for i in range(len(public)):
                ret[i][0] = 0
                ret[i][1] = self.PRNGs[1].randrange(self.modular)

        return ret

    def input_share_f(self, public: float, owner: int):
        public = [round(each * 2**self.demical) for each in public]
        ret = self.input_share_i(public, owner)
        for i in range(len(ret)):
            ret[i].set_decimal(self.demical)
        return ret

    def reveal(self, secret: list, to=None):
        """
        Reveal secret shared values
        """
        if secret[0].decimal > 0:
            return self.reveal_f(secret, to)
        else:
            return self.reveal_i(secret, to)

    def reveal_i(self, secret: list, to):
        if to is None:
            ret = [0 for _ in range(len(secret))]
            send_buffer = []

            for i in range(len(secret)):
                ret[i] = (secret[i][0] + secret[i][1]) % self.modular
                send_buffer.append(secret[i][0])

            recv_data = self.player.pass_around(send_buffer, 1)
            for i in range(len(secret)):
                ret[i] = (ret[i] + recv_data[i]) % self.modular

            ret = [each if each < self.modular // 2 else each - self.modular for each in ret]
            return ret
        else:
            if to == (self.player_id + 1) % 3:
                send_buffer = [secret[i][0] for i in range(len(secret))]
                self.player.send(send_buffer, 1)
            elif to == self.player_id:
                recv = self.player.recv(-1)
                ret = [
                    (secret[i][0] + secret[i][1] + recv[i]) % self.modular
                    for i in range(len(secret))
                ]
                ret = [each if each < self.modular // 2 else each - self.modular for each in ret]
                return ret

    def reveal_f(self, secret: list, to):
        ret = self.reveal_i(secret, to)
        if to is None or to == self.player_id:
            for i in range(len(ret)):
                ret[i] /= 2 ** secret[i].decimal
            return ret

    def shift_left(self, shares: list, shift: int):
        return self.mul_sp(shares, [2**shift] * len(shares))

    def shift_right(self, shares: list, shift: int):
        ret = [RSS3PC(0, 0) for _ in range(len(shares))]
        if self.player_id == 0:
            for i in range(len(shares)):
                ret[i][0] = arithmetic_shift_right(shares[i][0], shift, self.modular_bit)

            recv_data = self.player.recv(1)
            for i in range(len(shares)):
                ret[i][1] = recv_data[i]

        elif self.player_id == 1:
            rs = [self.PRNGs[1].randrange(self.modular) for _ in range(len(shares))]
            send_buffer = []
            for i in range(len(shares)):
                ret[i][0] = arithmetic_shift_right((shares[i][0] + shares[i][1]) % self.modular, shift, self.modular_bit)
                ret[i][0] = (ret[i][0] - rs[i]) % self.modular
                send_buffer.append(ret[i][0])

                ret[i][1] = rs[i]

            self.player.send(send_buffer, -1)

        else:
            rs = [self.PRNGs[0].randrange(self.modular) for _ in range(len(shares))]
            for i in range(len(shares)):
                ret[i][0] = rs[i]
                ret[i][1] = arithmetic_shift_right(shares[i][0], shift, self.modular_bit)

        return ret

    def i2f(self, shares: list):
        shares = self.shift_left(shares, self.demical)
        for i in range(len(shares)):
            shares[i].set_decimal(self.demical)

        return shares
    
    def f2i(self, shares: list):
        shares = self.shift_right(shares, self.demical)
        for i in range(len(shares)):
            shares[i].set_decimal(0)

        return shares
    
    def binary_wrapper(self, operation: str, lhs, rhs):
        """
        TODO: multiplication between fixed point numbers
            needs one truncation.

        Wrapper for all kinds (16) of binary operation.
        """

        l_type = self.get_type(lhs)
        r_type = self.get_type(rhs)

        binop_pp = getattr(self, operation + "_pp")
        binop_ss = getattr(self, operation + "_ss")
        binop_sp = getattr(self, operation + "_sp")

        if (not l_type[0]) and (not r_type[0]):
            return binop_pp(lhs, rhs)
        
        if l_type[1] == r_type[1]:
            if l_type[0] == r_type[0]:
                return binop_ss(lhs, rhs)
            else:
                return binop_sp(lhs, rhs)

        # Now, there must one fixed point and one integer. We have to convert integer to fixed point.

        if not l_type[1]:
            if l_type[0]:
                lhs = self.i2f(lhs)
            else:
                lhs = [each * 2**self.demical for each in lhs]
        else:
            if r_type[0]:
                rhs = self.i2f(rhs)
            else:
                rhs = [each * 2**self.demical for each in rhs]
        
        if l_type[0] == r_type[0]:
            return binop_ss(lhs, rhs)
        else:
            return binop_sp(lhs, rhs)
    
    def add(self, lhs, rhs):
        return self.binary_wrapper("add", lhs, rhs)
    
    def add_pp(self, lhs, rhs):
        """
        Add two public values
        """

        ret = [(lhs[i] + rhs[i]) % self.modular for i in range(len(lhs))]
        ret = [each if each < self.modular // 2 else each - self.modular for each in ret]
        return ret

    def add_ss(self, lhs: list, rhs: list):
        """
        Add two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"

        return [
            RSS3PC(
                (_lhs[0] + _rhs[0]) % self.modular, (_lhs[1] + _rhs[1]) % self.modular
            )
            for _lhs, _rhs in zip(lhs, rhs)
        ]

    def add_sp(self, lhs, rhs):
        """
        Add secret shared value with public value
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        if isinstance(rhs[0], RSS3PC):
            lhs, rhs = rhs, lhs

        ret = [RSS3PC(lhs[i][0], lhs[i][1]) for i in range(len(lhs))]
        if self.player_id == 0:
            for i in range(len(lhs)):
                ret[i][1] = (ret[i][1] + rhs[i]) % self.modular
        elif self.player_id == 1:
            for i in range(len(lhs)):
                ret[i][0] = (ret[i][0] + rhs[i]) % self.modular

        return ret
    
    def mul(self, lhs, rhs):
        return self.binary_wrapper("mul", lhs, rhs)

    def mul_pp(self, lhs, rhs):
        """
        Multiply two public values
        """

        ret = [(lhs[i] * rhs[i]) % self.modular for i in range(len(lhs))]
        ret = [each if each < self.modular // 2 else each - self.modular for each in ret]
        return ret

    def mul_ss(self, lhs, rhs):
        """
        Multiply two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"

        ret = [RSS3PC(0, 0) for _ in lhs]
        send_buffer = []

        for i in range(len(lhs)):
            ret[i][0] = (
                lhs[i][0] * rhs[i][0]
                + lhs[i][1] * rhs[i][0]
                + lhs[i][0] * rhs[i][1]
                + self.PRNGs[0].randrange(self.modular)
                - self.PRNGs[1].randrange(self.modular)
            ) % self.modular

            send_buffer.append(ret[i][0])

        recv_data = self.player.pass_around(send_buffer, -1)
        for i in range(len(lhs)):
            ret[i][1] = recv_data[i]

        return ret

    def mul_sp(self, lhs, rhs):
        """
        Multiply secret shared value with public value
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        if isinstance(rhs[0], RSS3PC):
            lhs, rhs = rhs, lhs

        return [
            RSS3PC(
                (lhs[i][0] * rhs[i]) % self.modular, (lhs[i][1] * rhs[i]) % self.modular
            )
            for i in range(len(lhs))
        ]
    
    def mat_mul(self, lhs, rhs):
        return self.binary_wrapper("mat_mul", lhs, rhs)

    def mat_mul_pp(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of public values (n x m)
        rhs: Matrix of public values (m x k)

        Returns: Matrix of public values (n x k)
        """

        n, m = lhs.dimensions()
        m2, k = rhs.dimensions()

        assert m == m2, "Dimensions of matrices do not match"

        ret = Matrix(n, k)
        for i in range(n):
            for j in range(k):
                ret[i, j] = sum(lhs[i, x] * rhs[x, j] for x in range(m)) % self.modular

        return ret

    def mat_mul_ss(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of secret shared values (n x m)
        rhs: Matrix of secret shared values (m x k)

        Returns: Matrix of secret shared values (n x k)
        """

        n, m = lhs.dimensions()
        m2, k = rhs.dimensions()

        assert m == m2, "Dimensions of matrices do not match"

        LHS = []
        RHS = []

        for i in range(n):
            LHS.extend(lhs.row(i) * k)
            for j in range(k):
                RHS.extend(rhs.col(j))

        ret = self.mul_ss(LHS, RHS)

        slices = [[ret[i + j * m] for j in range(n * k)] for i in range(m)]

        result = slices[0]
        for i in range(1, m):
            result = self.add_ss(result, slices[i])

        ret_mat = Matrix(n, k, result)
        return ret_mat

    def mat_mul_sp(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of secret shared values (n x m)
        rhs: Matrix of public values (m x k)

        Returns: Matrix of secret shared values (n x k)
        """

        n, m = lhs.dimensions()
        m2, k = rhs.dimensions()

        assert m == m2, "Dimensions of matrices do not match"

        LHS = []
        RHS = []

        for i in range(n):
            LHS.extend(lhs.row(i) * k)
            for j in range(k):
                RHS.extend(rhs.col(j))

        ret = self.mul_sp(LHS, RHS)
        slices = [[ret[i + j * m] for j in range(n * k)] for i in range(m)]

        result = slices[0]
        for i in range(1, m):
            result = self.add_ss(result, slices[i])

        ret_mat = Matrix(n, k, result)
        return ret_mat

    def mat_add(self, lhs, rhs):
        return self.binary_wrapper("mat_add", lhs, rhs)
    
    def mat_add_pp(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of public values (n x m)
        rhs: Matrix of public values (n x m)

        Returns: Matrix of public values (n x m)
        """

        n, m = lhs.dimensions()
        n2, m2 = rhs.dimensions()

        assert n == n2 and m == m2, "Dimensions of matrices do not match"

        ret = Matrix(n, m)
        for i in range(n):
            for j in range(m):
                ret[i, j] = (lhs[i, j] + rhs[i, j]) % self.modular

        return ret

    def mat_add_ss(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of secret shared values (n x m)
        rhs: Matrix of secret shared values (n x m)

        Returns: Matrix of secret shared values (n x m)
        """

        n, m = lhs.dimensions()
        n2, m2 = rhs.dimensions()

        assert n == n2 and m == m2, "Dimensions of matrices do not match"

        LHS = lhs.data
        RHS = rhs.data

        ret = self.add_ss(LHS, RHS)

        ret_mat = Matrix(n, m, ret)
        return ret_mat

    def mat_add_sp(self, lhs: Matrix, rhs: Matrix):
        """
        lhs: Matrix of secret shared values (n x m)
        rhs: Matrix of public values (n x m)

        Returns: Matrix of secret shared values (n x m)
        """

        n, m = lhs.dimensions()
        n2, m2 = rhs.dimensions()

        assert n == n2 and m == m2, "Dimensions of matrices do not match"

        LHS = lhs.data
        RHS = rhs.data

        ret = self.add_sp(LHS, RHS)

        ret_mat = Matrix(n, m, ret)
        return ret_mat

    def mat_div_sp(self, lhs: Matrix, rhs: int):
        raise NotImplementedError()  # TODO

    def mat_max_sp(self, lhs: Matrix, rhs: Matrix):
        raise NotImplementedError()  # TODO

    # TODO 四维矩阵或者维度


if __name__ == "__main__":
    import sys

    player_id = int(sys.argv[1])

    protocol = Aby3Protocol(player_id)
    private_data = [123, 234, 345][player_id]

    data0 = protocol.input_share([private_data], 0)
    data1 = protocol.input_share([private_data], 1)
    data2 = protocol.input_share([private_data], 2)

    data0 = protocol.add_ss(data0, data1)
    data0 = protocol.mul_ss(data0, data2)

    data0 = protocol.reveal(data0)
    print(data0[0])

    protocol.player.disconnect()
