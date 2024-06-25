import random

from network import Player
from OT import OT3


def arithmetic_shift_right(x, offset, nbits=64):
    """
    Performs an arithmetic right shift on the given value `x` by the specified `offset` number of bits.

    Args:
        x (int): The value to be shifted.
        offset (int): The number of bits to shift `x` to the right.
        nbits (int, optional): The total number of bits in `x`. Defaults to 64.

    Returns:
        int: The result of the arithmetic right shift operation.
    """
    if x >> (nbits - 1):
        return (x >> offset) + ((2**nbits - 1) << (nbits - offset))
    else:
        return x >> offset


class RSS3PC:
    data: list
    decimal: int
    modular: int
    binary: bool

    def __init__(self, s1, s2, modular=64, decimal=0, binary=False):
        """
        Initializes an instance of the replicated secret sharing (3 party) class.

        Args:
            s1 (int): The first share slice.
            s2 (int): The second share slice.
            modular (int, optional): The modular value `k` of Z2k. Defaults to 64.
            decimal (int, optional): The decimal bit numbers. Defaults to 0.
            binary (bool, optional): Specifies whether the share is in binary. Defaults to False.
        """
        self.data = [s1, s2]
        self.decimal = decimal
        self.modular = modular
        self.binary = binary

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __str__(self):
        return f"({self.data[0]}, {self.data[1]})"

    def __repr__(self):
        return f"({self.data[0]}, {self.data[1]})"

    def set_decimal(self, decimal):
        """
        Sets the decimal bit numbers.

        Args:
            decimal (int): The decimal bit numbers.
        """
        self.decimal = decimal


class Matrix:
    data: list
    nrows: int
    ncols: int

    def __init__(self, n, m, data=None):
        """
        Initializes an instance of the Matrix class.

        Args:
            n (int): The number of rows.
            m (int): The number of columns.
            data (list, optional): The initial data. Defaults to None.

        Attributes:
            data (list): The data stored in the instance.
            nrows (int): The number of rows.
            ncols (int): The number of columns.
        """
        if data is None:
            self.data = [0] * (n * m)
        else:
            self.data = data
        self.nrows = n
        self.ncols = m

    def row(self, index):
        """
        Returns a row from the data matrix.

        Args:
            index (int): The index of the row to retrieve.

        Returns:
            list: The row of data.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= self.nrows:
            raise IndexError("Index out of range")
        return self.data[index * self.ncols : (index + 1) * self.ncols]

    def col(self, index):
        """
        Returns a list containing the elements in the specified column.

        Args:
            index (int): The index of the column.

        Returns:
            list: A list containing the elements in the specified column.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= self.ncols:
            raise IndexError("Index out of range")
        return [self.data[i * self.ncols + index] for i in range(self.nrows)]

    def dimensions(self):
        """
        Returns the dimensions of the object.

        Returns:
            tuple: A tuple containing the number of rows and columns.
        """
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
        """
        Transposes the matrix by swapping rows with columns.

        Returns:
            Matrix: The transposed matrix.
        """
        data = []
        for j in range(self.ncols):
            data.extend(self.col(j))
        return Matrix(self.ncols, self.nrows, data)

    def __str__(self):
        each_width = [
            max(len(str(each)) for each in self.col(i)) + 1 for i in range(self.ncols)
        ]
        ret = ""
        for i in range(self.nrows):
            ret += (
                "["
                + " ".join(
                    [str(self[i, j]).rjust(each_width[j]) for j in range(self.ncols)]
                )
                + "]\n"
            )
        return ret

    def __repr__(self):
        return self.__str__()


class Aby3Protocol:

    player: Player
    PRNGs: list
    modular: int
    ot_for_bi: OT3  # ot for bit injection in ABY3

    def __init__(
        self, player_id, modular_bit=64, demical_bit=20, port_base=None, debug=False
    ):
        """
        Initializes an instance of the ABY3Protocol class.

        Args:
            player_id (int): The ID of the player.
            modular_bit (int, optional): The number of bits for modular arithmetic. Defaults to 64.
            demical_bit (int, optional): The number of bits for decimal arithmetic. Defaults to 20.
            port_base (int, optional): The base port number for communication. Defaults to None.
            debug (bool, optional): Flag indicating whether to enable verbose output mode. Defaults to False.
        """
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

        self.ot_for_bi = OT3(2, 1, 0, player_id, self)

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

        if isinstance(shares[0], int):
            return False, False

        elif isinstance(shares[0], RSS3PC):
            if shares[0].decimal > 0:
                return True, True
            else:
                return True, False

        elif isinstance(shares[0], float):
            return False, True
        else:
            raise ValueError(f"Invalid type {type(shares[0])} for shares")

    def input_share(self, public, owner: int, binary=False):
        """
        Wrapper function to share the public value (known to one player) to 3 players.

        Args:
            public (list of numbers, or Matrix): public values
            owner (int): the player who knows the public value
            binary (bool, optional): whether use binary secret sharing. Defaults to False.
        
        Returns:
            list of RSS3PC: secret shared values
        """

        if isinstance(public, Matrix):
            return Matrix(
                public.nrows, public.ncols, self.input_share(public.data, owner)
            )
        elif binary:
            return self.input_share_b(public, owner)
        elif all(isinstance(each_public, int) for each_public in public):
            return self.input_share_i(public, owner)
        elif any(isinstance(each_public, float) for each_public in public):
            return self.input_share_f(public, owner)
        else:
            raise ValueError(f"Invalid type {type(public[0])} for public value")

    def input_share_b(self, public: list, owner: int):
        """
        Share the public value in binary secret sharing.

        Args:
            public (list): public value
            owner (int): the player who knows the public value
        
        Returns:
            list of RSS3PC: secret shared values
        """

        ret = [
            RSS3PC(0, 0, modular=self.modular_bit, binary=True)
            for _ in range(len(public))
        ]

        if owner == self.player.player_id:
            send_buffer = []
            for i in range(len(public)):
                r = self.PRNGs[0].randrange(self.modular)
                ret[i][0] = r
                ret[i][1] = public[i] ^ r
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

    def input_share_i(self, public: list, owner: int):
        """
        Share the public value in arithmetic secret sharing.

        Args:
            public (list): public value
            owner (int): the player who knows the public value
        
        Returns:
            list of RSS3PC: secret shared values
        """

        ret = [RSS3PC(0, 0, modular=self.modular_bit) for _ in range(len(public))]

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

    def input_share_f(self, public: list, owner: int):
        """
        Share the public value in fixed point secret sharing.

        Args:
            public (list): public value
            owner (int): the player who knows the public value
        
        Returns:
            list of RSS3PC: secret shared values
        """

        public = [round(each * 2**self.demical) for each in public]
        ret = self.input_share_i(public, owner)
        for i in range(len(ret)):
            ret[i].set_decimal(self.demical)
        return ret

    def random_shares(self, length, binary=False):
        """
        Generate random secret shared values.

        Args:
            length (int): length of the secret shared values
            binary (bool, optional): whether use binary secret sharing. Defaults to False.
        
        Returns:
            list of RSS3PC: secret shared values
        """

        ret = [
            RSS3PC(0, 0, modular=self.modular_bit, binary=binary) for _ in range(length)
        ]
        for i in range(length):
            ret[i][0] = self.PRNGs[0].randrange(self.modular)
            ret[i][1] = self.PRNGs[1].randrange(self.modular)

        return ret

    def zero_shares(self, length, binary=False):
        """
        Generate zero secret shared values.

        Args:
            length (int): length of the secret shared values
            binary (bool, optional): whether use binary secret sharing. Defaults to False.
        
        Returns:
            list of RSS3PC: secret shared values with value = 0.
        """

        ret = [
            RSS3PC(0, 0, modular=self.modular_bit, binary=binary) for _ in range(length)
        ]
        buffer = []
        for i in range(length):
            ret[i][0] = self.PRNGs[0].randrange(self.modular)
            if binary:
                ret[i][0] ^= self.PRNGs[1].randrange(self.modular)
            else:
                ret[i][0] = (
                    ret[i][0] - self.PRNGs[1].randrange(self.modular)
                ) % self.modular

            buffer.append(ret[i][0])

        recv = self.player.pass_around(buffer, -1)

        for i in range(length):
            ret[i][1] = recv[i]

        return ret

    def reveal(self, secret, to=None, sign=True):
        """
        Wrapper function to reveal secret shared values.

        Args:
            secret (list or Matrix): secret shared values
            to (int, optional): the player to reveal the secret. Defaults to None, means reveal to all players.
            sign (bool, optional): whether to reveal the secret with sign. Defaults to True.
        """

        if isinstance(secret, Matrix):
            return Matrix(secret.nrows, secret.ncols, self.reveal(secret.data, to))

        if secret[0].binary:
            return self.reveal_b(secret, to, sign)

        if secret[0].decimal > 0:
            return self.reveal_f(secret, to, sign)
        else:
            return self.reveal_i(secret, to, sign)

    def reveal_b(self, secret: list, to, sign):
        """
        Reveal the binary secret shared values.

        Args:
            secret (list): binary secret shared values
            to (int): the player to reveal the secret
            sign (bool): whether to reveal the secret with sign
        
        Returns:
            list: revealed values
        """

        if to is None:
            ret = [0 for _ in range(len(secret))]
            send_buffer = []

            for i in range(len(secret)):
                ret[i] = secret[i][0] ^ secret[i][1]
                send_buffer.append(secret[i][0])

            recv_data = self.player.pass_around(send_buffer, 1)
            for i in range(len(secret)):
                ret[i] = ret[i] ^ recv_data[i]

            if sign:
                ret = [
                    each if each < self.modular // 2 else each - self.modular
                    for each in ret
                ]
            if secret[0].decimal > 0:
                for i in range(len(ret)):
                    ret[i] /= 2 ** secret[i].decimal

            return ret
        else:
            if to == (self.player_id + 1) % 3:
                send_buffer = [secret[i][0] for i in range(len(secret))]
                self.player.send(send_buffer, 1)
            elif to == self.player_id:
                recv = self.player.recv(-1)
                ret = [
                    secret[i][0] ^ secret[i][1] ^ recv[i] for i in range(len(secret))
                ]

                if sign:
                    ret = [
                        each if each < self.modular // 2 else each - self.modular
                        for each in ret
                    ]
                if secret[0].decimal > 0:
                    for i in range(len(ret)):
                        ret[i] /= 2 ** secret[i].decimal
                return ret

    def reveal_i(self, secret: list, to, sign):
        """
        Reveal the arithmetic secret shared values.

        Args:
            secret (list): arithmetic secret shared values
            to (int): the player to reveal the secret
            sign (bool): whether to reveal the secret with sign
        
        Returns:
            list: revealed values
        """

        if to is None:
            ret = [0 for _ in range(len(secret))]
            send_buffer = []

            for i in range(len(secret)):
                ret[i] = (secret[i][0] + secret[i][1]) % self.modular
                send_buffer.append(secret[i][0])

            recv_data = self.player.pass_around(send_buffer, 1)
            for i in range(len(secret)):
                ret[i] = (ret[i] + recv_data[i]) % self.modular

            if sign:
                ret = [
                    each if each < self.modular // 2 else each - self.modular
                    for each in ret
                ]
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
                if sign:
                    ret = [
                        each if each < self.modular // 2 else each - self.modular
                        for each in ret
                    ]
                return ret

    def reveal_f(self, secret: list, to, sign):
        """
        Reveal the arithmetic fixed-point secret shared values.

        Args:
            secret (list): arithmetic fixed-point secret shared values
            to (int): the player to reveal the secret
            sign (bool): whether to reveal the secret with sign
        
        Returns:
            list: revealed values
        """

        ret = self.reveal_i(secret, to, sign)
        if to is None or to == self.player_id:
            for i in range(len(ret)):
                ret[i] /= 2 ** secret[i].decimal
            return ret

    def shift_left(self, shares: list, shift: int):
        """
        Shift the secret shared values to the left.

        Args:
            shares (list): secret shared values
            shift (int): the number of bits to shift
        
        Returns:
            list: shifted secret shared values
        """
        return self.mul_sp(shares, [2**shift] * len(shares))

    def shift_right(self, shares: list, shift: int):
        """
        Shift the secret shared values to the right. A.K.A. truncation.
        Reference: ABY3: A Mixed Protocol Framework for Machine Learning, 
            Section 5.1.2, protocol trunc1. (Probalistic truncation)
        
        Args:
            shares (list): secret shared values
            shift (int): the number of bits to shift
        
        Returns:
            list: shifted secret shared values
        """

        mod_bit = shares[0].modular
        modular = 1 << mod_bit

        ret = [RSS3PC(0, 0, modular=mod_bit) for _ in range(len(shares))]
        if self.player_id == 0:
            for i in range(len(shares)):
                ret[i][0] = arithmetic_shift_right(shares[i][0], shift, mod_bit)

            recv_data = self.player.recv(1)
            for i in range(len(shares)):
                ret[i][1] = recv_data[i]

        elif self.player_id == 1:
            rs = [self.PRNGs[1].randrange(modular) for _ in range(len(shares))]
            send_buffer = []
            for i in range(len(shares)):
                ret[i][0] = arithmetic_shift_right(
                    (shares[i][0] + shares[i][1]) % modular, shift, mod_bit
                )
                ret[i][0] = (ret[i][0] - rs[i]) % modular
                send_buffer.append(ret[i][0])

                ret[i][1] = rs[i]

            self.player.send(send_buffer, -1)

        else:
            rs = [self.PRNGs[0].randrange(modular) for _ in range(len(shares))]
            for i in range(len(shares)):
                ret[i][0] = rs[i]
                ret[i][1] = arithmetic_shift_right(shares[i][1], shift, mod_bit)

        return ret

    def i2f(self, shares):
        """
        Convert integer secret shared values to fixed-point secret shared values.

        Args:
            shares (list or Matrix): integer secret shared values
        
        Returns:
            list or Matrix: fixed-point secret shared values
        """
        if isinstance(shares, Matrix):
            return Matrix(shares.nrows, shares.ncols, self.i2f(shares.data))

        if isinstance(shares[0], int):
            return [float(each) for each in shares]

        shares = self.shift_left(shares, self.demical)
        for i in range(len(shares)):
            shares[i].set_decimal(self.demical)

        return shares

    def f2i(self, shares: list):
        """
        Convert fixed-point secret shared values to integer secret shared values.

        Args:
            shares (list or Matrix): fixed-point secret shared values
        
        Returns:
            list or Matrix: integer secret shared values
        """
        if isinstance(shares, Matrix):
            return Matrix(shares.nrows, shares.ncols, self.f2i(shares.data))

        if isinstance(shares[0], float):
            return [int(each) for each in shares]

        shares = self.shift_right(shares, self.demical)
        for i in range(len(shares)):
            shares[i].set_decimal(0)

        return shares

    def binary_wrapper(self, operation: str, lhs, rhs):
        """
        Wrapper for all kinds (16) of binary operation.

        Reference: SecretFlow-SPU, Type System
            https://www.secretflow.org.cn/zh-CN/docs/spu/0.9.1b0/development/type_system
        """

        l_type = self.get_type(lhs)
        r_type = self.get_type(rhs)

        binop_pp = getattr(self, operation + "_pp")
        binop_ss = getattr(self, operation + "_ss")
        binop_sp = getattr(self, operation + "_sp")

        if (not l_type[0]) and (not r_type[0]):
            return binop_pp(lhs, rhs)  # public and public

        if (not l_type[1]) and (not r_type[1]):  # integer and integer
            if l_type[0] and r_type[0]:
                return binop_ss(lhs, rhs)
            else:
                return binop_sp(lhs, rhs)

        elif l_type[1] != r_type[1]:  # fixed point and integer
            if not l_type[1]:
                lhs = self.i2f(lhs)
            else:
                rhs = self.i2f(rhs)

        # now both are fixed point numbers

        if l_type[0] != r_type[0]:
            ret = binop_sp(lhs, rhs)

        else:
            ret = binop_ss(lhs, rhs)

        if operation == "mul":
            ret = self.shift_right(ret, self.demical)

        elif operation == "mat_mul":
            ret = Matrix(ret.nrows, ret.ncols, self.shift_right(ret.data, self.demical))
            for i in range(len(ret.data)):
                ret.data[i].set_decimal(self.demical)

            return ret

        for i in range(len(ret)):
            ret[i].set_decimal(self.demical)
        return ret

    def add(self, lhs, rhs):
        """
        Wrapper for addition operation.
        """
        return self.binary_wrapper("add", lhs, rhs)

    def add_pp(self, lhs, rhs):
        """
        Add two public values
        """

        return [lhs[i] + rhs[i] for i in range(len(lhs))]

    def add_ss(self, lhs: list, rhs: list):
        """
        Add two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        assert lhs[0].modular == rhs[0].modular, "Modulars of lhs and rhs must be equal"

        mod_bit = lhs[0].modular
        modular = 1 << mod_bit

        return [
            RSS3PC(
                (_lhs[0] + _rhs[0]) % modular,
                (_lhs[1] + _rhs[1]) % modular,
                modular=mod_bit,
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

        mod_bit = lhs[0].modular
        modular = 1 << mod_bit

        if isinstance(rhs[0], float):
            rhs = [round(each * 2**self.demical) for each in rhs]

        ret = [RSS3PC(lhs[i][0], lhs[i][1], modular=mod_bit) for i in range(len(lhs))]
        if self.player_id == 0:
            for i in range(len(lhs)):
                ret[i][1] = (ret[i][1] + rhs[i]) % modular
        elif self.player_id == 1:
            for i in range(len(lhs)):
                ret[i][0] = (ret[i][0] + rhs[i]) % modular

        return ret

    def neg(self, lhs):
        """
        Wrapper for negation operation.

        neg(x) = mul_sp(x, -1)
        """
        if isinstance(lhs, Matrix):
            return Matrix(lhs.nrows, lhs.ncols, self.neg(lhs.data))

        return self.mul(lhs, [-1] * len(lhs))

    def sub(self, lhs, rhs):
        """
        Wrapper for subtraction operation.

        sub(x, y) = add(x, neg(y))
        """
        return self.add(lhs, self.neg(rhs))

    def sub_pp(self, lhs, rhs):
        """
        Subtract two public values
        """

        return [lhs[i] - rhs[i] for i in range(len(lhs))]

    def mul(self, lhs, rhs):
        """
        Wrapper for multiplication operation.
        """
        return self.binary_wrapper("mul", lhs, rhs)

    def mul_pp(self, lhs, rhs):
        """
        Multiply two public values
        """

        return [lhs[i] * rhs[i] for i in range(len(lhs))]

    def mul_ss(self, lhs, rhs):
        """
        Multiply two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        assert lhs[0].modular == rhs[0].modular, "Modulars of lhs and rhs must be equal"

        mod_bit = lhs[0].modular
        modular = 1 << mod_bit

        ret = [RSS3PC(0, 0, modular=mod_bit) for _ in lhs]
        send_buffer = []

        for i in range(len(lhs)):
            ret[i][0] = (
                lhs[i][0] * rhs[i][0]
                + lhs[i][1] * rhs[i][0]
                + lhs[i][0] * rhs[i][1]
                + self.PRNGs[0].randrange(modular)
                - self.PRNGs[1].randrange(modular)
            ) % modular

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

        mod_bit = lhs[0].modular
        modular = 1 << mod_bit

        if isinstance(rhs[0], float):
            rhs = [round(each * 2**self.demical) for each in rhs]

        return [
            RSS3PC(
                (lhs[i][0] * rhs[i]) % modular,
                (lhs[i][1] * rhs[i]) % modular,
                modular=mod_bit,
            )
            for i in range(len(lhs))
        ]

    def mat_mul(self, lhs, rhs):
        """
        Wrapper for matrix multiplication operation.
        """
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
                ret[i, j] = sum(lhs[i, x] * rhs[x, j] for x in range(m))

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

        if isinstance(rhs.data[0], RSS3PC):
            lhs, rhs = rhs, lhs

        if isinstance(rhs.data[0], float):
            rhs = Matrix(
                rhs.nrows,
                rhs.ncols,
                [round(each * 2**self.demical) for each in rhs.data],
            )

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
        """
        Wrapper for matrix addition operation.
        """
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

    def xor_gate(self, lhs: list, rhs: list):
        """
        XOR operation for secret shared values.

        Args:
            lhs (list): binary secret shared values
            rhs (list): binary secret shared values
        
        Returns:
            list: binary secret shared values
        """


        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        return [
            RSS3PC(
                lhs[i][0] ^ rhs[i][0],
                lhs[i][1] ^ rhs[i][1],
                modular=lhs[0].modular,
                binary=True,
            )
            for i in range(len(lhs))
        ]

    def and_gate(self, lhs: list, rhs: list):
        """
        AND operation for secret shared values.

        Args:
            lhs (list): binary secret shared values
            rhs (list): binary secret shared values
        
        Returns:
            list: binary secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        assert lhs[0].modular == rhs[0].modular, "Modulars of lhs and rhs must be equal"

        mod_bit = lhs[0].modular
        ret = [RSS3PC(0, 0, modular=mod_bit, binary=True) for _ in lhs]

        send_buffer = []

        for i in range(len(lhs)):
            ret[i][0] = (
                (lhs[i][0] & rhs[i][0])
                ^ (lhs[i][1] & rhs[i][0])
                ^ (lhs[i][0] & rhs[i][1])
                ^ self.PRNGs[0].randrange(mod_bit)
                ^ self.PRNGs[1].randrange(mod_bit)
            )

            send_buffer.append(ret[i][0])

        recv_data = self.player.pass_around(send_buffer, -1)

        for i in range(len(lhs)):
            ret[i][1] = recv_data[i]

        return ret

    def or_gate(self, lhs: list, rhs: list):
        """
        OR operation for secret shared values.

        Args:
            lhs (list): binary secret shared value
            rhs (list): binary secret shared value
        
        Returns:
            list: binary secret shared value
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        assert lhs[0].modular == rhs[0].modular, "Modulars of lhs and rhs must be equal"

        return self.xor_gate(self.and_gate(lhs, rhs), self.xor_gate(lhs, rhs))

    def adder(self, a, b, c):
        """
        Performs addition of three inputs using bitwise operations, in MPC way.

        Args:
            a (list): The first input bit.
            b (list): The second input bit.
            c (list): The carry input bit.

        Returns:
            tuple: A tuple containing the sum bit and the carry bit.
        """
        sum = self.xor_gate(self.xor_gate(a, b), c)
        carry = self.or_gate(self.and_gate(a, b), self.and_gate(c, self.xor_gate(a, b)))
        return sum, carry

    def full_adder(self, a, b, bit_length):
        """
        Performs full addition of two n-bit numbers in MPC way.

        Args:
            a (list): The first n-bit number.
            b (list): The second n-bit number.
            bit_length (int): The number of bits in the numbers.
        
        Returns:
            list: The n bit sum of the two numbers.
        """

        sum = []
        carry = [RSS3PC(0, 0, modular=a[0].modular)]
        for i in range(bit_length):
            sum_bit, carry = self.adder([a[i]], [b[i]], carry)
            sum.append(sum_bit[0])
        sum.append(carry[0])
        return sum

    def bit_decomposition(self, a: list):
        """
        Decompose the secret shared value into bits, 
            or convert arithmetic secret shared value to binary secret shared value.
        
        Reference: ABY3: A Mixed Protocol Framework for Machine Learning,
            Section 5.3 (Bit Decomposition). 
            Slower implementation because no PPA.
        
        Args:
            a (list): arithmetic secret shared value
        
        Returns:
            list: list of binary secret shared value
        """

        nbits = a[0].modular
        decimal = a[0].decimal
        a0, a1, a2 = [], [], []

        if self.player_id == 0:
            for each in a:
                a0.append(RSS3PC(each[0], 0, each.modular, True))
                a1.append(RSS3PC(0, each[1], each.modular, True))
                a2.append(RSS3PC(0, 0, each.modular, True))
        elif self.player_id == 1:
            for each in a:
                a0.append(RSS3PC(0, 0, each.modular, True))
                a1.append(RSS3PC(each[0], 0, each.modular, True))
                a2.append(RSS3PC(0, each[1], each.modular, True))
        else:
            for each in a:
                a0.append(RSS3PC(0, each[1], each.modular, True))
                a1.append(RSS3PC(0, 0, each.modular, True))
                a2.append(RSS3PC(each[0], 0, each.modular, True))

        a0_bits = []
        a1_bits = []
        a2_bits = []

        batch_size = 1024
        a0_slices = [a0[i : i + batch_size] for i in range(0, len(a0), batch_size)]
        a1_slices = [a1[i : i + batch_size] for i in range(0, len(a1), batch_size)]
        a2_slices = [a2[i : i + batch_size] for i in range(0, len(a2), batch_size)]

        for a0, a1, a2 in zip(a0_slices, a1_slices, a2_slices):
            for i in range(nbits):
                a0_bits.append(
                    RSS3PC(
                        sum([((a0[j][0] >> i) & 1) * 2**j for j in range(len(a0))]),
                        sum([((a0[j][1] >> i) & 1) * 2**j for j in range(len(a0))]),
                        modular=len(a0),
                    )
                )
                a1_bits.append(
                    RSS3PC(
                        sum([((a1[j][0] >> i) & 1) * 2**j for j in range(len(a1))]),
                        sum([((a1[j][1] >> i) & 1) * 2**j for j in range(len(a1))]),
                        modular=len(a0),
                    )
                )
                a2_bits.append(
                    RSS3PC(
                        sum([((a2[j][0] >> i) & 1) * 2**j for j in range(len(a2))]),
                        sum([((a2[j][1] >> i) & 1) * 2**j for j in range(len(a2))]),
                        modular=len(a0),
                    )
                )

        result = self.full_adder(a0_bits, a1_bits, nbits)
        result = self.full_adder(result, a2_bits, nbits)

        ret = []
        for a0, a1, a2 in zip(a0_slices, a1_slices, a2_slices):
            for i in range(len(a0)):
                ret.append(
                    RSS3PC(
                        sum(
                            [
                                ((result[j][0] >> i) & 1) * 2**j
                                for j in range(len(result))
                            ]
                        ),
                        sum(
                            [
                                ((result[j][1] >> i) & 1) * 2**j
                                for j in range(len(result))
                            ]
                        ),
                        modular=nbits,
                        decimal=decimal,
                        binary=True,
                    )
                )

        return ret

    def bit_injection(self, bits: list):
        """
        Convert a list of single bit binary ss into a list of arithmetic share.

        Reference: ABY3: A Mixed Protocol Framework for Machine Learning,
            Section 5.4.1, Computing a[[b]]^B = [[ab]]^A by 3-party OT.

        Args:
             bits (list): list of RSS3PC with binary=True, with value in {0, 1}
           
        Returns: 
            list: list of RSS3PC with binary=False, with value corresponding to bits
        """

        ret = [RSS3PC(0, 0, modular=self.modular_bit, binary=False) for _ in bits]

        if self.player_id == 0:
            c1 = [self.PRNGs[0].randrange(self.modular) for _ in bits]
            choices = [bit[1] for bit in bits]
            self.ot_for_bi.choice(choices)
            c2 = self.player.recv(1)

            for i in range(len(bits)):
                ret[i][0] = c1[i]
                ret[i][1] = c2[i]

        elif self.player_id == 1:
            c3 = [self.PRNGs[1].randrange(self.modular) for _ in bits]
            c2 = self.ot_for_bi.receive()
            self.player.send(c2, -1)

            for i in range(len(bits)):
                ret[i][0] = c2[i]
                ret[i][1] = c3[i]

        else:
            c1 = [self.PRNGs[1].randrange(self.modular) for _ in bits]
            c3 = [self.PRNGs[0].randrange(self.modular) for _ in bits]

            M1 = []
            M2 = []

            for i in range(len(bits)):
                m1 = ((0 ^ bits[i][0] ^ bits[i][1]) - c1[i] - c3[i]) % self.modular
                m2 = ((1 ^ bits[i][0] ^ bits[i][1]) - c1[i] - c3[i]) % self.modular
                M1.append(m1)
                M2.append(m2)

            self.ot_for_bi.send(M1, M2)

            for i in range(len(bits)):
                ret[i][0] = c3[i]
                ret[i][1] = c1[i]

        return ret

    def ltz(self, values: list):
        """
        Return 1 if values < 0, 0 otherwise, in arithmetic ss.

        ltz(x) = msb(x) = bit_injection(bit_decomposition(x)[0])
        """

        if type(values[0]) in [int, float]:
            return [1 if each < 0 else 0 for each in values]

        mod_bit = values[0].modular
        decompositions = self.bit_decomposition(values)
        msb = []
        for each in decompositions:
            msb.append(
                RSS3PC((each[0] >> (mod_bit - 1)) & 1, (each[1] >> (mod_bit - 1)) & 1)
            )

        msb = self.bit_injection(msb)
        return msb

    def compare(self, lhs: list, rhs: list):
        """
        return 1 if lhs < rhs, 0 otherwise, in arithmetic ss.

        compare(x, y) = ltz(sub(x, y))
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"
        return self.ltz(self.sub(lhs, rhs))

    def max_sp(self, lhs, rhs):
        """
        Return the maximum of secret shared value and public value.

        max_sp(x, y) = x + (y - x) * compare(x, y)
        """

        smaller = self.compare(lhs, rhs)
        return self.add(self.mul(smaller, self.sub(rhs, lhs)), lhs)

    def div_sp(self, lhs: list, rhs: int):
        """
        Tricky implementation
        TODO
        """

        result = self.reveal(lhs)
        result = [each / rhs for each in result]
        return self.input_share(result, 0)

    def mat_max_sp(self, lhs, rhs: int):
        """
        Return the maximum of matrix and public value.
        """
        return Matrix(
            lhs.nrows, lhs.ncols, self.max_sp(lhs.data, [rhs] * lhs.nrows * lhs.ncols)
        )

    def mat_ltz(self, lhs: Matrix):
        """
        Return 1 if matrix[i, j] < 0, 0 otherwise, in arithmetic ss.
        """
        return Matrix(lhs.nrows, lhs.ncols, self.ltz(lhs.data))


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
