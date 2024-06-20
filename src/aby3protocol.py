import random

from network import Player

class RSS3PC:
    data: list

    def __init__(self, s1, s2):
        self.data = [s1, s2]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __str__(self):
        return f"({self.data[0]}, {self.data[1]})"

    def __repr__(self):
        return f"({self.data[0]}, {self.data[1]})"

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
        return self.data[index * self.ncols: (index + 1) * self.ncols]
    
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
            self.data[index * self.ncols: (index + 1) * self.ncols] = value
    

class Aby3Protocol:
    """
    ABY3 protocol
    """
    player: Player
    PRNGs: list
    modular: int

    def __init__(self, player_id, modular=256, port_base=None):
        self.player = Player(player_id, 3, port_base=port_base)
        seed = random.getrandbits(32)
        self.PRNGs = [None, None]
        self.PRNGs[1] = random.Random(seed)
        seed = self.player.pass_around(seed, 1)
        self.PRNGs[0] = random.Random(seed)

        self.modular = modular
        self.player_id = player_id

    def disconnect(self):
        self.player.disconnect()

    def input_share(self, public: list, owner: int):
        ret = [RSS3PC(0, 0) for _ in range(len(public))]
        
        if owner == self.player.player_id:
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

    def reveal(self, secret: list, to=None):
        """
        Reveal secret shared values
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
            
            return ret
        else:
            if to == (self.player_id + 1) % 3:
                send_buffer = [secret[i][0] for i in range(len(secret))]
                self.player.send(send_buffer, 1)
            elif to == self.player_id:
                recv = self.player.recv(-1)
                ret = [(secret[i][0] + secret[i][1] + recv[i]) % self.modular for i in range(len(secret))]
                return ret

    
    def add_ss(self, lhs: list, rhs: list):
        """
        Add two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"

        return [RSS3PC((_lhs[0] + _rhs[0]) % self.modular, (_lhs[1] + _rhs[1]) % self.modular)\
                for _lhs, _rhs in zip(lhs, rhs)]

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
    
    def mul_ss(self, lhs, rhs):
        """
        Multiply two secret shared values
        """

        assert len(lhs) == len(rhs), "Lengths of lhs and rhs must be equal"

        ret = [RSS3PC(0, 0) for _ in lhs]
        send_buffer = []

        for i in range(len(lhs)):
            ret[i][0] = (lhs[i][0] * rhs[i][0]\
                       + lhs[i][1] * rhs[i][0]\
                       + lhs[i][0] * rhs[i][1]\
                       + self.PRNGs[0].randrange(self.modular)\
                       - self.PRNGs[1].randrange(self.modular))\
                       % self.modular
            
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

        return [RSS3PC((lhs[i][0] * rhs[i]) % self.modular, (lhs[i][1] * rhs[i]) % self.modular)\
                for i in range(len(lhs))]
    
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