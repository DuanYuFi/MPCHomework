import socket
import threading
import time
import sys
import pickle
from hashlib import sha256
import enum

class SIGNAL(enum.Enum):
    TERMINAL = 0x01

class Player:
    parties: list       # parties[0] is party_{player_id - 1 (mod 3)}
                        #, parties[1] is party_{player_id + 1 (mod 3)}
    player_id: int
    num_players: int
    recv_buffers: list

    base_port: int = 34267

    def __init__(self, player_id, num_players):
        self.player_id = player_id
        self.parties = list(range(num_players - 1))
        self.num_players = num_players

        server_thread = threading.Thread(target=self.start_server)
        server_thread.start()

        for i in range(60):
            if self.connect_to_player((player_id + 1) % num_players):
                break
            time.sleep(1)
        else:
            print("Failed to connect after 60 seconds")
            sys.exit(1)
        
        server_thread.join()

        self.log_file = open(f"player_{player_id}.log", "w")
        self.recv_buffers = [b'' for _ in range(num_players - 1)]
        self.locks = [threading.Lock() for _ in range(num_players - 1)]

        self.TERMINAL = pickle.dumps(SIGNAL.TERMINAL)
        self.TERMINAL = int.to_bytes(len(self.TERMINAL), 4, "big") + self.TERMINAL

        self.recv_threads = [
            threading.Thread(target=self._recv_handler, args=(i,)) for i in range(num_players - 1)
        ]

        for thread in self.recv_threads:
            thread.start()
    
    def disconnect(self):
        self.boardcast(SIGNAL.TERMINAL)
        self.log_file.close()
        for thread in self.recv_threads:
            thread.join()

    def handle_client(self, client_socket: socket.socket):
        print(f"Got connection from {client_socket.getpeername()}")
        self.parties[0] = client_socket
    
    def start_server(self, host="127.0.0.1", port=None):
        if port is None:
            port = self.player_id + self.base_port
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen(5)

        print("Waiting for connections...")
        
        client_sock, addr = server.accept()
        self.handle_client(client_sock)
    
    def connect_to_player(self, player_id, player_host="127.0.0.1", player_port=None):
        if player_port is None:
            player_port = player_id + self.base_port
        
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            client.connect((player_host, player_port))
        except ConnectionRefusedError:
            return False
        
        self.parties[1] = client
        return True

    def send(self, data, player_offset):
        # player_offset = -1 means send to previous player
        # player_offset = 1 means send to next player

        data = pickle.dumps(data)
        data = int.to_bytes(len(data), 4, "big") + data
        player_idx = (player_offset + 1) >> 1
        self.parties[player_idx].send(data)
        
    def _recv_handler(self, player_index):
        while True:
            data = self.parties[player_index].recv(1024)
            if not data:
                continue
            
            self.locks[player_index].acquire()
            self.recv_buffers[player_index] += data
            self.locks[player_index].release()

            if data[-len(self.TERMINAL):] == self.TERMINAL:
                print("Recv TERMINAL signal from player", player_index)
                break
        
    def _recvall(self, player_idx, size):
        data = b""
        while len(data) < size:
            self.locks[player_idx].acquire()
            before_len = len(data)
            data += self.recv_buffers[player_idx][:size - len(data)]
            self.recv_buffers[player_idx] = self.recv_buffers[player_idx][len(data) - before_len:]
            self.locks[player_idx].release()
        return data

    def recv(self, player_offset):
        # player_offset = -1 means recv from previous player
        # player_offset = 1 means recv from next player

        player_idx = (player_offset + 1) >> 1
        size = int.from_bytes(self._recvall(player_idx, 4), "big")
        data = self._recvall(player_idx, size)
        data = pickle.loads(data)
        return data

    def pass_around(self, data, offset=1):
        self.send(data, offset)
        return self.recv(-offset)

    def boardcast(self, data):
        self.send(data, 1)
        self.send(data, -1)

if __name__ == "__main__":

    from hashlib import sha256
    from os import urandom

    player_id = int(sys.argv[1])
    num_players = 3
    player = Player(player_id, num_players)
    if player_id == 0:
        data = urandom(5000000)
        print(sha256(data).hexdigest())
        player.send(data, 0)
        player.send(data, 1)
    
    else:
        data = player.recv(player_id - 1)
        print(sha256(data).hexdigest())
    
    player.disconnect()
