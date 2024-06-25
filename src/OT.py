
class OT3:
    """
    A simple three party OT protocol.

    Reference: ABY3: A Mixed Protocol Framework for Machine Learning,
        Section 5.4.1, Three-Party OT.
    """

    sender: int
    receiver: int
    helper: int
    role: int

    def __init__(self, sender: int, receiver: int, helper: int, role: int, protocol):
        self.protocol = protocol
        self.sender = sender
        self.receiver = receiver
        self.helper = helper
        self.role = role

        if role == sender:
            offset = helper - role
            if offset == 2:
                offset = -1
            elif offset == -2:
                offset = 1
            idx = (offset + 1) // 2
            self.shared_PRNG = self.protocol.PRNGs[idx]
        elif role == helper:
            offset = sender - role
            if offset == 2:
                offset = -1
            elif offset == -2:
                offset = 1
            idx = (offset + 1) // 2
            self.shared_PRNG = self.protocol.PRNGs[idx]
    
    def send(self, msg1: list, msg2: list):
        if self.role == self.sender:
            masked1 = [(each_msg1 * 2 ** 64) ^ self.shared_PRNG.getrandbits(self.protocol.modular_bit + 64) for each_msg1 in msg1]
            masked2 = [(each_msg2 * 2 ** 64) ^ self.shared_PRNG.getrandbits(self.protocol.modular_bit + 64) for each_msg2 in msg2]
            offset = self.receiver - self.sender
            self.protocol.player.send((masked1, masked2), offset)

        else:
            return None
        
    def choice(self, choice_bits: list):
        if self.role == self.helper:
            mask1 = [self.shared_PRNG.getrandbits(self.protocol.modular_bit + 64) for _ in range(len(choice_bits))]
            mask2 = [self.shared_PRNG.getrandbits(self.protocol.modular_bit + 64) for _ in range(len(choice_bits))]
            choices = [mask1[i] if choice_bits[i] == 0 else mask2[i] for i in range(len(choice_bits))]
            offset = self.receiver - self.helper
            self.protocol.player.send(choices, offset)
        
        else:
            return None
        
    def receive(self):
        if self.role == self.receiver:
            choices = self.protocol.player.recv(self.helper - self.receiver)
            masked1, masked2 = self.protocol.player.recv(self.sender - self.receiver)

            receives = []
            for choice, c1, c2 in zip(choices, masked1, masked2):
                if (c1 ^ choice) & 0xffffffffffffffff == 0:
                    receives.append((c1 ^ choice) >> 64)
                elif (c2 ^ choice) & 0xffffffffffffffff == 0:
                    receives.append((c2 ^ choice) >> 64)
                else:
                    raise ValueError("Internal error in OT3.receive()")
            
            return receives
        
        else:
            return None

