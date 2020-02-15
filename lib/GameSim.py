import numpy as np 

class GameSim():
    def __init__(self, shape=(12, 15)):
        self.base = np.zeros(shape)
        self.players = np.zeros(shape)
        self.dimonds = np.zeros(shape)
        pass

    def get_state(self, id):
        pass

    def move(self, id, action):
        pass
