import numpy as torch 
import torch
from random import randint
import matplotlib.pyplot as plt

from .Data import Data
class GameSim():
    symbols = {
        0: "x ",
        1: "# ",
        2: "$ ",
        3: "@ ",
        "separator": ". "
    }
    actions = {
        0: lambda x, y: (x, y-1),
        1: lambda x, y: (x+1, y),
        2: lambda x, y: (x, y+1),
        3: lambda x, y: (x-1, y), 
    }

    def __init__(self, shape=(15, 12), 
                       max_cap=5, 
                       nb_plyers=1, 
                       nb_diamoinds=20, 
                       nb_portals=2, 
                       action_space=4, 
                       max_ticks=600, 
                       save_image=False, 
                       device="cuda"):
        self.device = torch.device(device)
        self.max_cap = max_cap
        self.id_to_pos = {}
        self.id_to_base = {}
        self.id_to_point = torch.zeros(nb_plyers, dtype=torch.long, device=self.device, requires_grad=False)
        self.portal_to = {}
        self.cached_picks = set()
        self.cached_diamonds = set()
        self.shape = shape
        self.bag = torch.zeros(shape, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.base = torch.zeros(shape, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.players = torch.zeros(shape, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.diamonds = torch.zeros(shape, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.portals = torch.zeros(shape, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.game_ticks = 0
        self.max_ticks = max_ticks
        self.nb_plyers = nb_plyers
        self.nb_diamoinds = nb_diamoinds
        self.nb_portals = nb_portals
        self.action_space = action_space
        self.save_image = save_image

    def __repr__(self):
        stack = [self.players, self.base, self.diamonds, self.portals]
        rep = ""
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                for i, p in enumerate(stack):
                    if p[x,y] != 0:
                        rep += GameSim.symbols[i]
                        break
                else:
                    rep += GameSim.symbols["separator"]
            rep += "\n"
        return rep

    def get_players(self):
        return self.id_to_pos.keys()

    def clear_obj(self):
        self.bag.fill_(0)
        self.base.fill_(0)
        self.players.fill_(0)
        self.diamonds.fill_(0)
        self.portals.fill_(0)
        self.cached_picks = set()
        self.portal_to = {}
        self.game_ticks = 0
        
    def new_game(self, nb_plyers=1, nb_diamoinds=20, nb_portals=2):
        self.nb_plyers = nb_plyers
        self.nb_diamoinds = nb_diamoinds
        self.nb_portals = nb_portals
        self.clear_obj()
        self.id_to_pos = {p: self.random_pos(*self.shape) for p in range(nb_plyers)}
        self.id_to_base = {p: self.id_to_pos[p] for p in range(nb_plyers)}
        self.id_to_point = torch.zeros(nb_plyers, dtype=torch.long)

        for players, (x,y) in self.id_to_pos.items():
            self.base[x][y] = 1
            self.players[x][y] = 1
        portals = [self.random_pos(*self.shape) for _ in range(nb_portals)]
        for i, (x,y) in enumerate(portals):
            self.portals[x][y] = 1
            self.portal_to[(x,y)]= portals[(i+1)%nb_portals]
        self.reset_diamonds()

    def reset_diamonds(self):
        self.cached_diamonds = set()
        for x,y in (self.random_pos(*self.shape, diamonds=True) for _ in range(self.nb_diamoinds)):
            self.diamonds[x][y] = 1

    def get_state(self, id):
        current_player = self.players.clone()
        other_players = self.players.clone()
        current_player[self.id_to_pos[id]] = 1
        other_players[self.id_to_pos[id]] = 0
        current_base = self.base.clone()
        other_bases = self.base.clone()
        current_base[self.id_to_base[id]] = 1
        other_bases[self.id_to_base[id]] = 0
        bag = torch.zeros(self.max_cap, dtype=torch.float32, device=self.device)
        bag[self.bag[self.id_to_pos[id]]-1] = 1
        return torch.stack([current_player, other_players, current_base,  other_bases, self.diamonds, self.portals]).float(), bag

    def get_image(self):
        return (self.players*4 + self.base*3 + self.diamonds*2 +  self.portals*1).cpu()

    def update(self, id, action):
        _from = self.id_to_pos[id]
        _to = GameSim.actions[action](*_from)
        home, carry = -0.1, -0.1
        if self.valid(id, *_to):
            home, carry = self.move(id, _from, _to)
        elif self.unaccepteble(id, *_to):
            home, carry = -1.0, -1.0

        if self.save_image:
            plt.imshow(self.get_image())
            plt.savefig("moves.png")
            plt.close()

        if torch.sum(self.diamonds) == 0:
            self.reset_diamonds()

        return self.bag[self.id_to_pos[id]], home + carry*0.1, self.game_ticks >= self.max_ticks, self.max_ticks - self.game_ticks 


    def move(self, id, _from, _to):
        home = 0
        carry = 0
        curr_bag = self.bag[_from] + self.diamonds[_to]
        if curr_bag <= self.max_cap:
            self.bag[_to] = curr_bag
            carry += self.diamonds[_to]*0.1
            self.diamonds[_to] = 0
        else:
            self.bag[_to] = self.bag[_from]
        self.bag[_from] = 0
        
        self.players[self.id_to_pos[id]] = 0
        if _to in self.portal_to:
            self.id_to_pos[id] = self.portal_to[_to]
        else:
            self.id_to_pos[id] = _to
        self.players[self.id_to_pos[id]] = 1

        if self.id_to_pos[id] == self.id_to_base[id]:
            self.id_to_point[id] += self.bag[_to]
            home += self.bag[_to]
            self.bag[_to] = 0

        self.game_ticks += 1
        return home, carry


    def valid(self, id, x, y):
        if x < 0 or x >= self.shape[0]:
            return False

        if y < 0 or y >= self.shape[1]:
            return False

        for _id, (_x, _y) in self.id_to_base.items():
            if _id == id:
                continue
            if x == _x and y == _y:
                return False

        dest_x, dest_y = -1, -1

        if (x,y) in self.portal_to:
            dest_x, dest_y = self.portal_to[x,y]

        for _id, (_x, _y) in self.id_to_pos.items():
            if _id == id:
                continue
            if x == _x and y == _y:
                return False
            if dest_x == _x and dest_y == _y:
                return False

        return True
    def unaccepteble(self, id, x, y):
        if x < 0 or x >= self.shape[0]:
            return True

        if y < 0 or y >= self.shape[1]:
            return True

        # for _id, (_x, _y) in self.id_to_base.items():
        #     if _id == id:
        #         continue
        #     if x == _x and y == _y:
        #         return True
        return False
        
    def get_data(self, id):
        player = {
            "name": id,
            "pos": self.id_to_pos[id],
            "diamond": self.bag[self.id_to_pos[id]], 
            "base": self.id_to_base[id]
        }

        agents = []
        for i, pos in self.id_to_pos.items():
            agents.append({
                "name": i,
                "pos": pos,
                "diamond": self.bag[pos], 
                "base": self.id_to_base[i]
            })
        diamonds = []
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                value = self.diamonds[x,y]
                if value > 0:
                    diamonds.append({
                        "pos": (x,y),
                        "value": value
                    })
        portals = []
        for i, p in self.portal_to.items():
            portals.append({
                "pos":p
            })

        return Data(player, agents, diamonds, portals)


    def random_pos(self, x, y, diamonds=False):
        pos = randint(0, x-1), randint(0, y-1)
        while pos in self.cached_picks or pos in self.cached_diamonds:
            pos = randint(0, x-1), randint(0, y-1)
        if diamonds:
            self.cached_diamonds.add(pos)
        else:
            self.cached_picks.add(pos)
        return pos