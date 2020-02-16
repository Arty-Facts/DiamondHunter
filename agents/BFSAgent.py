from collections import deque
import queue as Q
import copy
from random import random, randint
from time import time
delay = 100

class BFSAgent(object):
    Remapper = {
        (0,-1): 0,
        (1,0): 1,
        (0,1): 2,
        (-1,0): 3,
        (0,0): 4
    }

    def at_goal(self, x, y, data):
        if data.player.diamonds >= 5:
            if (x,y) == data.player.base:
                return True
            return False

        for d in data.diamonds:
            if (x,y) == d.pos and data.player.diamonds + d.value <=5 :
                return True
        return False

    def valid_move(self, x, y, data):
        if x == 15 or x == -1:
            return False
            
        if y == 12 or y == -1:
            return False

        for bot in data.agents:
            if (x,y) == bot.pos:
                return False
        # if self.at_portal(x,y, data):
        #     return False

        return True

    def at_portal(self, x, y, data):
        for portal in data.portals:
            if (x,y) == portal.pos:
                return True
        return False

    
    def expand(self, x, y, m, viseted, data, frontear):
        for portal in data.portals:
            px, py = portal.pos
            if not (x,y) in viseted and px == x and py == y:
                viseted.add((x,y))
                viseted.add((px,py))
                frontear.append([px, py, m])
                break
        else:
            if not (x,y) in viseted and self.valid_move(x, y, data): 
                viseted.add((x,y))
                frontear.append([x, y, m])

    def get_move(self, data):
        frontear = deque()  
        x, y = data.player.pos
        viseted = set((x,y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            self.expand(x+dx, y+dy, (dx,dy), viseted, data, frontear)

        while len(frontear) > 0:
            curr = frontear.popleft()
            x, y, m = curr
            if self.at_goal(x, y, data):
                return m
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                self.expand(x+dx, y+dy, m, viseted, data, frontear)
        return 0,0


    def next_move(self, data):
        move = self.get_move(data)
        return BFSAgent.Remapper[move]