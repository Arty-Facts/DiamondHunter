
class Player():
    def __init__(self, id, pos, diamonds, base):
        self.id = id
        self.pos = pos
        self.diamonds = diamonds
        self.base = base

class Diamond():
    def __init__(self, pos, value):
        self.pos = pos
        self.value = value

class Portal():
    def __init__(self, pos, goto):
        self.pos = pos
        self.goto = goto

class Data():
    def __init__(self, player, agnets, diamonds, portals):
        self.player = Player(player["name"], player["pos"], player["diamond"], player["base"])
        self.agents = []
        self.diamonds = []
        self.portals = []
        for a in agnets:
            if a["name"] == self.player.id:
                continue
            self.agents.append(Player(a["name"], a["pos"], a["diamond"], a["base"]))
        
        for d in diamonds:
            self.diamonds.append(Diamond(d["pos"], d["value"]))

        for i, p in enumerate(portals):
            self.portals.append(Portal(p["pos"], portals[(i+1)%len(portals)]["pos"]))
