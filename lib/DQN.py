import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, outputs=4, in_chanals=6, bag=5, hiden=256):
        super(DQN, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv2d(in_chanals, 64, kernel_size=3, stride=2, bias=True), #7, 5
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=True),#3, 2
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=True),#1, 1
                            nn.ReLU(inplace=True),
                            nn.Flatten()
        )
        self.hiden = nn.Sequential(
            nn.Linear(256 + bag, hiden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hiden, outputs, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        state, bag = inputs
        #for m in self.encoder:
        state = self.encoder(state)
        #print(state.shape)
        x = torch.cat([state, bag], dim=1)
        x = self.hiden(x)
        #print(x)
        return x