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
            nn.LSTM(hiden, hiden, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(hiden, outputs, bias=True)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, inputs):
        state, bag = inputs
        for m in self.encoder:
            state = m(state)
            print(state.shape)
        x = torch.cat([state.squeeze(), bag])
        x = self.hiden(x)
        print(x.shape)
        return self.out(x)
