
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from lib.DQN import DQN
from lib.ReplayMemory import ReplayMemory
from lib.GameSim import GameSim

from agents.BFSAgent import BFSAgent

from random import randint

from time import time




NB_PLAYERS = 2
BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000*600*NB_PLAYERS
TARGET_UPDATE = 10
num_episodes = 100000*NB_PLAYERS
env = GameSim(max_ticks=600*NB_PLAYERS, nb_plyers=NB_PLAYERS ,device="cuda" if torch.cuda.is_available() else "cpu", save_image=False)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def get_screen():
    screen, bag = env.get_state(0)
    return screen.unsqueeze(0), bag.unsqueeze(0)

env.new_game( nb_plyers=NB_PLAYERS)
init_screen = get_screen()

# Get number of actions 
n_actions = env.action_space

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
agent = BFSAgent()
optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state, id=0):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if steps_done < 100*600*NB_PLAYERS and False:
        data = env.get_data(id)
        action = agent.next_move(data)
        if action == 4:
            action = randint(0,3)
        return torch.tensor([[action]], device=device, dtype=torch.long)

    elif sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        action = randint(0,3)
        return torch.tensor([[action]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Diamonds')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig('progress.png')
    plt.close()



# ckpt = torch.load("checkpoints/latest")
# policy_net.load_state_dict(ckpt["policy_net"])
# target_net.load_state_dict(ckpt["target_net"])
# optimizer.load_state_dict(ckpt["optimizer"])
# episode_durations = ckpt["episode_durations"]
# steps_done = ckpt["steps_done"]


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
   
    # for d in batch.next_state:
    #     if d is not None:
    #         s, b = d
    #         print(s.shape)
    #         print(b.shape)
    non_final_next_states = (torch.cat([s[0] for s in batch.next_state if s is not None]),
                             torch.cat([s[1] for s in batch.next_state if s is not None]))

    state_batch = (torch.cat(tuple(map(lambda x: x[0], batch.state))),
                   torch.cat(tuple(map(lambda x: x[1], batch.state))))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    #print(non_final_mask)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(num_episodes):
    print(f"\repisode: {i_episode}/{num_episodes}, steps_done: {steps_done}", end="\r")
    # Initialize the environment and state
    env.new_game(nb_plyers=NB_PLAYERS)
    last_screen, last_bag = get_screen()
    current_screen, current_bag = get_screen()
    state = current_screen - last_screen , current_bag - last_bag
    for t in count():
        # Select and perform an action
        action = select_action(state, t%NB_PLAYERS)
        _, reward, done, _ = env.update(t%NB_PLAYERS, action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen, last_bag = current_screen, current_bag 
        current_screen, current_bag = get_screen()
        if not done:
            next_state = current_screen - last_screen, current_bag - last_bag
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(reward)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        torch.save({
            'policy_net': policy_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'episode_durations':episode_durations, 
            'steps_done': steps_done
            }, "checkpoints/latest")
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')


torch.save({
            'policy_net': policy_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'episode_durations':episode_durations

            }, "checkpoints/latest")



