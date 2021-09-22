"""
implementation of DDPG

DPG + DQN: deterministic PG + deep Q network
model-free, off-policy, actor-critic
DQN - discrete space; DDPG - continuous space

Adapted from https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py

Lily Xu, 2021
"""

import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input: state, action
        output: value of each action
        """

        hidden_size1 = 16
        hidden_size2 = 32
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        """
        params
        state and action - both torch Tensors
        """
        x = torch.cat([state, action], 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate=3e-4):
        """
        input: state
        output: action to take
        """
        hidden_size1 = 16
        hidden_size2 = 32
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x



########### replay buffer

import random
from collections import deque

class ReplayBuffer:
    """ experience replay to overcome challenge of data not being independently distributed
    during on-policy training """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, s, a, r, ss, done):
        """ state, action, reward, next_state """
        experience = (s, a, r, ss, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s_batch    = [experience[0] for experience in batch]
        a_batch    = [experience[1] for experience in batch]
        r_batch    = [experience[2] for experience in batch]
        ss_batch   = [experience[3] for experience in batch]
        done_batch = [experience[4] for experience in batch]

        return s_batch, a_batch, r_batch, ss_batch, done_batch

    def __len__(self):
        return len(self.buffer)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv:
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)



class DDPG:
    def __init__(self, n_targets, actor_learning_rate=1e-4,
        critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
        memory_max_size=50000):

        # params
        self.states_dim = 2 * n_targets + 1
        self.actions_dim = n_targets
        self.gamma = gamma
        self.tau = tau

        ##### networks
        # randomly initialize critic and actor network
        self.actor         = Actor(self.states_dim, self.actions_dim)
        self.critic        = Critic(self.states_dim + self.actions_dim, 1)

        # initialize critic and actor target network
        self.actor_target  = Actor(self.states_dim, self.actions_dim)
        self.critic_target = Critic(self.states_dim + self.actions_dim, 1)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = ReplayBuffer(memory_max_size)
        self.loss = nn.MSELoss()


    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action


    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # update critic by minimizing loss
        Q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + self.gamma * next_Q

        critic_loss = self.loss(Q_vals, Q_prime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor policy using sampled policy gradient
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks (slowly using soft updates)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
