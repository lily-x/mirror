"""
implement DDPG for nature oracle

Lily Xu, 2021
"""

import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpg import Critic, Actor, NormalizedEnv, ReplayBuffer

import random
from collections import deque


class NatureDDPG:
    def __init__(self, n_targets, attractiveness,
        hidden_size=256, actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        attractiveness_learning_rate=1e-2,
        gamma=0.99, tau=1e-2,
        memory_max_size=50000):

        # params
        self.states_dim = 3 * n_targets + 1  # effort, wildlife, attractiveness, timestep
        self.actions_dim = n_targets  # alternative pi
        self.gamma = gamma
        self.tau = tau

        self.attractiveness = attractiveness  # torch tensor

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

        self.actor_optimizer  = torch.optim.Adam([
                {'params': list(self.actor.parameters()), 'lr': actor_learning_rate},
                {'params': [self.attractiveness], 'lr': attractiveness_learning_rate},
            ], lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = ReplayBuffer(memory_max_size)
        self.loss = nn.MSELoss()


    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)
        action = self.actor.forward(state)[0]
        return action

    # toggle freeze parameters
    def freeze_attractiveness(self):
        self.attractiveness.requires_grad = False

    def unfreeze_attractiveness(self):
        self.attractiveness.requires_grad = True

    def freeze_policy(self):
        for param in self.actor.parameters():
            param.requires_grad = False

    def unfreeze_policy(self):
        for param in self.actor.parameters():
            param.requires_grad = True


    def update(self, batch_size, display=False):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states      = torch.stack(states)
        actions     = torch.stack(actions)
        rewards     = torch.stack(rewards)
        next_states = torch.stack(next_states)

        # update critic by minimizing loss
        Q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + self.gamma * next_Q

        critic_loss = self.loss(Q_vals, Q_prime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # update actor policy using sampled policy gradient
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        if display:
            print('    actor loss {:.4f}, critic loss {:.4f}'.format(actor_loss, critic_loss))

        # update target networks (slowly using soft updates)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
