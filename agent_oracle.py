"""
implement agent oracle of double oracle

Lily Xu, 2021
"""

import sys, os
import pickle
import itertools
import random
import numpy as np
import torch
from collections.abc import Iterable

from park import Park
from ddpg import *

from nature_oracle import sample_strategy

N_DISPLAY = 100


class AgentOracle:
    def __init__(self, park_params, checkpoints, n_train, n_eval):
        self.park_params = park_params
        self.checkpoints = checkpoints
        self.n_train     = n_train
        self.n_eval      = n_eval
        self.budget      = park_params['budget']

    def simulate_reward(self, def_strategies, nature_strategies, def_distrib=None, nature_distrib=None, display=True):
        """ this is similar to evaluate_DDPG() from defender_oracle_evaluation.py
        if def_distrib=None, def_strategies only a single strategy
        if nature_distrib=None, nature_strategies only a single strategy """
        if def_distrib is None:
            assert len(def_strategies) == 1
            def_strategy = def_strategies[0]
        else:
            assert len(def_strategies) == len(def_distrib)

        if nature_distrib is None:
            assert len(nature_strategies) == 1
            attractiveness = nature_strategies[0]
        else:
            assert len(nature_strategies) == len(nature_distrib)

        rewards = np.zeros(self.n_eval)
        for i_episode in range(self.n_eval):
            if nature_distrib is not None:
                nature_strategy_i = sample_strategy(nature_distrib)
                attractiveness = nature_strategies[nature_strategy_i]
            if def_distrib is not None:
                def_strategy_i = sample_strategy(def_distrib)
                def_strategy = def_strategies[def_strategy_i]

            park_params = self.park_params
            env = Park(attractiveness, park_params['initial_effort'], park_params['initial_wildlife'], park_params['initial_attack'],
                    park_params['height'], park_params['width'], park_params['n_targets'], park_params['budget'], park_params['horizon'],
                    park_params['psi'], park_params['alpha'], park_params['beta'], park_params['eta'], param_int=park_params['param_int'])

            # initialize the environment and state
            state = env.reset()
            for t in itertools.count():
                # select and perform an action
                action = def_strategy.select_action(state)

                # if DDPG (which returns softmax): take action up to budget and then clip each location to be between 0 and 1
                if isinstance(def_strategy, DDPG):
                    before_sum = action.sum()
                    action = (action / action.sum()) * self.budget
                    action[np.where(action > 1)] = 1 # so DDPG learns to not make actions greater than budget

                next_state, reward, done, _ = env.step(action, use_torch=False)

                if display and i_episode % 1000 == 0:
                    print('  ', i_episode, t, action)

                # move to the next state
                state = next_state

                # evaluate performance if terminal
                if done:
                    rewards[i_episode] = reward
                    break

        avg_reward = np.mean(rewards)
        return avg_reward


    def best_response(self, nature_strategies, nature_distrib, display=False):
        assert len(nature_strategies) == len(nature_distrib), 'nature strategies {}, distrib {}'.format(len(nature_strategies), len(nature_distrib))
        br, checkpoint_rewards = run_DDPG(self.park_params, nature_strategies, nature_distrib, self.checkpoints, self.n_train, display=display)

        return br



def run_DDPG(park_params, nature_strategies, nature_distrib, checkpoints, n_train, display=True):
    state_dim  = 2*park_params['n_targets'] + 1
    action_dim = park_params['n_targets']

    ddpg = DDPG(park_params['n_targets'])

    batch_size = 128
    rewards = []
    avg_rewards = []
    checkpoint_rewards = []

    # if args.load: agent.load()
    total_step = 0
    for i_episode in range(n_train):
        episode_reward = 0

        nature_strategy_i = sample_strategy(nature_distrib)
        attractiveness = nature_strategies[nature_strategy_i]

        park = Park(attractiveness, park_params['initial_effort'], park_params['initial_wildlife'], park_params['initial_attack'],
                park_params['height'], park_params['width'], park_params['n_targets'], park_params['budget'], park_params['horizon'],
                park_params['psi'], park_params['alpha'], park_params['beta'], park_params['eta'])

        # initialize the environment and state
        state = park.reset()

        for t in itertools.count():
            action = ddpg.select_action(state)

            next_state, reward, done, info = park.step(action)
            reward = info['expected_reward']  # use expected reward

            ddpg.memory.push(state, action, np.expand_dims(reward, axis=0), next_state, done)

            if len(ddpg.memory) > batch_size:
                ddpg.update(batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))


        if display and i_episode % N_DISPLAY == 0:
            print('episode {:4d}   reward: {:.2f}   average reward: {:.2f}'.format(i_episode, np.round(episode_reward, 2), np.mean(rewards[-10:])))

    return ddpg, checkpoint_rewards
