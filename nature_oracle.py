"""
implement nature oracle of double oracle

Lily Xu, 2020
"""

import sys, os
import pickle
import itertools
import random
import numpy as np
import torch

from park import Park, convert_to_a
from ddpg_nature import NatureDDPG

torch.autograd.set_detect_anomaly(True)


def sample_strategy(distrib):
    """ return the index of a strategy
    general util used by nature and agent oracle """
    strategy_i = random.choices(list(range(len(distrib))), weights=distrib)[0]
    return strategy_i


class NatureOracle:
    def __init__(self, park_params, checkpoints, n_train,
        use_wake, freeze_policy_step, freeze_a_step):
        """
        use_wake: whether to use wake/sleep option
        freeze_policy_step: how often to freeze policy
        freeze_a_step: how often to unfreeze attractiveness
        """
        self.park_params = park_params
        self.n_train     = n_train
        self.use_wake    = use_wake
        self.freeze_policy_step = freeze_policy_step
        self.freeze_a_step      = freeze_a_step

    def best_response(self, agent_strategies, agent_distrib, display=True):
        """
        agent_strategies: agent strategy set
        agent_distrib: mixed strategy distribution over the set

        returns: best response attractiveness
        """

        br = self.run_DDPG(agent_strategies, agent_distrib, display=display)
        br = br.detach().numpy()

        return br


    def run_DDPG(self, agent_strategies, agent_distrib, display=True):
        """ LEARNING ORACLE

        freeze_policy_step: for wake/sleep procedure, how often to freeze policy
        freeze_a_step: for wake/sleep procedure, how often to *unfreeze* attractiveness """

        # initialize with random attractiveness values in interval
        attractiveness = (np.random.rand(self.park_params['n_targets']) - .5) * 2
        attractiveness = attractiveness.astype(float)
        print('attractiveness (raw)', np.round(attractiveness, 3))

        attractiveness = torch.tensor(attractiveness, requires_grad=True, dtype=torch.float32)

        ddpg = NatureDDPG(self.park_params['n_targets'], attractiveness, actor_learning_rate=10, critic_learning_rate=10)

        batch_size  = 10
        rewards     = []
        avg_rewards = []

        def get_agent_avg_reward(env, agent_strategy, n_iter=100):
            agent_total_rewards = torch.zeros(self.park_params['horizon'])
            for i in range(n_iter):
                state = env.reset()
                for t in itertools.count():
                     action = agent_strategy.select_action(state)
                     action = torch.Tensor(action)
                     next_state, reward, done, info = env.step(action, use_torch=True)
                     agent_total_rewards[t] += reward
                     state = next_state

                     if done:
                         break

            agent_avg_rewards = agent_total_rewards / n_iter
            if display:
                print('agent avg rewards', agent_avg_rewards.detach().numpy())
            return agent_avg_rewards

        total_step = 0

        env = Park(attractiveness, self.park_params['initial_effort'], self.park_params['initial_wildlife'], self.park_params['initial_attack'],
            self.park_params['height'], self.park_params['width'], self.park_params['n_targets'], self.park_params['budget'], self.park_params['horizon'],
            self.park_params['psi'], self.park_params['alpha'], self.park_params['beta'], self.park_params['eta'], param_int=self.park_params['param_int'])

        # memoize agent average reward for each policy
        agent_avg_rewards = []
        for agent_strategy in agent_strategies:
            agent_avg_rewards.append(get_agent_avg_reward(env, agent_strategy))

        print('agent strategies', len(agent_strategies))
        print('avg rewards', len(agent_avg_rewards), np.array([np.round(r.detach().numpy(), 2) for r in agent_avg_rewards]))

        # until convergence
        for i_episode in range(self.n_train):
            i_display = True if display and i_episode % 50 == 0 else False

            if self.use_wake:
                updating_a      = i_episode % self.freeze_a_step == 0 # are we updating a?
                updating_policy = i_episode % self.freeze_policy_step > 0 # are we updating policy?

                if updating_a:
                    ddpg.unfreeze_attractiveness()
                else:
                    ddpg.freeze_attractiveness()

                if updating_policy and i_episode > 0:
                    ddpg.freeze_policy()
                else:
                    ddpg.unfreeze_policy()

            else:
                updating_a = True
                updating_policy = True

            env = Park(attractiveness, self.park_params['initial_effort'], self.park_params['initial_wildlife'], self.park_params['initial_attack'],
                self.park_params['height'], self.park_params['width'], self.park_params['n_targets'], self.park_params['budget'], self.park_params['horizon'],
                self.park_params['psi'], self.park_params['alpha'], self.park_params['beta'], self.park_params['eta'], param_int=self.park_params['param_int'])

            state = env.reset()
            episode_reward = 0

            a = convert_to_a(env.attractiveness.detach().numpy(), self.park_params['param_int'])
            if i_display:
                print('episode {} attractiveness {} raw {}'.format(i_episode, np.round(a, 3), np.round(raw_a, 3)))
            state = torch.cat([state, attractiveness])

            # get reward of sampled agent strategy
            agent_strategy_i  = sample_strategy(agent_distrib)
            agent_avg_reward = agent_avg_rewards[agent_strategy_i]

            # for timesteps in one episode
            for t in itertools.count():

                action = ddpg.select_action(state)

                next_state, reward, done, info = env.step(action, use_torch=True)
                next_state = torch.cat([next_state, attractiveness])

                if i_display:
                    print('t {} action {} reward {:.3f}'. format(t, np.round(action.detach().numpy(), 3), reward.item()))

                reward = reward - agent_avg_reward[t]  # want to max agent regret
                reward = reward.unsqueeze(0)
                ddpg.memory.push(state, action, reward, next_state, done)

                if len(ddpg.memory) > batch_size:
                    ddpg.update(batch_size, display=i_display)

                state = next_state
                episode_reward += reward

                if done:
                    if updating_a:
                        state = env.reset()
                        # if we update attractiveness, update agent avg rewards
                        for i, agent_strategy in enumerate(agent_strategies):
                            agent_avg_rewards[i] = get_agent_avg_reward(env, agent_strategy)
                    break

            rewards.append(episode_reward)
            avg_rewards.append(torch.mean(torch.stack(rewards[-10:])))

        return attractiveness
