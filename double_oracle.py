"""
double oracle implementation - putting it all together

Lily Xu, 2021
"""

import sys, os
import pickle
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from scipy import signal # for 2D gaussian kernel
import torch

import matplotlib.pyplot as plt

from park import convert_to_a
from agent_oracle import AgentOracle
from nature_oracle import NatureOracle
from nfg_solver import solve_game, solve_minimax_regret, get_payoff

from min_reward_oracle import min_reward

if not os.path.exists('plots'):
    os.makedirs('plots')


class DoubleOracle:
    def __init__(self, max_epochs, height, width, budget, horizon,
                n_perturb, n_eval, agent_n_train, nature_n_train,
                attract_vals, psi, alpha, beta, eta,
                max_interval, wildlife_setting, use_wake,
                checkpoints, freeze_policy_step, freeze_a_step,
                ):
        self.max_epochs  = max_epochs

        self.park_height = height
        self.park_width  = width
        self.budget      = budget
        self.horizon     = horizon

        self.n_targets   = self.park_height * self.park_width

        self.n_perturb = n_perturb

        # attractiveness parameter interval
        int = np.random.uniform(0, max_interval, size=self.n_targets)
        self.param_int = [(attract_vals[i]-int[i], attract_vals[i]+int[i]) for i in range(self.n_targets)]
        self.param_int = [(attract_vals[i], attract_vals[i]+int[i]) for i in range(self.n_targets)]
        print('param_int', [tuple(np.round(int, 2)) for int in self.param_int])

        self.param_int = np.array(self.param_int)
        assert np.all(self.param_int[:, 1] >= self.param_int[:, 0])

        def gkern(kernlen=21, std=3):
            """ returns a 2D Gaussian kernel array """
            gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
            gkern2d = np.outer(gkern1d, gkern1d)
            return gkern2d.flatten()

        if wildlife_setting == 1: # random
            initial_wildlife = np.random.rand(self.n_targets) * 3
        elif wildlife_setting == 2: # gaussian kernel - peaked
            assert height == width
            initial_wildlife = gkern(height, 1.5) * 3
        elif wildlife_setting == 3: # gaussian kernel - not very peaked
            assert height == width
            initial_wildlife = gkern(height, 5) * 3
        else:
            raise Exception('wildlife setting {} not implemented'.format(wildlife_setting))


        # setup park parameters dict
        self.park_params = {
          'height':           self.park_height,
          'width':            self.park_width,
          'budget':           self.budget,
          'horizon':          self.horizon,
          'n_targets':        self.n_targets,
          'initial_effort':   np.zeros(self.n_targets),
          'initial_wildlife': initial_wildlife,
          'initial_attack':   np.zeros(self.n_targets),
          'param_int':        self.param_int,
          'psi':   psi,
          'alpha': alpha,
          'beta':  beta,
          'eta':   eta
        }

        print('initial wildlife {:.2f} {}'.format(np.sum(initial_wildlife), np.round(initial_wildlife, 2)))


        self.agent_oracle  = AgentOracle(self.park_params, checkpoints, agent_n_train, n_eval)
        self.nature_oracle = NatureOracle(self.park_params, checkpoints, nature_n_train, use_wake, freeze_policy_step, freeze_a_step)

        # initialize attractiveness
        init_attractiveness = (np.random.rand(self.n_targets) - .5) * 2

        # initialize strategy sets
        self.agent_strategies  = []  # agent policy
        self.nature_strategies = [init_attractiveness]  # attractiveness
        self.payoffs           = [] # agent regret for each (agent strategy, attractiveness) combo


    def run(self):
        agent_eq  = np.array([1.]) # account for baselines
        nature_eq = np.array([1.])  # initialize nature distribution

        # repeat until convergence
        converged = False
        n_epochs = 1
        while not converged:
            print('-----------------------------------')
            print('epoch {}'.format(n_epochs))
            print('-----------------------------------')

            # if first epoch, agent response is ideal agent for initial attractiveness
            agent_br = self.agent_oracle.best_response(self.nature_strategies, nature_eq, display=False)
            nature_br = self.nature_oracle.best_response(self.agent_strategies, agent_eq, display=False)

            # REWARD RANDOMIZATION
            # repeat with more perturbations of nature strategies
            print('  nature BR', np.round(nature_br, 3))

            self.update_payoffs(nature_br, agent_br)

            # perturb
            for i in range(self.n_perturb):
                perturb = np.random.normal(scale=0.5, size=nature_br.shape)
                perturbed_br = nature_br + perturb
                print('  perturbed', np.round(perturbed_br, 3))

                agent_perturbed_br = self.agent_oracle.best_response([perturbed_br], [1], display=False)

                self.update_payoffs(perturbed_br, agent_perturbed_br)

            # find equilibrium of subgame
            agent_eq, nature_eq = self.find_equilibrium()

            print('agent equilibrium  ', np.round(agent_eq, 3))
            print('nature equilibrium ', np.round(nature_eq, 3))

            max_regret_game = np.array(self.payoffs) - np.array(self.payoffs).max(axis=0)

            if n_epochs >= self.max_epochs: # terminate after a max number of epochs
                converged = True
                break

            n_epochs += 1

            assert len(self.payoffs) == len(self.agent_strategies), '{} payoffs, {} agent strategies'.format(len(self.payoffs), len(self.agent_strategies))
            assert len(self.payoffs[0]) == len(self.nature_strategies), '{} payoffs[0], {} nature strategies'.format(len(self.payoff[0]), len(self.nature_strategies))

        return agent_eq, nature_eq


    def compute_regret(self, agent_s, nature_s, max_reward):
        reward = self.agent_oracle.simulate_reward([agent_s], [nature_s], display=False)
        regret = max_reward - reward
        if regret < 0:
            print('  uh oh! regret is negative. max reward {:.3f}, reward {:.3f}'.format(max_reward, reward))
        return regret

    def compute_payoff_regret(self, agent_eq):
        """ given a agent mixed strategy, compute the expected regret in the payoff matrix """
        assert abs(sum(agent_eq) - 1) <= 1e-3

        regret = np.array(do.payoffs) - np.array(do.payoffs).max(axis=0)
        # if agent playing a pure strategy
        if len(np.where(agent_eq > 0)[0]) == 1:
            agent_strategy_i = np.where(agent_eq > 0)[0].item()
            strategy_regrets = regret[agent_strategy_i]
            return -np.min(strategy_regrets) # return max regret (min reward)
        else:
            raise Exception('not implemented')

    def find_equilibrium(self):
        """ solve for minimax regret-optimal mixed strategy """
        agent_eq, nature_eq = solve_minimax_regret(self.payoffs)
        return agent_eq, nature_eq

    def update_payoffs(self, nature_br, agent_br):
        """ update payoff matrix (in place) """
        self.update_payoffs_agent(agent_br)
        self.update_payoffs_nature(nature_br)

    def update_payoffs_agent(self, agent_br):
        """ update payoff matrix (only adding agent strategy)

        returns index of new strategy """
        self.agent_strategies.append(agent_br)

        # for new agent strategy: compute regret w.r.t. all nature strategies
        new_payoffs = []
        for i, nature_s in enumerate(self.nature_strategies):
            reward = self.agent_oracle.simulate_reward([agent_br], [nature_s], display=False)
            new_payoffs.append(reward)
        self.payoffs.append(new_payoffs)

        return len(self.agent_strategies) - 1

    def update_payoffs_nature(self, nature_br):
        """ update payoff matrix (only adding nature strategy)

        returns index of new strategy """
        self.nature_strategies.append(nature_br)

        # update payoffs
        # for new nature strategy: compute regret w.r.t. all agent strategies
        for i, agent_s in enumerate(self.agent_strategies):
            reward = self.agent_oracle.simulate_reward([agent_s], [nature_br], display=False)
            self.payoffs[i].append(reward)

        return len(self.nature_strategies) - 1



###################################################
# baselines
###################################################

def use_middle(param_int, agent_oracle):
    """ solve optimal reward relative to midpoint of uncertainty interval
    sequential policy, but based on the center of the uncertainty set """

    attractiveness = param_int.mean(axis=1)
    agent_br = agent_oracle.best_response([attractiveness], [1.], display=True)
    return agent_br

def maximin(park_params, agent_oracle):
    """ maximize min reward, analogous to robust adversarial RL (RARL) """
    n_iters = 10
    # pick initial attractiveness at random
    attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
    attractiveness = attractiveness.astype(float)

    for iter in range(n_iters):
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], display=False)
        if iter == n_iters - 1: break
        attractiveness = min_reward(park_params, agent_strategy, attractiveness_learning_rate=5e-2, n_iter=500, batch_size=64, visualize=False, init_attractiveness=attractiveness)

    return agent_strategy

def RARL_regret(park_params, agent_oracle, nature_oracle):
    """ use a weakened form of MIRROR that is equivalent to RARL with regret,
    using the nature oracle to compute regret instead of maximin reward """
    n_iters = 10
    # pick initial attractiveness at random
    attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
    attractiveness = attractiveness.astype(float)

    for iter in range(n_iters):
        agent_strategy = agent_oracle.best_response([attractiveness], [1.], display=False)
        if iter == n_iters - 1: break
        attractiveness = nature_oracle.best_response([agent_strategy], [1.], display=False)

    return agent_strategy

def myopic(param_int, agent_oracle):
    """ regular myopic - can use whatever method to come up with policies. will need to evaluate based on minimax regret

    myopic minimax? look at only our reward in the next timestep - would need
    to use bender's decomposition to compute. and we also have continuous policies
    """
    pass

class RandomPolicy:
    def __init__(self, park_params):
        self.n_targets = park_params['n_targets']
        self.budget    = park_params['budget']

    def select_action(self, state):
        max_effort = 1 # max effort at any target

        action = np.random.rand(self.n_targets)
        action /= action.sum()
        action *= self.budget

        # ensure we never exceed effort = 1 on any target
        while len(np.where(action > max_effort)[0]) > 0:
            excess_idx = np.where(action > 1)[0][0]
            excess = action[excess_idx] - max_effort

            action[excess_idx] = max_effort

            # add "excess" amount of effort randomly on other targets
            add = np.random.uniform(size=self.n_targets - 1)
            add = (add / np.sum(add)) * excess

            action[:excess_idx] += add[:excess_idx]
            action[excess_idx+1:] += add[excess_idx:]

        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIRROR robust reinforcement learning under minimax regret')

    parser.add_argument('--seed',         type=int, default=0, help='random seed')
    parser.add_argument('--n_eval',       type=int, default=100, help='number of points to evaluate agent reward')
    parser.add_argument('--agent_train',  type=int, default=100, help='number of training iterations for agent')
    parser.add_argument('--nature_train', type=int, default=100, help='number of training iterations for nature')
    parser.add_argument('--max_epochs',   type=int, default=5, help='max num epochs to run double oracle')
    parser.add_argument('--n_perturb',    type=int, default=3, help='number of perturbations to add in each epoch')
    parser.add_argument('--wake',         type=int, default=1, help='whether to use wake/sleep (binary option)')

    parser.add_argument('--freeze_policy_step', type=int, default=5, help='how often to freeze policy (nature wake/sleep)')
    parser.add_argument('--freeze_a_step',      type=int, default=5, help='how often to unfreeze attractiveness (nature wake/sleep)')

    # set park parameters
    parser.add_argument('--height',  type=int, default=5, help='park height')
    parser.add_argument('--width',   type=int, default=5, help='park width')
    parser.add_argument('--budget',  type=int, default=5, help='agent budget')
    parser.add_argument('--horizon', type=int, default=5, help='agent planning horizon')

    parser.add_argument('--interval',   type=float, default=3, help='uncertainty interval max size')
    parser.add_argument('--wildlife',   type=int,   default=1, help='wildlife option')
    parser.add_argument('--deterrence', type=int,   default=1, help='deterrence option')

    parser.add_argument('--prefix', type=str, default='', help='filename prefix')

    args = parser.parse_args()

    seed           = args.seed
    n_eval         = args.n_eval
    agent_n_train  = args.agent_train
    nature_n_train = args.nature_train
    max_epochs     = args.max_epochs
    n_perturb      = args.n_perturb
    use_wake       = args.wake == 1

    # parameters for nature oracle wake/sleep
    freeze_policy_step = args.freeze_policy_step
    freeze_a_step      = args.freeze_a_step

    height  = args.height
    width   = args.width
    budget  = args.budget
    horizon = args.horizon

    max_interval = args.interval
    wildlife_setting = args.wildlife
    deterrence_setting = args.deterrence

    prefix  = args.prefix

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_filename = './data/sample.p'
    data = pickle.load(open(data_filename, 'rb'))
    start_idx = np.random.randint(len(data['attract_vals'][0]) - height*width)
    attract_vals = data['attract_vals'][0][start_idx:start_idx + height*width] # pick random series of attractiveness values
    print('attract_vals', np.round(attract_vals, 2))
    attract_vals = np.array(attract_vals) + 13
    print('attract_vals', np.round(attract_vals, 2))

    psi     = 1.1 # wildlife growth ratio
    alpha   = .5  # strength that poachers eliminate wildlife
    eta     = .3  # effect of neighbors
    if deterrence_setting == 1:
        beta = -5
    elif deterrence_setting == 2:
        beta = -3
    elif deterrence_setting == 3:
        beta = -8
    print('beta is', beta)
    print('eta is', eta)
    print('all beta', data['past_effort_vals'])

    print('beta {:.3f}, eta {:.3f}'.format(beta, eta))

    checkpoints = [1, 50, 100, 500, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 80000, 100000, 120000, 150000, 170000]#, N_TRAIN-1]



    do = DoubleOracle(max_epochs, height, width, budget, horizon,
        n_perturb, n_eval, agent_n_train, nature_n_train,
        attract_vals, psi, alpha, beta, eta,
        max_interval, wildlife_setting, use_wake,
        checkpoints, freeze_policy_step, freeze_a_step)

    print('max_epochs {}, n_train agent {}, nature {}'.format(max_epochs, agent_n_train, nature_n_train))
    print('n_targets {}, horizon {}, budget {}'.format(do.n_targets, horizon, budget))

    # # baseline: middle of uncertainty interval
    baseline_middle_i = len(do.agent_strategies)
    start_time = time.time()
    for i in range(n_perturb+1):
        baseline_middle = use_middle(do.param_int, do.agent_oracle)
        do.update_payoffs_agent(baseline_middle)
    middle_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline middle runtime {:.1f} seconds'.format(middle_time))

    # baseline: random
    baseline_random_i = len(do.agent_strategies)
    for i in range(n_perturb+1):
        random_policy = RandomPolicy(do.park_params)
        do.update_payoffs_agent(random_policy)

    # baseline: maximin robust (robust adversarial RL - RARL)
    baseline_maximin_i = len(do.agent_strategies)
    start_time = time.time()
    for i in range(n_perturb+1):
        maximin_policy = maximin(do.park_params, do.agent_oracle)
        do.update_payoffs_agent(maximin_policy)
    maximin_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline maximin runtime {:.1f} seconds'.format(maximin_time))

    # baseline: RARL with regret
    baseline_RARL_regret_i = len(do.agent_strategies)
    start_time = time.time()
    for i in range(n_perturb+1):
        RARL_regret_policy = RARL_regret(do.park_params, do.agent_oracle, do.nature_oracle)
        do.update_payoffs_agent(RARL_regret_policy)
    RARL_regret_time = (time.time() - start_time) / (n_perturb+1)
    print('baseline RARL_regret runtime {:.1f} seconds'.format(RARL_regret_time))

    print('strategies', do.agent_strategies)

    start_time = time.time()
    agent_eq, nature_eq = do.run()
    do_time = time.time() - start_time
    print('DO runtime {:.1f} seconds'.format(do_time))

    print('\n\n\n\n\n-----------------------')
    # print('equilibrium value is ', val_upper, val_lower)
    print('agent BR mixed strategy           ', np.round(agent_eq, 4))
    print('Nature attractiveness mixed strategy ', np.round(nature_eq, 4))
    print('Nature attractiveness are')
    for nature_strategy in do.nature_strategies:
        a = convert_to_a(nature_strategy, do.param_int)
        print('   ', np.round(a, 3))

    print()
    print('payoffs (regret)', np.array(do.payoffs).shape)
    regret = np.array(do.payoffs) - np.array(do.payoffs).max(axis=0)
    for p in regret:
        print('   ', np.round(p, 2))


    ##########################################
    # compare and visualize
    ##########################################

    baseline_middle_regrets = np.empty(n_perturb+1)
    baseline_middle_regrets[:] = np.nan
    for i in range(n_perturb+1):
        baseline_middle_distrib = np.zeros(len(do.agent_strategies))
        baseline_middle_distrib[baseline_middle_i+i] = 1
        baseline_middle_regrets[i] = do.compute_payoff_regret(baseline_middle_distrib)
    baseline_middle_regret = np.min(baseline_middle_regrets)
    print('avg regret of baseline middle {:.3f}'.format(baseline_middle_regret))

    baseline_random_regrets = np.empty(n_perturb+1)
    baseline_random_regrets[:] = np.nan
    for i in range(n_perturb+1):
        baseline_random_distrib = np.zeros(len(do.agent_strategies))
        baseline_random_distrib[baseline_random_i+i] = 1
        baseline_random_regrets[i] = do.compute_payoff_regret(baseline_random_distrib)
    baseline_random_regret = np.min(baseline_random_regrets)
    print('avg regret of baseline random {:.3f}'.format(baseline_random_regret))

    baseline_maximin_regrets = np.empty(n_perturb+1)
    baseline_maximin_regrets[:] = np.nan
    for i in range(n_perturb+1):
        baseline_maximin_distrib = np.zeros(len(do.agent_strategies))
        baseline_maximin_distrib[baseline_maximin_i+i] = 1
        baseline_maximin_regrets[i] = do.compute_payoff_regret(baseline_maximin_distrib)
    baseline_maximin_regret = np.min(baseline_maximin_regrets)
    print('avg regret of baseline maximin {:.3f}'.format(baseline_maximin_regret))

    baseline_RARL_regret_regrets = np.empty(n_perturb+1)
    baseline_RARL_regret_regrets[:] = np.nan
    for i in range(n_perturb+1):
        baseline_RARL_regret_distrib = np.zeros(len(do.agent_strategies))
        baseline_RARL_regret_distrib[baseline_RARL_regret_i+i] = 1
        baseline_RARL_regret_regrets[i] = do.compute_payoff_regret(baseline_RARL_regret_distrib)
    baseline_RARL_regret_regret = np.min(baseline_RARL_regret_regrets)
    print('avg regret of baseline RARL_regret {:.3f}'.format(baseline_RARL_regret_regret))

    # optimal reward of nature strategy
    do_regret = -get_payoff(regret, agent_eq, nature_eq)
    print('avg regret of DO {:.3f}'.format(do_regret))


    print('max_epochs {}, n_train agent {}, nature {}'.format(max_epochs, agent_n_train, nature_n_train))
    print('n_targets {}, horizon {}, budget {}'.format(do.n_targets, horizon, budget))

    bar_vals = [do_regret, baseline_middle_regret, baseline_random_regret, baseline_maximin_regret, baseline_RARL_regret_regret]
    tick_names = ('double oracle', 'baseline middle', 'baseline random', 'baseline maximin', 'baseline RARL regret')

    print('regrets', tick_names)
    print(np.round(bar_vals, 3))

    now = datetime.now()
    str_time = now.strftime('%d-%m-%Y_%H:%M:%S')

    filename = '{}double_oracle.csv'.format(prefix)
    with open(filename, 'a') as f:
        if f.tell() == 0:
            print('creating file {}'.format(filename))
            f.write(("seed, n_targets, budget, horizon, do_regret,"
            "baseline_middle_regret, baseline_random_regret, baseline_maximin_regret, baseline_RARL_regret_regret"
            "n_eval, agent_n_train, nature_n_train, max_epochs, n_perturb,"
            "max_interval, wildlife, deterrence, use_wake,"
            "freeze_policy_step, freeze_a_step, middle_time, maximin_time, do_time,"
            "time\n"))
        f.write((f"{seed}, {do.n_targets}, {budget}, {horizon}, {do_regret:.5f},"
        f"{baseline_middle_regret:.5f}, {baseline_random_regret:.5f}, {baseline_maximin_regret:.5f}, {baseline_RARL_regret_regret:.5f},"
        f"{n_eval}, {agent_n_train}, {nature_n_train}, {max_epochs}, {n_perturb},"
        f"{max_interval}, {wildlife_setting}, {deterrence_setting}, {use_wake},"
        f"{freeze_policy_step}, {freeze_a_step}, {middle_time}, {maximin_time}, {do_time},"
        f"{str_time}") +
        '\n')

    x = np.arange(len(bar_vals))
    plt.figure()
    plt.bar(x, bar_vals)
    plt.xticks(x, tick_names)
    plt.xlabel('method')
    plt.ylabel('avg regret')
    plt.title('n_targets {}, budget {}, horizon {}, max_epochs {}'.format(do.n_targets, budget, horizon, max_epochs))
    plt.savefig('plots/regret_n{}_b{}_h{}_epoch{}_{}.png'.format(do.n_targets, budget, horizon, max_epochs, str_time))
    # plt.show()
