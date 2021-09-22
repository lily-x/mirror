""" oracle to minimize reward
used for robust adversarial RL approach

Lily Xu, 2021
"""

import sys, os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from park import Park, convert_to_a

def smooth(x, window_len=11, window='hanning'):
    assert x.ndim == 1, "smooth only accepts 1 dimension arrays."
    assert x.size >= window_len, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'], "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def min_reward(park_params, def_strategy, attractiveness_learning_rate=5e-2,
        n_iter=400, batch_size=64, visualize=False, init_attractiveness=None):
    """
    given a defender strategy, learn updated attractiveness parameters to minimize reward
    """

    if init_attractiveness is None:
        # initialize with random attractiveness values in interval
        attractiveness = (np.random.rand(park_params['n_targets']) - .5) * 2
        attractiveness = attractiveness.astype(float)
    else:
        attractiveness = init_attractiveness.copy()

    print('attractiveness (raw)', np.round(attractiveness, 3))
    print('attractiveness (convert)', np.round(convert_to_a(attractiveness, park_params['param_int']), 3))

    attractiveness = torch.tensor(attractiveness, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([attractiveness], lr=attractiveness_learning_rate)

    print('initial {}'.format(attractiveness))

    env = Park(attractiveness, park_params['initial_effort'], park_params['initial_wildlife'], park_params['initial_attack'],
        park_params['height'], park_params['width'], park_params['n_targets'], park_params['budget'], park_params['horizon'],
        park_params['psi'], park_params['alpha'], park_params['beta'], park_params['eta'], param_int=park_params['param_int'])

    state = env.reset()

    all_r = []

    for t in range(n_iter):
        rewards = []
        for i in range(batch_size):
            action = def_strategy.select_action(state)
            action = torch.FloatTensor(action)
            next_state, reward, done, info = env.step(action, use_torch=True)
            rewards.append(info['expected_reward'])

            if done:
                state = env.reset()

        loss = torch.sum(torch.stack(rewards))

        if t % 100 == 99:
            print('{} {:.2f} {}'.format(t, loss.item(), np.round(convert_to_a(attractiveness.detach().numpy(), park_params['param_int']), 3)))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        state = next_state

        all_r.append(loss)

    all_r = np.array(all_r)  / batch_size

    if visualize:
        plt.plot(smooth(all_r, window_len=40))
        plt.show()

    return attractiveness.detach().numpy()
