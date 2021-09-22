""" normal-form game (NFG) solver

(1) Take a normal form game as input, find minimax regret optimal mixture.
(2) Take a normal form game as input, find an MNE.

Lily Xu, 2020
"""

import numpy as np
import sys
import nashpy as nash # Nash eq solver
from scipy.optimize import linprog

def solve_game(payoffs):
    """ given payoff matrix for a zero-sum normal-form game,
    return first mixed equilibrium (may be multiple)
    returns a tuple of numpy arrays """
    game = nash.Game(payoffs)
    equilibria = game.lemke_howson_enumeration()
    equilibrium = next(equilibria, None)

    # Lemke-Howson couldn't find equilibrium OR
    # Lemke-Howson return error - game may be degenerate. try other approaches
    if equilibrium is None or (equilibrium[0].shape != (payoffs.shape[0],) and equilibrium[1].shape != (payoffs.shape[0],)):
        # try other
        print('\n\n\n\n\nuh oh! degenerate solution')
        print('payoffs are\n', payoffs)
        equilibria = game.vertex_enumeration()
        equilibrium = next(equilibria)
        if equilibrium is None:
            print('\n\n\n\n\nuh oh x2! degenerate solution again!!')
            print('payoffs are\n', payoffs)
            equilibria = game.support_enumeration()
            equilibrium = next(equilibria)

    assert equilibrium is not None
    return equilibrium


def solve_minimax_regret(payoffs):
    """ given payoff matrix for a zero-sum normal-form game,
    return a minimax regret optimal mixture """
    # degenerate case: if num rows == 1, regret will be 0
    if len(payoffs) == 1:
        eq1 = [1.]
        eq2 = np.zeros(len(payoffs[0]))
        eq2[0] = 1.
        return eq1, eq2
    # subtract max from each column
    mod_payoffs = np.array(payoffs)
    print('payoffs before')
    print(np.round(mod_payoffs, 2))
    mod_payoffs = mod_payoffs - mod_payoffs.max(axis=0)
    print('payoffs after')
    print(np.round(mod_payoffs, 2))
    return solve_game(mod_payoffs)

def get_payoff(payoffs, agent_eq, nature_eq):
    """ given player mixed strategies, return expected payoff """

    game = nash.Game(payoffs)
    print('  payoff shape', game.payoff_matrices[0].shape)
    print('  agent eq', np.round(agent_eq, 2))
    print('  nature eq', np.round(nature_eq, 2))
    expected_utility = game[agent_eq, nature_eq]
    print('  expected_utility', expected_utility)
    return expected_utility[0]


def get_agent_best_strategy(payoffs, nature_eq):
    """ for a given a fixed nature mixed strategy, find the best mixed strategy
    for the agent to respond with """
    assert np.abs(np.sum(nature_eq) - 1) <= 1e-2

    num_nature_strategies = len(payoffs[0])
    num_agent_strategies = len(payoffs)

    assert num_nature_strategies == len(nature_eq)

    # if nature has only one strategy
    if np.argwhere(nature_eq).size == 1:
        nature_eq_idx = np.argwhere(nature_eq).item()
        best_agent_strategy = payoffs[nature_eq_idx, :].argmax()
        agent_eq = np.zeros(num_agent_strategies)
        agent_eq[best_agent_strategy] = 1
        return agent_eq
    c = nature_eq.dot(payoffs) # coefficients - average regret per nature strategy
    A = [[1]*num_agent_strategies]
    b = [1]
    res = linprog(-c, A_eq=A, b_eq=b, bounds=[(0,1)]*num_agent_strategies)
    assert res.success

    return res.x

def get_nature_best_strategy(payoffs, agent_eq):
    """ for a given a fixed agent mixed strategy, find the best mixed strategy
    for nature to respond with """
    assert np.abs(np.sum(agent_eq) - 1) <= 1e-2

    num_nature_strategies = len(payoffs[0])
    num_agent_strategies = len(payoffs)

    assert num_agent_strategies == len(agent_eq)

    # if agent has only one strategy
    if np.argwhere(agent_eq).size == 1:
        agent_eq_idx = np.argwhere(agent_eq).item()
        best_nature_strategy = payoffs[agent_eq_idx, :].argmax()
        nature_eq = np.zeros(num_nature_strategies)
        nature_eq[best_nature_strategy] = 1
        return nature_eq
    c = agent_eq.dot(payoffs) # coefficients - average regret per nature strategy
    A = [[1]*num_nature_strategies]
    b = [1]
    res = linprog(-c, A_eq=A, b_eq=b, bounds=[(0,1)]*num_nature_strategies)

    assert res.success

    return res.x



if __name__ == '__main__':
    payoffs = np.random.randint(0, 10, (2, 2))

    print(payoffs)

    equilibria = solve_game(payoffs)
    for eq in equilibria:
        print(eq)

    print('-----')
    A = np.array([[2, 3], [1, 4]])
    equilibria = solve_game(A)
    for eq in equilibria:
        print(eq)


    print('-----')
    B = np.array([[3/2, 3], [1, 4]])
    equilibria = solve_game(B)
    for eq in equilibria:
        print(eq)

    print('-----')
    C = np.array([[-2, 3], [3, -4]])
    equilibria = solve_game(C)
    for eq in equilibria:
        print(eq)
