# MIRROR: Robust Reinforcement Learning Under Minimax Regret

This code implements and evaluates algorithms for ["Robust Reinforcement Learning Under Minimax Regret for Green Security"](https://arxiv.org/abs/2106.08413) from UAI-21, including the MIRROR algorithm provided in the paper.

```
@inproceedings{xu2021robust,
  title={Robust Reinforcement Learning Under Minimax Regret for Green Security},
  author={Xu, Lily and Perrault, Andrew and Fang, Fei and Chen, Haipeng and Tambe, Milind},
  booktitle={Proc.~37th Conference on Uncertainty in Artifical Intelligence (UAI-21)},
  year={2021},
}
```

This project is licensed under the terms of the MIT license.

Due to the sensitive nature of poaching data, we provide dummy data for the park simulator rather than the real-world poacher behavioral data used in the paper experiments.


## Usage

To run one complete execution of MIRROR to learn an optimal agent strategy (defender policy) with default settings, execute:
```sh
python double_oracle.py
```

To vary the settings, use the options:
```sh
python double_oracle.py --seed 0 --n_eval 100 --agent_train 100 --nature_train 100 --max_epochs 5 --n_perturb 3 --wake 1 --freeze_policy_step 5 --freeze_a_step 5 --height 5 --width 5 --budget 5 --horizon 5 --interval 3 --wildlife 1 --deterrence 1 --prefix ''
```

The options used to configure the wildlife park (varied settings in Figure 4) are
- `horizon` - horizon for planning patrols, `H`
- `height`, `width` - set the size of the park. height x width = `N` in the paper
- `budget` - budget for ranger resources, `B` in the paper
- `interval` - uncertainty interval size
- `deterrence` - deterrence strength, `beta`
- `wildlife` - initial wildlife distribution

The options used to configure the MIRROR algorithm (including the RL oracles) are
- `seed` - random seed
- `n_eval` - number of timesteps to run to evaluate average reward
- `agent_train` - number of iterations to train agent DDPG
- `nature_train` - number of iterations to train nature DDPG, `J` in Algorithm 2
- `max_epochs` - number of epochs to run MIRROR
- `n_perturb` - number of perturbations, `O` in Algorithm 1
- `wake` - (binary) whether to use wake/sleep
- `freeze_policy_step` - how often to freeze policy parameters `kappa` in Algorithm 2
- `freeze_a_step` - number of steps before unfreezing attractiveness parameters, `kappa` in Algorithm 2

## Files

- `double_oracle.py` executes the whole MIRROR process
- `agent_oracle.py` has implementation of the agent RL oracle (to learn agent best response in response to nature mixed strategy)
- `nature_oracle.py` has implementation of the nature RL oracle (to learn attractiveness and alternate policy in response to agent mixed strategy)
- `park.py` implements the park environment described in Section 3.2
- `ddpg.py` implements deep deterministic policy gradient, used by the agent oracle
- `ddpg_nature.py` implements the version of DDPG used by the nature oracle
- `min_reward_oracle.py` implements an oracle used by the RARL baseline
- `nfg_solver.py` normal-form game solver used by double oracle to solve for equilibria

## Requirements

- python 3.6
- pytorch 1.0.1
- matplotlib 3.2.2
- numpy 1.15.3
- pandas 1.0.5
- scikit-learn 0.23.2
- scipy 1.5.3
- nashpy 0.0.19
