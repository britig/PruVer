# Safety Aware Neural Pruning for Deep Reinforcement Learning
This repo contains the experiments and code for Safety Aware Neural Pruning for Deep Reinforcement Learning 
# System Requirements

The code has been tested in systems with the following OS

- Ubuntu 20.04.2 LTS

## Installation

1. Setup conda environment

```
$ conda create -n env_name python=3.8.5
$ conda activate env_name
```
2. Clone the repository to an appropriate folder
3. Navigate to Code For Neural Pruning folder and Install requirements

```
$ pip install -r requirements.txt
$ pip install -e .
```

4. All code should be run from Code For Neural Pruning folder. The output files (policies and failure trajectory files are also saved inside this folder).

## Usage

All the trained policies, pruned-policies and refined policies are avialable in the Policies folder


The main program takes the following command line arguments

1) --env : environment name (default is LunarLanderContinuous-v2)
2) --actor : filepath to the actor network (default is Policies/ppo_actorLunarLanderContinuous-v2.pth)
3) --isdiscrete : True if environment is discrete (default False)
4) --k : The percentage of one shot pruning (default 0.70)

The hyperparameters can be changed in the hyperparameters.yml file


Note : Change the default arguments inside the main.py file otherwise the command line may become too long


### Testing

To test a trained model run:

```
$ python main.py --test
```

Press ctr+c to end testing

### Pruning a policy

For pruning a trained policy run:

```
$ python main.py --prune
```
Mention the actor policy and the k value (default 0.7) in arguments or in the main.py file

### Generating Failure trajectories for a specific environment

Failure trajectories uncovered with our tests are available in Failure_Trajectories Folder

Each environment has a seperate Testing file. Run the Testing correspondig to the environment
We use GpyOpt Library for Bayesian Optimization. As per (https://github.com/SheffieldML/GPyOpt/issues/337) GpyOpt has stochastic evaluations even when the seed is fixed.
This may lead to identification of a different number failure trajectories (higher or lower) than the mean number of trajectories reported in the paper.

For example to generate failure trajectories for the Lunar Lander environment run:

```
$ python LunarLanderTesting.py
```

The failure trajectories will be written in the corresponding data files in the same folder


### Refining the pruned policy

Each environment has a seperate refinement file. Run the refinement file correspondig to the environment

```
$ python LunarLanderRefinement.py
```

This file should contain
1) The original dense network actor model (ex : Policies/ppo_actorLunarLanderContinuous-v2.pth)
2) The reward pruned sparse network actor model (ex: Policies/ppo_actorLunarLanderContinuous-v2_0.7.pth)
3) The file containing counterexample trajectories (ex: Failure_Trajectories/counterexample_trajectory_lunar_lander.data)

The output is refined pruned network (ex : Policies/ppo_actor_refinedLunarLanderContinuous-v2.pth)


### Visualize the weights of the actor network

```
$ python main.py --visualize
```
default function parameters are:
1) --actor : filepath to the actor network (default is Policies/ppo_actorLunarLanderContinuous-v2.pth) ( change this to Policies/ppo_actor_refinedLunarLanderContinuous-v2.pth to visualize pruned network)
3) --env : environment name (default is LunarLanderContinuous-v2)
4) --isdiscrete : True if environment is discrete (default False)

Our plots are stored inside the Graph folder

### Training a policy from scratch

To train a model run:

```
$ python main.py --train
```
The hyperparameters can be changed in the hyperparameters.yml file
