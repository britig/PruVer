"""
    Code for collecting failure trajectories using Bayesian Optimization for environment Lunar Lander
    Project : Policy correction using Bayesian Optimization
    Description : The file contains functions for computing failure trajectories given RL policy and
    safety specifications
"""

import numpy as np
import gym
from numpy.random import seed
from eval_policy import display
import gym
from network import FeedForwardActorNN
import torch
import pickle
import time
import torch.nn as nn
from scipy.stats import rankdata

'''
    Bayesian Optimization module for uncovering failure trajectories

    Safety Requirement
    # Requirement 1: The lander should not fall down in any trajectory
'''

#=============================================Global Variables =================================#
policy = None
env = None
traj_spec_dic = {}
traj_count = 0
index_count = 0

# Method to calculate the recovery state for the failure trajectories
# Takes the optimized policy as input and calculates the last state where the optimized policy can correct the trajectory
def find_recovery(policy,fail_traj):
    # Failure observations
    set_recovery = []
    for i in range(len(fail_traj)):
        seed = fail_traj[i][2]
        # Trajectory observations
        for j in range(len(fail_traj[i][0])):
            env.seed(seed[0])
            env.reset()
            env.env.state = fail_traj[i][0][0]
            # set the environment by execution actions
            obs = env.env.state
            ep_rew = 0
            done = False
            for k in range(j):
                obs_prev = obs
                obs, rew, done, _ = env.step(fail_traj[i][3][k])
                ep_rew = ep_rew+rew
            while not done:
                obs_prev = obs
                #env.render()
                #time.sleep(0.01)
                action = policy(obs).detach().numpy() 
                obs, rew, done, _ = env.step(action)
                ep_rew = ep_rew+rew
            #print(f'episode reward ========== {ep_rew}')
            if ep_rew < 200:
                saddle_obs = obs_prev
                print(f'saddle_obs ========== {saddle_obs}')
                break
        #break
        set_recovery.append(saddle_obs)
    print(f'set_recovery ========== {set_recovery}')
    return set_recovery

def find_actions(set_recovery,policy_old,policy_compressed):
    set_recovery = torch.tensor(set_recovery, dtype=torch.float)
    action_batch_old = policy_old(set_recovery)
    action_batch_comp= policy_compressed(set_recovery)
    '''for s in set_recovery:
        action_old = policy_old(s) 
        action_comp = policy_compressed(s) 
        action_batch_old.append(action_old)
        action_batch_comp.append(action_comp)'''
    return action_batch_old,action_batch_comp

def update_pruning(policy,policy_compressed,fail_traj,action_batch_old,action_batch_comp):
    p = 0.95
    loss = nn.MSELoss()(action_batch_comp, action_batch_old)
    loss.backward()
    ranks = {}
    data_old = {}
    data_compressed = {}
    weight_grad = {}
    weight_grad['layer1'] = policy_compressed.layer1.weight.grad
    weight_grad['layer2'] = policy_compressed.layer2.weight.grad
    weight_grad['layer3'] = policy_compressed.layer3.weight.grad
    data_old['layer1'] = policy.layer1.weight.detach().numpy()
    data_old['layer2'] = policy.layer2.weight.detach().numpy()
    data_old['layer3'] = policy.layer3.weight.detach().numpy()
    data_compressed['layer1'] = policy_compressed.layer1.weight.detach().numpy()
    data_compressed['layer2'] = policy_compressed.layer2.weight.detach().numpy()
    data_compressed['layer3'] = policy_compressed.layer3.weight.detach().numpy()
    ranks['layer1']=(rankdata(np.abs(weight_grad['layer1']),method='dense') - 1).astype(int).reshape(weight_grad['layer1'].shape)
    ranks['layer2']=(rankdata(np.abs(weight_grad['layer2']),method='dense') - 1).astype(int).reshape(weight_grad['layer2'].shape)
    ranks['layer3']=(rankdata(np.abs(weight_grad['layer3']),method='dense') - 1).astype(int).reshape(weight_grad['layer3'].shape)
    while p>=0:    
        lower_bound_rank_layer1 = np.ceil(np.max(ranks['layer1'])*p).astype(int)
        lower_bound_rank_layer2 = np.ceil(np.max(ranks['layer2'])*p).astype(int)
        lower_bound_rank_layer3 = np.ceil(np.max(ranks['layer3'])*p).astype(int)
        index_rank_layer1 = np.where(ranks['layer1'] >= lower_bound_rank_layer1)
        index_rank_layer2 = np.where(ranks['layer2'] >= lower_bound_rank_layer2)
        index_rank_layer3 = np.where(ranks['layer3'] >= lower_bound_rank_layer3)
        for i in range(len(index_rank_layer1[0])):
            data_compressed['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]] = data_old['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]]
        policy_compressed.__dict__['_modules']['layer1'].weight = torch.nn.Parameter(torch.FloatTensor(data_compressed['layer1']))
        flag = check_corrections(policy_compressed,fail_traj)
        if flag:
            return policy_compressed
        for i in range(len(index_rank_layer2[0])):
            data_compressed['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]] = data_old['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]]
        policy_compressed.__dict__['_modules']['layer2'].weight = torch.nn.Parameter(torch.FloatTensor(data_compressed['layer2']))
        flag = check_corrections(policy_compressed,fail_traj)
        if flag:
            return policy_compressed
        for i in range(len(index_rank_layer3[0])):
            data_compressed['layer3'][index_rank_layer3[0][i]][index_rank_layer3[1][i]] = data_old['layer3'][index_rank_layer3[0][i]][index_rank_layer3[1][i]]
        policy_compressed.__dict__['_modules']['layer3'].weight = torch.nn.Parameter(torch.FloatTensor(data_compressed['layer3']))
        flag = check_corrections(policy_compressed,fail_traj)
        if flag:
            return policy_compressed
        print(f'flag ======== {flag} ======= p ======= {p}')
        p = p - 0.02

    return policy_compressed

def check_corrections(policy_compressed,fail_traj):
    flag = False
    for i in range(len(fail_traj)):
        seed = fail_traj[i][2]
        env.seed(seed[0])
        env.reset()
        env.env.state = fail_traj[i][0][0]
        obs = env.env.state
        ep_rew = 0
        done = False
        while not done:
            env.render()
            #time.sleep(0.01)
            action = policy_compressed(obs).detach().numpy() 
            obs, rew, done, _ = env.step(action)
            ep_rew = ep_rew+rew
        print(ep_rew)
        if ep_rew<100:
            return False
    return True




def analysing_weights(policy,policy_compressed,fail_traj):
    obs = np.array([0.19149475,-0.03878709,-0.33019605,-0.04373148,-0.19916481,0.15565039,1.,1.])
    action_old = policy(obs) 
    action_comp = policy_comp(obs)
    print(f'action_old ========== {action_old} ======== action_comp ======= {action_comp}')
    k = 0.95
    diff = nn.MSELoss()(action_comp, action_old)
    diff.backward()
    ranks = {}
    data_old = {}
    weight_grad = {}
    #print(policy_compressed.layer3.weight.grad)
    weight_grad['layer1'] = policy_compressed.layer1.weight.grad
    weight_grad['layer2'] = policy_compressed.layer2.weight.grad
    data_old = {}
    data_compressed = {}
    data_old['layer1'] = policy.layer1.weight.detach().numpy()
    data_old['layer2'] = policy.layer2.weight.detach().numpy()
    data_compressed['layer1'] = policy_compressed.layer1.weight.detach().numpy()
    data_compressed['layer2'] = policy_compressed.layer2.weight.detach().numpy()
    ranks['layer1']=(rankdata(np.abs(weight_grad['layer1']),method='dense') - 1).astype(int).reshape(weight_grad['layer1'].shape)
    ranks['layer2']=(rankdata(np.abs(weight_grad['layer2']),method='dense') - 1).astype(int).reshape(weight_grad['layer2'].shape)
    lower_bound_rank_layer1 = np.ceil(np.max(ranks['layer1'])*k).astype(int)
    lower_bound_rank_layer2 = np.ceil(np.max(ranks['layer2'])*k).astype(int)
    index_rank_layer1 = np.where(ranks['layer1'] >= lower_bound_rank_layer1)
    index_rank_layer2 = np.where(ranks['layer2'] >= lower_bound_rank_layer2)
    print(index_rank_layer1)
    for i in range(len(index_rank_layer1[0])):
        #print(data_old['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]])
        #print(data_compressed['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]])
        data_compressed['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]] = data_old['layer1'][index_rank_layer1[0][i]][index_rank_layer1[1][i]]
    for i in range(len(index_rank_layer2[0])):
        #print(data_old['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]])
        #print(data_compressed['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]])
        #pass
        data_compressed['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]] = data_old['layer2'][index_rank_layer2[0][i]][index_rank_layer2[1][i]]
    # Refine Step
    policy_compressed.__dict__['_modules']['layer1'].weight = torch.nn.Parameter(torch.FloatTensor(data_compressed['layer1']))
    policy_compressed.__dict__['_modules']['layer2'].weight = torch.nn.Parameter(torch.FloatTensor(data_compressed['layer2']))
    y_comp_new = policy_compressed(obs)
    print(f'action_new ========== {y_comp_new}')
    seed = fail_traj[0][2]
    env.seed(seed[0])
    env.reset()
    env.env.state = fail_traj[0][0][0]
    ep_rew = 0
    done = False
    while not done:
        env.render()
        #time.sleep(0.01)
        action = policy_compressed(obs).detach().numpy() 
        obs, rew, done, _ = env.step(action)
        ep_rew = ep_rew+rew
    print(f'ep_rew ========== {ep_rew}')





if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    env_name = "BipedalWalker-v3"
    actor_model = 'ppo_actorBipedalWalker-v3.pth'
    actor_model_comp = 'ppo_actorBipedalWalker-v3_0.5.pth'
    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    failure_trajectory = 'counterexample_trajectory_bipedal.data'
    with open(failure_trajectory, 'rb') as filehandle:
        # read env_state
        failure_trajectories = pickle.load(filehandle)

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActorNN(obs_dim, act_dim,False)
    policy_comp = FeedForwardActorNN(obs_dim, act_dim,False)
    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))
    policy_comp.load_state_dict(torch.load(actor_model_comp))
    # find_saddle(policy,failure_trajectories)
    set_recovery = find_recovery(policy,failure_trajectories)
    action_batch_old,action_batch_comp = find_actions(set_recovery,policy,policy_comp)
    policy_compressed = update_pruning(policy,policy_comp,failure_trajectories,action_batch_old,action_batch_comp)
    torch.save(policy_compressed.state_dict(), './ppo_actor_refined'+env_name+'.pth')
    #actor_model_refnd = 'ppo_actor_refinedBipedalWalker-v3.pth'
    #policy_refnd = FeedForwardActorNN(obs_dim, act_dim,False)
    #policy_refnd.load_state_dict(torch.load(actor_model_refnd))
    #analysing_weights(policy,policy_comp,failure_trajectories)
    #check_corrections(policy_refnd,failure_trajectories)
