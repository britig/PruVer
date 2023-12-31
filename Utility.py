"""
	Code for distance measurement between old policy and updated policy
	Project : Policy correction using Bayesian Optimization
	Description : The file contains utility functions for distance calculations
"""

import sys
import torch
from network import FeedForwardActorNN,FeedForwardState
import numpy as np
import gym
import gym_cartpole_swingup
from eval_policy import choose_best_action


'''
	Compute distance between the old policy and the updated policy 

	Parameters:
				policy_old (string), policy_new (string), env (open ai gym environment)

	Return:
				Distance metric : Dv(πold||πnew) =1n∑ξi,ξ′i∈ξ12∑ai∈ξi,a′i∈ξ′i|πold(si|ai)−πnew(s′i|a′i)|
'''
def compute_distance(policy_old,policy_new,env,is_discrete):
	#Load the policies
	if policy_old == '':
		print(f"Didn't specify old model file. Exiting.", flush=True)
		sys.exit(0)
	if policy_new == '':
		print(f"Didn't specify new model file. Exiting.", flush=True)
		sys.exit(0)
	obs_dim = env.observation_space.shape[0]
	if is_discrete:
		act_dim = env.action_space.n #env.action_space.shape[0]
	else:
		act_dim = env.action_space.shape[0]
	policy_old = FeedForwardActorNN(obs_dim, act_dim,is_discrete)
	policy_new = FeedForwardActorNN(obs_dim, act_dim,is_discrete)
	trajectory_set_old, trajectory_set_new = collect_trajectories(policy_old,policy_new,env,is_discrete)
	distance = 0
	# One trajectory can be greater than the other
	# traj = {s0,s1.........,s_n}
	for i in range(len(trajectory_set_old)):
		traj_old_i = trajectory_set_old[i]
		traj_new_i = trajectory_set_new[i]
		#print(f'traj_old_i ======== {traj_old_i}')
		#print(f'traj_old_i ======== {traj_new_i}')
		traj_len = min(len(trajectory_set_old[i]),len(trajectory_set_new[i]))
		sub_distance = 0
		for j in range(traj_len):
			sub_distance += (traj_old_i[j]-traj_new_i[j])**2
		#Account for the extra states if any
		if(len(trajectory_set_old[i]) > len(trajectory_set_new[i])):
			for k in range(len(trajectory_set_old[i])):
				sub_distance += (traj_old_i[k]-0)**2
		else:
			for k in range(len(trajectory_set_new[i])):
				sub_distance += (traj_new_i[k]-0)**2


		#print(f'sub_distance =========== {sub_distance}')
		sub_distance = np.sqrt(sub_distance)/2
		distance+= sub_distance
	distance = distance/len(trajectory_set_old)
	distance = sum(distance)/len(distance)
	print(f'distance ========== {distance}')
	return distance


'''
	Collect trajectories for old policy and the updated policy 

	Parameters:
				policy_old (string), policy_new (string), env (open ai gym environment)

	Return:
				trajectory_set_old
'''
def collect_trajectories(policy_old,policy_new,env,is_discrete):
	# Collect n random trajectories let n=1000 for our experiments 
	trajectory_set_old = []
	trajectory_set_new = []
	t = 0
	print(f'Collecting trajectories for both the policies')
	while t<1000:
		obs_old = env.reset()
		env_state = env
		obs_new = obs_old
		episode_observation_old = []
		episode_observation_new = []
		done_old = False
		done_new = False
		while not done_old:
			#Collecting trajectories for the old policy
			if is_discrete:
				action_old = choose_best_action(obs_old, policy_old) #policy_old(obs_old).detach().numpy()
			else:
				action_old = policy_old(obs_old).detach().numpy()
			obs_old, rew_old, done_old, _ = env.step(action_old)
			episode_observation_old.append(obs_old)

		while not done_new:
			#Collecting trajectories for the updated policy
			if is_discrete:
				action_new = choose_best_action(obs_new, policy_new) #policy_new(obs_new).detach().numpy()
			else:
				action_new = policy_new(obs_new).detach().numpy()
			obs_new, rew_new, done_new, _ = env_state.step(action_new)
			episode_observation_new.append(obs_new)
	
		trajectory_set_old.append(episode_observation_old)
		trajectory_set_new.append(episode_observation_new)
		t = t+1
		#print(f'trajectory_set_old ======== {trajectory_set_old}')

	return trajectory_set_old, trajectory_set_new


'''
	set a particular gym environment

	Parameters:
				Environment name

	Return:
				instance of openai gym environment
'''
def set_environment(env_name,seed):
	env = gym.make(env_name)
	env.seed(seed)
	return env

def load_model(actor_model,env,isdescrete):
	print(actor_model)
	if 'state' in actor_model:
		obs_dim = env.observation_space.shape[0]+1
		act_dim = env.observation_space.shape[0]
		policy = FeedForwardState(obs_dim,act_dim)
	else:
		obs_dim = env.observation_space.shape[0]
	if isdescrete:
		act_dim = env.action_space.n
	else:
		act_dim = env.action_space.shape[0]
	policy = FeedForwardActorNN(obs_dim,act_dim,isdescrete)
	policy.load_state_dict(torch.load(actor_model))
	return policy

#Function to Convert to ONNX 
def Convert_ONNX(actor_model,env,isdescrete): 
	model = load_model(actor_model,env,isdescrete)
	# set the model to inference mode 
	model.eval() 
	if 'state' in actor_model:
		input_size = env.observation_space.shape[0]+1
	else:
		input_size = env.observation_space.shape[0]

	# Let's create a dummy input tensor  
	dummy_input = torch.randn(1, input_size, requires_grad=True)  
	print(dummy_input)
	onxx_model_name = actor_model.split('.pth')[0]+".onxx"

	# Export the model   
	torch.onnx.export(model,         # model being run 
		 dummy_input,       # model input (or a tuple for multiple inputs) 
		 onxx_model_name,       # where to save the model  
		 export_params=True,  # store the trained parameter weights inside the model file 
		 opset_version=10,    # the ONNX version to export the model to 
		 do_constant_folding=True,  # whether to execute constant folding for optimization 
		 input_names = ['modelInput'],   # the model's input names 
		 output_names = ['modelOutput'], # the model's output names 
		 dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
								'modelOutput' : {0 : 'batch_size'}}) 
	print(" ") 
	print('Model has been converted to ONNX') 
