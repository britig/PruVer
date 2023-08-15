"""
	The main file to run the policy correction code
"""

from ppoPolicyTraining import PPO, test
from eval_policy import display
import numpy as np
#Update failure network
from Utility import set_environment, load_model,Convert_ONNX
import argparse
import yaml
import pickle
from prune import prune_test
from visualize import visualize_model_weight


if __name__ == '__main__':
	#=============================== Environment and Hyperparameter Configuration Start ================================#
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--env', dest='env', action='store_true', help='environment_name')
	parser.add_argument('--train', dest='train', action='store_true', help='train model')
	parser.add_argument('--train_state', dest='train_state', action='store_true', help='train state estmator model')
	parser.add_argument('--test', dest='test', action='store_true', help='test model')
	parser.add_argument('--convert', dest='convert', action='store_true', help='Convert a model to onxx')
	parser.add_argument('--display', dest='display', action='store_true', help='Display Failure Trajectories')
	parser.add_argument('--isdiscrete', dest='isdiscrete', action='store_true', help='Whether model discrete or continuous')
	parser.add_argument('--prune', dest='prune', action='store_true', help='Pruning the model')
	parser.add_argument('--k', dest='k', action='store_true', help='Pruning Percentage')
	parser.add_argument('--visualize', dest='visualize', action='store_true', help='Visualizing the weight of the model')
	args = parser.parse_args()
	actor_model = None
	critic_model = None
	failure_trajectory = None
	env_name = None
	is_discrete = False
	old_actor = None
	old_critic = None
	sub_actor = None
	sub_critic = None
	new_actor = None
	if args.env:
		env_name = args.env
	else:
		env_name = "LunarLanderContinuous-v2"
	if args.isdiscrete:
		is_discrete = args.isdiscrete

	if args.k:
		k = args.k
	else:
		k = 0.70

	actor_model = 'ppo_actor_refinedLunarLanderContinuous-v2.pth'
	state_estimator_model = 'state_estimatorCartPoleSwingUp-v0.pth'
	failure_trajectory = 'Failure_Trajectories/counterexample_trajectory_lunar_lander.data'

	env = set_environment(env_name,0)
	#policy = load_model(actor_model,env,is_discrete)
	print(env.action_space)
	with open('hyperparameters.yml') as file:
		paramdoc = yaml.full_load(file)
	#=============================== Environment and Hyperparameter Configuration End ================================#
	#=============================== Original Policy Training Code Start ================================#
	if args.train:
		for item, param in paramdoc.items():
			if(str(item)==env_name):
				hyperparameters = param
				print(param)
		model = PPO(env=env, **hyperparameters)
		model.learn(env_name, [], is_discrete)
	#=============================== Original Policy Training Code End ================================#
	#=============================== Policy Testing Code Start ==========================#
	if args.test:
		test(env,actor_model, is_discrete)
	#=============================== Policy Testing Code End ============================#
	#=============================== Policy Pruning Code Start ==========================#
	if args.prune:
		prune_test(env,actor_model,is_discrete,k)
	#=============================== Policy Pruning Code End ==========================#
	#=============================== Visualization of Pruned model Code Start ==========================#
	if args.visualize:
		policy = load_model(actor_model,env,is_discrete)
		visualize_model_weight(policy)
	#=============================== Visualization of Pruned model Code End ==========================#
	#=============================== Displaying Failure Trajectories Code Start  ==========================#
	if args.display:
		policy = load_model(actor_model,env,is_discrete)
		with open(failure_trajectory, 'rb') as filehandle1:
			# read env_state
			failure_observations = pickle.load(filehandle1)
		print(f'Number of failure trajectories=========={len(failure_observations)}')
		if env_name == 'LunarLanderContinuous-v2':
			for i in range(len(failure_observations)):
				seed = failure_observations[i][2]
				#print(seed)
				env.seed(seed[0])
				env.reset()
				#env.env.state = failure_observations[i][0][0]
				display(failure_observations[i][0][0],policy,env,False)
	#=============================== Displaying Failure Trajectories Code End  ==========================#
	#=============================== Converting Model to ONXX format Code Start ==========================#
	if args.convert:
		Convert_ONNX(state_estimator_model,env,is_discrete)
	#=============================== Converting Model to ONXX format Code End ==========================#

	


