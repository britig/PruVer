# Let us try with weight pruning first
import torch
import gym
from Utility import set_environment
from network import FeedForwardActorNN
from scipy.stats import rankdata
import numpy as np

def prune_test(env,actor_model,is_discrete,prune_percent):

    obs_dim = env.observation_space.shape[0]
    if is_discrete:
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]
    print(act_dim)

    policy = FeedForwardActorNN(obs_dim,act_dim,True)

    policy.load_state_dict(torch.load(actor_model))

    # Show layer wise weight

    # weights = list(policy.parameters())

    # print(weights)
    data = {}
    ranks = {}
    for name, param in policy.named_parameters():
        if 'weight' in name:
            print(f'Name========={name} ======= param ====== {param.detach().numpy().size}')
            name = name.split(".")[0]
            data[name] = param.detach().numpy()

    # First let us prune k% of the weights and test
    k = prune_percent

    # Prune 25% of the weights as their minimum value
    for key,val in data.items():
        ranks[key]=(rankdata(np.abs(val),method='dense') - 1).astype(int).reshape(val.shape)
        #print(ranks[key])
        lower_bound_rank = np.ceil(np.max(ranks[key])*k).astype(int)
        print(lower_bound_rank)
        # Make weights below lower bound rank
        ranks[key][ranks[key]<=lower_bound_rank] = 0
        ranks[key][ranks[key]>lower_bound_rank] = 1
        # print(ranks[key])
        data[key] = data[key] * ranks[key]
        #print(key)
        #print(policy.__dict__['_modules'][key].weight)
        with torch.no_grad():
            policy.__dict__['_modules'][key].weight = torch.nn.Parameter(torch.FloatTensor(data[key]))
        # print(policy.__dict__['_modules'][key].weight)
    model_name = actor_model.split(".")[0]
    torch.save(policy.state_dict(),model_name+'_'+str(k)+'.pth')
        


