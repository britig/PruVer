import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardActorNN(nn.Module):
    def __init__(self,in_dim,out_dim,is_discrete):
        #torch.manual_seed(0)
        super(FeedForwardActorNN, self).__init__()

        # Increasing connections for pruning
        self.layer1 = nn.Linear(in_dim,128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_dim)
        self.is_discrete = is_discrete
        print(self.is_discrete)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)
        
        # print(self.layer1(obs))

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        if self.is_discrete:
            output = torch.softmax(self.layer3(activation2),dim=0) #For catpole environment
        else:
            output = self.layer3(activation2)

        return output

'''class FeedForwardModifiedNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(FeedForwardModifiedNN, self).__init__()

        # Creating a concatenated network
        self.layer1 = nn.Linear(in_dim,128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_dim)
        self.layer4 = nn.Linear(in_dim+1,64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, in_dim)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = torch.argmax(self.layer3(activation2))'''





class FeedForwardCriticNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(FeedForwardCriticNN, self).__init__()

        self.layer1 = nn.Linear(in_dim,128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_dim)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

class FeedForwardState(nn.Module):
    def __init__(self,in_dim,out_dim):
        #torch.manual_seed(0)
        super(FeedForwardState, self).__init__()

        self.layer1 = nn.Linear(in_dim,64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self,obs):
        if isinstance(obs,np.ndarray):
            obs = torch.tensor(obs,dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output