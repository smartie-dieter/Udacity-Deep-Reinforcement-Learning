import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size * 2
        self.action_size = action_size * 2
        
        self.fc1 = nn.Linear(self.state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + self.action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
        
    def forward(self, state, action):
        """Build a Cretic (value) network that maps (state, action)  -->  Q-values."""
        xs = F.relu(self.fc1(state)) 
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)
