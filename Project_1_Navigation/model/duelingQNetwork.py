import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable
from torchsummary import summary

class DuelingQNetwork(nn.Module):
    """ Dueling Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128, fc3_units=64, fc4_units=32):
        super(DuelingQNetwork, self).__init__()
        
        # store the action size and state size
        self.state_size = state_size
        self.action_size = action_size
        
        # create the network - 3 layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        
        self.AL = nn.Linear(fc4_units, action_size)
        nn.init.xavier_uniform_(self.AL.weight)
        
        self.V = nn.Linear(fc4_units, 1)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, state):
    
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = F.relu(self.fc4(action))
        value = self.V(action)
        af = self.AL(action)
        action = value + (af - torch.mean(af))
        return action
    
    
    def show_model_info(self):
        # Just some code to visualize the neural network 
        
        device = torch.device("cpu") # PyTorch v0.4.0
        
        # create a summary
        model = DuelingQNetwork(self.state_size, self.action_size,0).to(device)
        summary(model, (1, self.state_size))
        
        # Print the graph
        input_v = torch.randn(1, self.state_size)
        y = model(Variable(input_v))
        return make_dot(y)