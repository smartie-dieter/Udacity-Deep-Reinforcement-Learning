import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable
from torchsummary import summary

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, fc1_units,) # 37 --> 64 first layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)   # 64 --> 32 Second layer
        self.fc3 = nn.Linear(fc2_units, action_size) # 32 -->  4 ouput layer

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    
    def show_model_info(self):
        # Just some code to visualize the neural network 
        
        device = torch.device("cpu") # PyTorch v0.4.0
        
        # create a summary
        model = QNetwork(self.state_size, self.action_size,0).to(device)
        summary(model, (1, self.state_size))
        
        # Print the graph
        input_v = torch.randn(1, self.state_size)
        y = model(Variable(input_v))
        return make_dot(y)