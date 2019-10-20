
from model.replayBuffer import ReplayBuffer
from model.duelingQNetwork import DuelingQNetwork
from model.qNetwork import QNetwork

from datetime import datetime
import numpy as np
import random 
import torch
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, prioritize_perc, importance_sampling_weight_perc, calculate_b_xth_percentage, calculate_b_value, delta_init, gamma, tau, lr, update_every, ddqn, dueling, episode= 1, epsilon_to_minimum_decay = 1000, b_value_to_minimum_decay = 1000, epsilon_init = 1, epsilon_max= 1., epsilon_min= 0.1):
        
        # Army of hyper parameters
        
        # Replay buffer
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.DELTA_INIT = delta_init
        self.prioritize_perc = prioritize_perc
        self.importance_sampling_weight_perc = importance_sampling_weight_perc
        self.importance_sampling_weight_decay = 0
        self.CALCULATE_B_XTH_PERCENTAGE = calculate_b_xth_percentage
        self.CALCULATE_B_XTH_VALUE = calculate_b_value
        self.calculate_b_deacy(b_value_to_minimum_decay, self.CALCULATE_B_XTH_PERCENTAGE, self.CALCULATE_B_XTH_VALUE)
        self.B_VALUE_TO_MINIMUM_DECAY = b_value_to_minimum_decay
        
        self.GAMMA = gamma
        self.TAU = tau
        
        # Learning rate
        self.LR = lr
        
        # Learn every x ticks
        self.UPDATE_EVERY = update_every
        
        # Do we want to use
        self.DDQN = ddqn
        
        # Do we want to use a normal Q network or a Dueling Q network
        self.DUELING = dueling
        
        # EPSILON
        self.episode = episode - 1                                   #
        self.EPSILON_TO_MINIMUM_DECAY = epsilon_to_minimum_decay   #
        self.EPSILON_INIT = epsilon_init                             #
        self.EPSILON_MAX = epsilon_max                               #
        self.EPSILON_MIN = epsilon_min                               #
        
     
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size
        self.SEED = random.seed(seed)
        
        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        #----------------------#
        # select the Q-Network #
        #----------------------#
        if self.DUELING:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.LR)

        #----------------------#
        # Set the ReplayBuffer #
        #----------------------#
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.SEED, importance_sampling_weight_perc, prioritize_perc)
        
        #----------------------------#
        # Initialize a learning tick #
        #----------------------------#-------------#
        # If we would like to UPDATE_EVERY x steps #
        # then we need a variable which says in    #
        # which step we are                        #
        #------------------------------------------#
        self.tick = 0   
        
        
        #------------------#
        # Store some stuff #
        #------------------#
        self.last_loss = 0
        self.last_weighted_loss = 0
        self.creation_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #---------------#
    # Help methodes #
    #---------------#
    
    def new_episode(self):
        self.episode += 1
        self.last_loss = 0
        self.last_weighted_loss = 0
        self.memory.set_importance_sampling_weight_perc(self.get_b_decay())
        self.optimizer.param_groups[0]['lr'] *= 0.999
        
    def get_epsilon(self):
        return max(self.EPSILON_MIN ,(self.EPSILON_MIN / self.EPSILON_MAX)**(self.episode / self.EPSILON_TO_MINIMUM_DECAY))
          
    def get_b_decay(self, b = None, episode = None):
        if episode is None:
            return 1 - np.exp(- self.importance_sampling_weight_decay * self.episode)
        return min(1, 1 - np.exp(- b * episode))
        
    def calculate_b_deacy(self, nr_episodes, xth_percentage = 0.20, value = 0.75):
        """
        Find a b, where the value in step x == yield
        e.g. nr_episodes = 100
             xth_percentage = 0.20
             value = 0.75
             
        then we would like to find b where:
        0.75  =  1 - np.exp(- b * int(nr_episodes*xth_percentage))
        (its works pritty wel :) 
        
        """
        i_episode = max(1,int(nr_episodes*xth_percentage))
        self.importance_sampling_weight_decay = np.log(max(1-value,0.000001))/(- i_episode)
        
    
    
    
    def get_last_loss(self):
        return self.last_loss
        
    def get_last_weighted_loss(self):    
        return self.last_weighted_loss
        
    def get_tick(self):
        return self.tick

    def get_creation_timestamp(self):
        return self.creation_timestamp
    def save_model_checkpoints(self):
        print('saving the network!')
        torch.save(self.qnetwork_local.state_dict(), f"./model_checkpoints/local_network_{self.creation_timestamp}.pt")
        torch.save(self.qnetwork_target.state_dict(), f"./model_checkpoints/target_network_{self.creation_timestamp}.pt")
        print('The newtork has been saved!\nCheck the checkpoints folder!')
        
    #------------#
    # STEP 1     #
    #------------#
    #
    #
      
    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        # Epsilon-greedy action selection
        # Eps will decrease over time, but in the begining it will be near 1. this we'll take more 
        # random decisions, the smaller eps becomes, the less random we go!
        if random.random() <= self.get_epsilon():
            return random.choice(np.arange(self.ACTION_SIZE))
        
        
        
        # Transform the state input to a torch value
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Step into the eval modus (a.k.a. do not update weigths)
        self.qnetwork_local.eval()
        
        # Get the probabilities of the possible actions
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        # Return back into the Train modus
        self.qnetwork_local.train()

        # Return the action back with highest probability
        return np.argmax(action_values.cpu().data.numpy())
        
        
    #----------#
    # STEP 2   #
    #----------#
    # 
    # Previously in step 1, we performed a (random) action & we asked the environment
    # what is the reward, next_state, done values, given our current_state and action
    #
    # Now if we've enough data, we try to learn some stuff
    #
    def step(self, state, action, reward, next_state, done):
        
        # Save experience in replay memory
        # we add a small delta value to 
        self.memory.add_record(state, action, reward, next_state, done, delta = 0.01)
        
        # Learn every UPDATE_EVERY time steps. 0 1 2 3 0 1 2 3 ....
        self.tick = (self.tick + 1) % self.UPDATE_EVERY
        
        if self.tick == 0:
            
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                
                experiences = self.memory.sample()
                self.learn(experiences)
                

    #-------------#
    # LEARN STUFF #
    #-------------#
    #
    #
    #
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, records = experiences

        #-----#
        # DQN #
        #-----#
        # for a batch (64 records ):
        # - Predict the action to undertake (64 X 4) (if we've 4 actions)
        #
        # tensor([[ 0.1045, -0.0075, -0.2736,  0.1418],
        #         [ 0.1603, -0.0054,  0.1768,  0.1021],
        #         [ 0.1830, -0.0304, -0.1600,  0.1432],
        #         [ 0.1380,  0.0194, -0.1237,  0.0512]])
        #
        # - Take the max value of each tensor (.max(1))
        # (tensor([ 0.1418,  0.1768,  0.1830,  0.1380]), tensor([ 3,  2,  0,  0]))
        #
        # Result : Grep the max value
        # Q_targets_next = tensor([[ 0.1418], [ 0.1768], [ 0.1830], [ 0.1380]])
        
        if self.DDQN == False:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            #-----#
            # DQNN #
            #-----#
            # for a batch (64 records ):
            # - Predict the action to undertake (64 X 4) (if we've 4 actions)
            #
            # tensor([[ 0.1045, -0.0075, -0.2736,  0.1418],
            #         [ 0.1603, -0.0054,  0.1768,  0.1021],
            #         [ 0.1830, -0.0304, -0.1600,  0.1432],
            #         [ 0.1380,  0.0194, -0.1237,  0.0512]])
            #
            # - Take the probabilities of the MAX value of the LOCAL NETWORK !
            local_network = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            
            Q_targets_next = self.qnetwork_target(next_states).gather(1, local_network).detach()
        
        # Compute Q targets for current states 
        # 0 + 0.8 * 0.4 * 1  (if done == True, we take the whole reward else gamme * Q_targets_next)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Get the values of the state, actions
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        
      
        # Compute loss (mean square error)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss += loss
        
        
        #------------------------------------#
        # edit the delta value if the record #
        #------------------------------------#
        weights = np.zeros(len(records))
        for e, (r, exp, tar) in enumerate(zip(records, Q_expected.detach(), Q_targets.detach())):
            r.set_delta(np.abs(exp[0].numpy() - tar[0].numpy()))
            weights[e] = r.get_importance_sampling_weight()
        
        # adjust he weighted_loss
        weighted_loss = torch.mean(torch.from_numpy(weights).float() * loss)
        self.last_weighted_loss += weighted_loss
        
        
        
        # Minimize the loss
        self.optimizer.zero_grad()
        
        #loss.backward()
        weighted_loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data) 
        