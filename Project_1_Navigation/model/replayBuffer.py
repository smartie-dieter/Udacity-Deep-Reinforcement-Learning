import numpy as np
import random
import torch 

from model.queue import Queue
from model.queue import Record


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, importance_sampling_weight_perc= 0, prioritize_perc = 0, min_prob = 0.01):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            
        prioritize_perc
        ================
        If prioritize_perc == 0 then: Each record has an equal chance to be sampled
        If prioritize_perc == 1 then: we'll sample with respect to the delta value
        
        importance_sampling_weight
        ================
        If importance_sampling_weight == 0 then: Samples wont get punished because of their previous delta value
        If importance_sampling_weight == 1 then: Samples will be punished because of their previous delta value
                                                         
        """
        
        # initialize
        self._action_size = action_size
        self._memory = Queue(buffer_size = buffer_size, min_prob = min_prob)
        self._batch_size = batch_size
   
        # set the importance_sampling_weight
        self._importance_sampling_weight_perc = importance_sampling_weight_perc

        # set the prioritize_perc value
        self._prioritize_perc = prioritize_perc
        
        # set a seed
        self._seed = random.seed(seed)
        
        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def get_prioritize_perc(self):
        return self._prioritize_perc
        
    def set_prioritize_perc(self, prioritize_perc):
        self._prioritize_perc = prioritize_perc
        
    def get_importance_sampling_weight_perc(self):
        return self._importance_sampling_weight_perc

    def set_importance_sampling_weight_perc(self, importance_sampling_weight_perc):
        self._importance_sampling_weight_perc = importance_sampling_weight_perc

        
    def add_record(self, state, action, reward, next_state, done, delta = 0):
        """Add a new experience to memory."""
        
        # add a new record to the memory queue
        self._memory.add_record(Record(state, action, reward, next_state, done, delta))
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self._memory.get_sample(sample_size = self._batch_size, 
                                             prioritize_perc = self._prioritize_perc, 
                                             importance_sampling_weight_perc = self._importance_sampling_weight_perc)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones, experiences)

    def __len__(self):
        """Return the current size of internal memory."""
        return self._memory.get_size()

