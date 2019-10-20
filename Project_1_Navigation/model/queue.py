from collections import deque
import numpy as np

# Custom Queue and records
class Queue:
    def __init__(self, buffer_size, min_prob = 0.01):
        """

        #min_prob
        Each record contains a minimum probability, this prevent us from dividing by 0
        """        
        # queue 
        self.queue = deque(maxlen = buffer_size) 

        # min_prob
        # Each record contains a minimum probability, this prevent us from dividing by 0
        self.min_prob = min_prob

    def add_record(self, record):
        """ Add a record to the queue
            when the queue is full, the oldest record will be removed
        """
        self.queue.append(record)
        

        
    def get_sample(self, sample_size, prioritize_perc, importance_sampling_weight_perc):
        """
            Get_sample is a method, which will return a sample back from the queue
            If the queue_size < sample_size, we'll return everything back
            Else we will return N samples back, where N equals the batch size.
            
            Also if the supplied prioritize_perc value > 0, then we will sample from an uniform distribution
            otherwise, we'll sample with respect to delta
            
            Last but not least, we'll update the record's importance sampling weight.
            This controls the impact during learning.
            Basically, records which have a high probability, 
            (and thus have a higher change to be sampled more often then other records) 
            will have a lower weight | impact during the backpropagation phase! when the value goes to 1
            
            (might become a bottle neck)
        """
        
        

        # calculate the probabilities
        probabilities = [ ((r.get_delta() + self.min_prob) **prioritize_perc) for r in self.queue]
        denominator = sum(probabilities)
        probabilities = [p/denominator for p in probabilities]
        
        
        # calculate the probabilities
        index_records = np.random.choice(   a = len(self.queue), 
                                         size = sample_size if sample_size < len(self.queue) else len(self.queue), 
                                            p = probabilities,
                                      replace = False)
                                      
        # grep the records
        records = [self.queue[i] for i in index_records] 
        
        # update the importance sample weight of the record
        weights = [(self.queue.maxlen * probabilities[index])**-importance_sampling_weight_perc for index in index_records]
        
        for weight, record in zip(weights, records):
            record.set_importance_sampling_weight( weight/max(weights) )

            
        return records
            
    
    def get_size(self):
        """Get the current size of the queue"""
        return len(self.queue)

    
    
class Record:
    def __init__(self, state, action, reward, next_state, done, delta = 0):
        """
        -- Record --
        The record is the lowest level in our Replay Buffer. It represents an item in a list.
        Each item contains the next attributes:
        - state
        - action
        - reward
        - next_state
        - done       
        
        Extra parameters:
        - delta
        - importance_sampling_weight
        
        # Delta
        Delta is the basic value to prioritize the records in our Replay Buffer.
        Delta itself, is the absolute difference between the expected value and predicted value.
        If both values were equal then we could say that our network did a great job. But in the other case we would like to investigate it again.
        
        # importance_sampling_weight
        A weight which we'll use during training
        """

        # Basic knowledge
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state =  next_state
        self.done = done
        
        # Extra parameters
        
        # Delta is the basic value to prioritize the records in our Replay Buffer.
        # Delta itself, is the absolute difference between the expected value and predicted value.
        self.delta = None
        self.set_delta(delta)
        self.importance_sampling_weight = 0
        
        
    def set_delta(self, delta):
        self.delta = delta

        
    def get_delta(self):
        return self.delta
        
    def set_importance_sampling_weight(self, importance_sampling_weight):
        self.importance_sampling_weight = importance_sampling_weight

        
    def get_importance_sampling_weight(self):
        return self.importance_sampling_weight
        
        
    def get_values(self):
        return self.state, self.action, self.reward, self.next_state, self.done, self.delta