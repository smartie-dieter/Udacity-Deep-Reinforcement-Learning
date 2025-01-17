import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from files.replayBuffer import ReplayBuffer
from ouNoise import OUNoise
from model_actor import Actor
from model_critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed = 1, \
                       learn_interval = 4, learn_num = 1, lr_actor = 1e-4, lr_critic = 1e-3, \
                       gamma = 0.99, weight_decay = 0, tau = 0.001, batch_size = 128, buffer_size = 1e5):
        """Initialize an Agents object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            seed (int): random_seed
        """
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size
        self.NUM_AGENTS = num_agents
        self.seed = random.seed(random_seed)


        # hyper static parameters:
        self.LEARN_INTERVAL = learn_interval
        self.LEARN_NUM = learn_num
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.GAMMA = gamma
        self.WEIGHT_DECAY = weight_decay
        self.TAU = tau
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        
        
        # Actor network with target network
        self.actor_local = Actor(self.STATE_SIZE, self.ACTION_SIZE, random_seed).to(device)
        self.actor_target = Actor(self.STATE_SIZE, self.ACTION_SIZE, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr= self.LR_ACTOR)

        # Critic network with target network
        self.critic_local = Critic(self.STATE_SIZE, self.ACTION_SIZE, random_seed).to(device)
        self.critic_target = Critic(self.STATE_SIZE, self.ACTION_SIZE, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr= self.LR_CRITIC, weight_decay= self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((self.NUM_AGENTS, self.ACTION_SIZE), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(self.ACTION_SIZE, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        

    def reset(self):
        self.noise.reset()
             
    def step(self, step, states, actions, rewards, next_states, dones):
        """Save experience in replay memory and use random sample from buffer to learn."""
        for n in range(self.NUM_AGENTS):
            self.memory.add(states[n, :], actions[n, :], rewards[n], next_states[n, :], dones[n])

        # Learn every X frames | intervals
        if step % self.LEARN_INTERVAL == 0:
            
            # Learn, if we have enough samples to learn
            if len(self.memory) > self.BATCH_SIZE:

                # amount of times that we want to learn
                # is not the same as batch size
                for _ in range(self.LEARN_NUM):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.GAMMA)

    def action(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
       

        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.NUM_AGENTS, self.ACTION_SIZE))
        
        # get out of the training environment
        self.actor_local.eval()
        with torch.no_grad():
            # for each state, predict the next action
            for n, state in enumerate(states):
                actions[n, :] = self.actor_local(state).cpu().data.numpy()
                
        # enter the training environment
        self.actor_local.train()

        # add some noise
        if add_noise:
            actions += self.noise.sample()
 
        # clip the action
        return np.clip(actions, -1, 1)

        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
            
    # Load and Save data
    def save_agent(self, checkpoint_name):
        torch.save(self.actor_local.state_dict(), f'./checkpoints/{checkpoint_name}_actor.pth')
        torch.save(self.critic_local.state_dict(), f'./checkpoints/{checkpoint_name}_critic.pth')

    def load_agent(self, checkpoint_name):
        self.actor_local.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_actor.pth'))
        self.critic_local.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_critic.pth'))
        