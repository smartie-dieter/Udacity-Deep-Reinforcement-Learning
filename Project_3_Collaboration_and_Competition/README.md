[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Project 3: Collaboration and Competition

### Introduction

In this project, i've worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 


### How to solve this environment? 
The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions & Things to know

Moste of the documentation is written in: `Continuous_Control(.ipynb|.html)`  
Basically, this project starts at : `PROJECT 2: CONTINUOUS CONTROL!`  
(Therefor you can skip the first 6 cells)  

#### How do you reproduce this project? 
Run the `Continuous_Control.ipynb` from top till bottom to get started with training your own agent! 

#### my results:

I managed to get an average score of +30 between the 36 and 60 steps. Depending on the configuration.  
All the models, hyper-parameters and logs are visible via: [Weights and Biases](https://app.wandb.ai/verbeemen/udacity_deep-reinforcement-learning_project-2?workspace=user-verbeemen).   
(Weights and Biases is a developer tools for deep learning. By adding a couple lines of code, your training script is able to keep track of your hyperparameters, system metrics, and outputs so that you can compare experiments, see live graphs of training, and easily share your findings with colleagues.)



table



![](images/wandb_chart_24-11-2019_15_19_05.svg)

#### Project folder structure.
- checkoints
  - The weights of the actor and cretic networks
- files
  - agent.py
  - model_actor.py
  - model_critic.py
  - ouNoise.py
  - replayBuffer.py
- images
  - images used in Continuous_Control.ipynb report
- wandb
  - contains logging files
- Continuous_Control.ipynb (the report)
- Continuous_Control.html  (the report)
- hyper_parameters.json    (a json which contains the hyperparameters)

#### Python dependencies
Python packages which were used in this project
- python 3.7
- jupyter
- unityagents
- pytorch
- numpy
- collections
- wandb
    


