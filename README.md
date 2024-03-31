# Reinforcement Learning PPO - Fight Zombies in Minecraft

This project is writen in Python using the [Malmo](https://github.com/microsoft/malmo/tree/master) platform, to create the 
reinforcement learning scenario of fighting zombies in Minecraft. Also, this project uses the Pytorch library for the
implementation of the DQN algorithm. The main idea, is to find out if the state of the art DQN algorithm is
able to learn and find strategies to win against the zombies in the Minecraft game.

## Training Scenario
The scenario in which the agent is trained involves combating zombies in Minecraft, designed using the 
[Malmo](https://github.com/microsoft/malmo/tree/master) platform. Spawned in a 20x20 open arena in the ear with three zombies
, the agent is equipped with iron armor and a diamond sword. Its objective is to learn appropriate actions 
(such as moving left, moving right, attacking, etc.) to survive against the zombies and of course not to fall form the platform.

![PPO.gif](MyResults%2FPPO.gif)

## Rewards
* Damage zombie: **+ 50**
* Lose health: **- 10**
* Die: **- 100**
* Every step: **- 0.1**

## PPO Algorithm
The implementation of the Proximal Policy Optimization (PPO) model utilized the [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) libraries that provides several 
abstractions and facilitates easy design. The PPO algorithm used for this example was based on [Gym BipedalWalker-v3 Rokas Liuberskis](https://pylessons.com/BipedalWalker-v3-PPO) and follows the Actor-Critic 
architecture which is illustrated in the picture below. Basically, the Actor-Critic model is represented by two neural networks: 
the Actor and the Critic and then the agent's policy is trained using the PPO algorithms to update these two networks.

![PPO.png](MyResults%2FPPO.png)

## Installation
* Install the Malmo library. The easiest method is to [install Marlo using Anaconda](https://marlo.readthedocs.io/en/latest/installation.html) (which includes Malmo).
* Ensure you have Python 3.7.12 installed.
* Install Tensorflow 2.3.0 and Keras 2.10.0

## Execution
Running `main.py` using method `PPOAgent.train()` will start the training of the model and by using method `PPOAgent.test()`
will start the evaluation of the last saved model.

## Results
Following training, the agent demonstrates acceptable performance against the zombies, achieving:
* **72%** win rate
* **13.86** average life (with 20 being the maximum)

![FightingZombies-Training.png](MyResults%2FResults11%2FFightingZombies-Training.png)