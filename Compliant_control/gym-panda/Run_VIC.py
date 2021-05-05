import gym
import gym_panda
import random
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

from gym_panda.envs.VIC_func import plot_run
from gym_panda.envs.VIC_config import list_of_limits, GAMMA_B_INIT, GAMMA_K_INIT, KP_POS_INIT 
 


"""
This script is running the Variable Impedance controller in the Gym-interface

- An agent is performing random actions (no training) 
"""



class Agent():
	def __init__(self, action_space):
		self.action_space  = action_space


	def get_action(self):
		return self.action_space.sample()

if __name__ == "__main__":
    print('started')
    env = gym.make('panda-VIC-v0')
    agent = Agent(env.action_space)
    number_of_runs = 1

    u = [ GAMMA_B_INIT/100, GAMMA_K_INIT/100, 100] #constant actions
    for episode in range(number_of_runs):
        print('starting run ', episode+1, ' /',number_of_runs)
        done= False
        x = env.reset()
        while done==False:
            #u = agent.get_action()
            #u = [random.uniform(0,30),random.uniform(10,80), 100]#[0.045,45]
            x_new, reward, done, info = env.step(u)

        plot_run(info, list_of_limits)