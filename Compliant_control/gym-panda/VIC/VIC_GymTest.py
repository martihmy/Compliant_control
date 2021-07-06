import gym
import gym_panda
import random
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

from gym_panda.envs.VIC_func import plot_run
#from gym_panda.envs.VIC_config import list_of_limits, GAMMA_B_INIT, GAMMA_K_INIT, KP_POS_INIT 
 


"""
This script is running the Variable Impedance controller in the Gym-interface

- An agent is performing random actions (no training) 
"""



if __name__ == "__main__":
    print('started')
    env = gym.make('panda-VIC-v0')
    number_of_runs = 1

    for episode in range(number_of_runs):
        print('starting run ', episode+1, ' /',number_of_runs)
        done= False
        x = env.reset()
        while done==False:
            pot = [random.uniform(1,3),random.uniform(1,3)] #can be used to test new action-regions
            u = [10**(-pot[0]),10**(-pot[1])]
            x_new, reward, done, info = env.step(u)

        plot_run(info)