import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs.Admittance_support import admittance_config as cfg


F_WINDOW_SIZE = 5
Fd_WINDOW_SIZE = 10
DELTA_Xd_SIZE = 10
SIM_STATUS = True

publish_rate = 50
T = 0.001*(1000/publish_rate) # The control loop's time step
duration = 15
MAX_NUM_IT = int(duration*publish_rate)


M = 5
B_START = 10
K_START = 100

