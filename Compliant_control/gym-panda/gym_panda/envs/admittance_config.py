import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
#from gym_panda.envs import admittance_config as cfg
from gym_panda.envs import admittance_functionality as af

F_WINDOW_SIZE = 5
Fd_WINDOW_SIZE = 10
DELTA_Xd_SIZE = 10
SIM_STATUS = True

PUBLISH_RATE = 50
T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
duration = 15
MAX_NUM_IT = int(duration*PUBLISH_RATE)

ALTERNATIVE_START = af.cartboard

M = 5
B_START = 10
K_START = 100

INCREMENT = 0.1
