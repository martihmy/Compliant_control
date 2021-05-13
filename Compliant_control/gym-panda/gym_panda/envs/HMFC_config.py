import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
from gym_panda.envs import HMFC_func as func

SIM_STATUS = True

#ACTION SPACE
KD_LAMBDA_LOWER = 0
KD_LAMBDA_UPPER = 15

KP_LAMBDA_LOWER = 10
KP_LAMBDA_UPPER = 90

KP_POS_LOWER = 125
KP_POS_UPPER =  175

#initialization

KD_LAMBDA_INIT = 1
KP_LAMBDA_INIT = 15
KP_POS_INIT = 150

#CONTROL PARAMETERS FOR POSITION

#with time.sleep(0.4):
#Kp_p = 50#60#120#60
Kp_o = 100#120#150#120
Kd_p = 10#35#30#20#10#0.01#2#60*0.025
Kd_o = 0.5#1#5#40


#with no delay (old values):
"""
KD_LAMBDA_LOWER = 0
KD_LAMBDA_UPPER = 30

KP_LAMBDA_LOWER = 10
KP_LAMBDA_UPPER = 110

KP_POS_LOWER = 100
KP_POS_UPPER =  150

#initialization

KD_LAMBDA_INIT = 0.03
KP_LAMBDA_INIT = 30
KP_POS_INIT = 50

Kp_o = 200
Kd_p = 10
Kd_o = 1
"""


"""
Kp_r_init = np.array([[Kp_p, 0, 0, 0, 0], # Stiffness matrix
                [0, Kp_p, 0, 0, 0],
                [0, 0, Kp_o, 0, 0],
                [0, 0, 0, Kp_o, 0],
                [0, 0, 0, 0, Kp_o]])
                """

Kd_r = np.array([[Kd_p, 0, 0, 0, 0], # Damping matrix
                [0, Kd_p, 0, 0, 0],
                [0, 0, Kd_o, 0, 0],
                [0, 0, 0, Kd_o, 0],
                [0, 0, 0, 0, Kd_o]])

Fd = 3
PUBLISH_RATE = 50
T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
duration = 5
MAX_NUM_IT = int(duration*PUBLISH_RATE)
ALTERNATIVE_START = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}


#OBSERVATION SPACE

LOWER_F = -50
UPPER_F = 100

LOWER_DELTA_Z = -0.5
UPPER_DELTA_Z = 0.5

LOWER_VEL = -10
UPPER_VEL = 10

LOWER_X_ERROR = -0.1
UPPER_X_ERROR = 0.1

LOWER_FORCE_DOT = -1000
UPPER_FORCE_DOT = 1000
