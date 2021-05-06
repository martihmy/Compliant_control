import gym
#from gym import ...

#! /usr/bin/env python
import numpy as np
import gym
from gym import spaces
#from gym_panda.envs import VIC_func as func

SIM_STATUS = True

#ACTION SPACE  RANDOM VALUES
GAMMA_B_LOWER = 10**(-3)
GAMMA_B_UPPER = 10**(-1)

GAMMA_K_LOWER = 10**(-4)
GAMMA_K_UPPER = 10**(-1)

KP_POS_LOWER = 500
KP_POS_UPPER =  1000


#initialization

GAMMA_B_INIT = (GAMMA_B_UPPER + GAMMA_B_LOWER) /2
GAMMA_K_INIT = (GAMMA_K_UPPER + GAMMA_K_LOWER) /2
KP_POS_INIT = (KP_POS_UPPER + KP_POS_LOWER) /2



# With M=5:
# parameters of stiffness and damping matrices
Kp =  700 # learning!
Kpz = 25 #initial value (adaptive)
Ko = 1000#900

Bp = 700/4
Bpz = 15 # #initial value (adaptive)
Bo = 10#10

# With M = 10:
"""
Kp =  40 # learning!
Kpz = 25 #initial value (adaptive)
Ko = 1000#900

Bp = 20
Bpz = 10 # #initial value (adaptive)
Bo = 50#10
"""
# MASS, DAMPING AND STIFFNESS MATRICES (ONLY M IS COMPLETELY CONSTANT)
M = np.identity(6)*10
B = np.array([[Bp, 0, 0, 0, 0, 0],
                [0, Bp, 0, 0, 0, 0],
                [0, 0, Bpz, 0, 0, 0],
                [0, 0, 0, Bo, 0, 0],
                [0, 0, 0, 0, Bo, 0],
                [0, 0, 0, 0, 0, Bo]])
K = np.array([[Kp, 0, 0, 0, 0, 0],
                [0, Kp, 0, 0, 0, 0],
                [0, 0, Kpz, 0, 0, 0],
                [0, 0, 0, Ko, 0, 0],
                [0, 0, 0, 0, Ko, 0],
                [0, 0, 0, 0, 0, Ko]])

K_v = np.identity(6)
P = np.identity(6)

B_hat_lower = 0
B_hat_upper = 300
B_hat_limits = [B_hat_lower,B_hat_upper]

K_hat_lower = 10
K_hat_upper = 1000
K_hat_limits = [K_hat_lower,K_hat_upper]

list_of_limits = [GAMMA_B_LOWER, GAMMA_B_UPPER, GAMMA_K_LOWER,GAMMA_K_UPPER, KP_POS_LOWER, KP_POS_UPPER,B_hat_lower,B_hat_upper,K_hat_lower,K_hat_upper ]



Fd = 3
PUBLISH_RATE = 40
T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
duration = 5
MAX_NUM_IT = int(duration*PUBLISH_RATE)
ALTERNATIVE_START = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}


#OBSERVATION SPACE

LOWER_F = -50
UPPER_F = 60

LOWER_DELTA_Z = -0.5
UPPER_DELTA_Z = 0.5

LOWER_VEL = -10
UPPER_VEL = 10

LOWER_X_ERROR = -0.1
UPPER_X_ERROR = 0.1

LOWER_FORCE_DOT = -1000
UPPER_FORCE_DOT = 1000

LOWER_FORCE_OVERSHOOT = 0
UPPER_FORCE_OVERSHOOT = 50