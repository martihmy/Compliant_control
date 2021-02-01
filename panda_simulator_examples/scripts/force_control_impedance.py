#! /usr/bin/env python
import copy
from copy import deepcopy
import rospy
import threading
import quaternion
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
#from interactive_markers.interactive_marker_server import *
from franka_interface import ArmInterface
from panda_robot import PandaArm
#import pytransform3d.rotations

#from rviz_markers import RvizMarkers
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

# --------- Constants -----------------------------

S_f = np.array([[0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]])

S_v = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]])
#should be estimated (this one is chosen at random)
K = np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 5000, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])

C = np.linalg.inv(K)


K_Plambda = 10*np.identity(6) #random
K_Dlambda = np.identity(6) #random

K_Pr = 10*np.identity(?) #random
K_Dr = np.identity(?) #random

# ------------ Helper functions --------------------------------

# GET STUFF
def get_joint_velocities():
    return np.array([robot.joint_velocity(robot.joint_names()[0]),robot.joint_velocity(robot.joint_names()[1]),robot.joint_velocity(robot.joint_names()[2]),robot.joint_velocity(robot.joint_names()[3]),robot.joint_velocity(robot.joint_names()[4]),robot.joint_velocity(robot.joint_names()[5]),robot.joint_velocity(robot.joint_names()[6])])

def get_endpoint_velocities():
    return np.append(robot.endpoint_velocity()['linear'],robot.endpoint_velocity()['angular']).reshape((6,1))


def get_r(): #
    return r

def get_r_d(i,current_r_d):
    if i < 3000:
        new_r_d = current_r_d + np.zeros(len(current_r_d)) #doing nothing
        return new_r_d
    else:
        return current_r_d

def get_S_f_inv(S_f,C):
    a = np.linalg.inv(np.linalg.multi_dot([S_f.T,C,S_f]))
    return np.linalg.multi_dot([a,S_f.T,C])
    
def get_K_dot(S_f,S_f_inv,K):
    return np.linalg.multi_dot([S_f,S_f_inv,K])

def get_dot(history,iteration,T):
    if iteration > 0:
        return (history[iteration]-history[iteration-1])/T
    else:
        return np.zeros(6)

def get_ddot(history,iteration,T):
    if iteration > 1:
        return get_dot(history,iteration,T) - get_dot(history,iteration-1,T)/T
    else:
        return np.zeros(6)



# TO BE DETERMINED
"""
lambda_d_ddot fixed
lambda_d_dot fixed
r
r_d_dot
r_d_ddot
"""
# control parameters
"""
K_Dlambda
K_Plambda
K_Pr
K_Dr
"""


# CALCULATE TORQUE
def calculate_f_lambda(i,T, S_f,C,  K_Dlambda,K_Plambda,lambda_d_history, lambda_d):
    S_f_inv = get_S_f_inv(S_f,C):
    K_dot = get_K_dot(S_f,S_f_inv,K)
    lambda_dot = np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()])
    lambda_true = np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
    return get_ddot(lambda_d_history,i,T) + np.dot(K_Dlambda,get_dot(lambda_d_history,i,T)-lambda_dot) + np.dot(K_Plambda,lambda_d-lambda_true)

def calculate_alpha_v(r_d_ddot,r_d_dot,r_d,v,r,K_Pr,K_Dr):
    return r_d_ddot + np.dot(K_Dr,r_d_dot-v)+ np.dot(K_Pr, r_d-r)

def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_f_inv(S_v,C)Pv/
    C_dot = np.dot(np.identity(3?)-P_v,C)
    return np.dot(S_v, alpha_v) + np.linalg.multi_dot([C_dot,S_f,f_lambda])

def perform_torque(alpha,J=robot.jacobian()):
    cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([J,np.linalg.inv(robot.joint_inertia_matrix()),J.T]))
    alpha_torque = np.linalg.multi_dot([J.T,cartesian_inertia,alpha])
    external_torque = np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
    torque = alpha_torque + robot.coriolis_comp() + external_torque

    robot.set_joint_torques(dict(list(zip(robot.joint_names(),torque))))



# MAIN FUNCTION

if __name__ == "__main__":
    rospy.init_node("impedance_control")
    robot = PandaArm()
    robot.move_to_neutral() 

    max_num_it=12000 # 12 seconds
    # MUST BE INITIALISED BEFORE LOOP
    T = 0.001 #for sim
    lambda_d = np.zeros(6)
    lambda_d_history = np.zeros((len(lambda_d),max_num_it))
    r_d = ?
    r_d_history = np.zeros((len(r_d),max_num_it)) 
    max_num_it = 12000
    for i in range(max_num_it):
        # IN LOOP:
        lambda_d_history[:,i]=lambda_d
        r_d = get_r_d(i,r_d)
        r_d_history[:,i] = r_d
        r_d_ddot = get_ddot(r_d_history,i,T)
        r_d_dot = get_dot(r_d_history,i,T)
    

        f_lambda = calculate_f_lambda(i, T, S_f ,C , K_Dlambda, K_Plambda, lambda_d_history, lambda_d)
        alpha_v = calculate_alpha_v(r_d_ddot,r_d_dot, r_d,get_endpoint_velocities(),r=?,K_Pr,K_Dr)
        alpha = calculate_alpha(S_v,alpha_v,C,S_f,f_lambda)
        perform_torque(alpha)Pv/