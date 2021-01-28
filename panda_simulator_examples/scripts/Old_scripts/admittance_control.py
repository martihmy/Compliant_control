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
#import pytransform3d.rotations

#from rviz_markers import RvizMarkers
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# --------- Constants -----------------------------


"""
M_d = np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])

B_d = np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])

K_d = 50*np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])
"""
#--------------------------------------------------



# ------------ Helper functions --------------------------------

def update_temp_lists(delta_F,delta_X,x_d,F_d):
    for i in range(3):
        delta_F[i][2]=delta_F[i][1]
        delta_X[i][2]=delta_X[i][1]

        delta_F[i][1]=delta_F[i][0]
        delta_X[i][1]=delta_X[i][0]

        delta_X[i][0] = robot.endpoint_pose()['position'][i] - x_d[i]
        delta_F[i][0] = robot.endpoint_effort()['force'][i] - F_d[i]

def quatdiff_in_euler(quat_curr, quat_des):
    """
        Compute difference between quaternions and return 
        Euler angles as difference
    """
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
        
    return -des_mat.dot(vec)

def position_control_loop(goal_pos, goal_ori, P_pos=50,P_ori = 25, D_pos=10, D_ori=1):
    curr_pos = robot.endpoint_pose()['position']
    curr_ori = np.asarray(robot.endpoint_pose()['orientation'])

    delta_pos = (goal_pos - curr_pos).reshape([3,1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3,1])


    curr_vel = robot.endpoint_velocity()['linear'].reshape([3,1])
    curr_omg = robot.endpoint_velocity()['angular'].reshape([3,1])

    # Desired task-space force using PD law
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel), D_ori*(curr_omg)])

    #error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)
            
    J = robot.zero_jacobian()  
    
    tau = np.dot(J.T,F) # joint torques to be commanded

    robot.set_joint_torques(dict(list(zip(robot.joint_names(), tau)))) # command robot using joint torques

def compute_compliant_position(omega_1,omega_2,omega_3, delta_F,delta_X,x_d, T=0.01):
    
    E_x = omega_1[0]**(-1) * (T**(2) *(delta_F[0][0] + 2*delta_F[0][1] + delta_F[0][2]) -omega_2[0]*delta_X[0][1] - omega_3[0]*delta_X[0][2]) 

    E_y = omega_1[1]**(-1) * (T**(2) *(delta_F[1][0] + 2*delta_F[1][1] + delta_F[1][2]) -omega_2[1]*delta_X[1][1] - omega_3[1]*delta_X[1][2]) 

    E_z = omega_1[2]**(-1) * (T**(2) *(delta_F[2][0] + 2*delta_F[2][1] + delta_F[2][2]) -omega_2[2]*delta_X[2][1] - omega_3[2]*delta_X[2][2]) 

    return np.array([E_x, E_y, E_z]) + x_d


def calculate_omega(T,M_d = np.array([1, 1, 1]),B_d = np.array([1, 1, 1]),K_d = 50*np.array([1, 1, 1])):
    omega_1 = 4*M_d + 2* B_d
    omega_2 = -8*M_d + 2*K_d*T**2           # 3*1 vectors [omega_0[x],omega_0[y],omega_0[z]]
    omega_3 = 4*M_d-2*B_d*T + K_d*T**2
    return np.array([omega_1,omega_2,omega_3]) #3x3

if __name__ == "__main__":
        robot = ArmInterface()
        T=0.1
        omega = calculate_omega(T)

        F_d =np.array([0,0,15])
        goal_ori = np.asarray(robot.endpoint_pose()['orientation']) #goal = current
        goal_pos = np.asarray([0,0.3,0.1]) #random goal position

        # ---------- Initialization of temp-lists -------------------
        delta_F = np.zeros((3,3))    # x[N, N-1, N-2],
        delta_X = np.zeros((3,3))    # y[N, N-1, N-2],
                                     # z[N, N-1, N-2]
        for i in range(3000):
            position_control_loop(goal_pos,goal_ori)
            update_temp_lists(delta_F,delta_X,goal_pos,F_d)
            if i%3==0:
                compute_compliant_position(omega[0],omega[1],omega[2],delta_F, delta_X, goal_pos)