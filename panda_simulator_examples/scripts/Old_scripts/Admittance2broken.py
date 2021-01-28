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

np.set_printoptions(precision=2)

# --------- Constants -----------------------------


"""
M_d = np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])

B_d = np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])

K_d = 50*np.array([[1, 0, 0],[0, 1, 0],[0 ,0 ,1]])
"""
#--------------------------------------------------



# ------------ Helper functions --------------------------------

def update_temp_lists(delta_F,delta_X,x_d,F_d,E):
    for i in range(3):
        delta_F[i][2]=delta_F[i][1]
        delta_X[i][2]=delta_X[i][1]

        delta_F[i][1]=delta_F[i][0]
        delta_X[i][1]=delta_X[i][0]

        delta_X[i][0] = E[i]
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

def position_control_loop(x_d,E, goal_ori, P_pos=100, P_ori = 25, D_pos=50, D_ori=1): #200, 50
    curr_pos = robot.endpoint_pose()['position']
    curr_ori = np.asarray(robot.endpoint_pose()['orientation'])

    goal_pos = x_d + E

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

def compute_compliant_position(omega_1,omega_2,omega_3, delta_F,delta_X,E, T=0.01):
    
    #E_x = omega_1[0]**(-1) * (T**(2) *(delta_F[0][0] + 2*delta_F[0][1] + delta_F[0][2]) -omega_2[0]*delta_X[0][1] - omega_3[0]*delta_X[0][2]) 
    E_x = 0
    #E_y = omega_1[1]**(-1) * (T**(2) *(delta_F[1][0] + 2*delta_F[1][1] + delta_F[1][2]) -omega_2[1]*delta_X[1][1] - omega_3[1]*delta_X[1][2]) 
    E_y = 0

    E_z =omega_1[2]**(-1) * (T**(2) *(delta_F[2][0] + 2*delta_F[2][1] + delta_F[2][2]))# -omega_2[2]*delta_X[2][1] - omega_3[2]*delta_X[2][2]) 
    #E_z = 0
    return np.array([E_x, E_y, E_z]) + E


def compute_compliant_position2(omega_1,omega_2,omega_3, delta_F,delta_X,E, T=0.01):
    
    #E_x = omega_1[0]**(-1) * (T**(2) *(delta_F[0][0] + 2*delta_F[0][1] + delta_F[0][2]) -omega_2[0]*delta_X[0][1] - omega_3[0]*delta_X[0][2]) 
    E_x = 0
    #E_y = omega_1[1]**(-1) * (T**(2) *(delta_F[1][0] + 2*delta_F[1][1] + delta_F[1][2]) -omega_2[1]*delta_X[1][1] - omega_3[1]*delta_X[1][2]) 
    E_y = 0

    E_z =omega_3[2]**(-1) * (T**(2) *(delta_F[2][0] + 2*delta_F[2][1] + delta_F[2][2]))# -omega_2[2]*delta_X[2][1] - omega_3[2]*delta_X[2][2]) 
    #E_z = 0
    return np.array([E_x, E_y, E_z]) + E

def compute_compliant_position3(omega_1,omega_2,omega_3, delta_F, E_list,E, T=0.01):
    
    #E_x = omega_1[0]**(-1) * (T**(2) *(delta_F[0][0] + 2*delta_F[0][1] + delta_F[0][2]) -omega_2[0]*delta_X[0][1] - omega_3[0]*delta_X[0][2]) 
    E_x = 0
    #E_y = omega_1[1]**(-1) * (T**(2) *(delta_F[1][0] + 2*delta_F[1][1] + delta_F[1][2]) -omega_2[1]*delta_X[1][1] - omega_3[1]*delta_X[1][2]) 
    E_y = 0

    E_z =omega_1[2]**(-1) * (T**(2) *(delta_F[2][0] + 2*delta_F[2][1] + delta_F[2][2]) -omega_2[2]*E_list[2][1] - omega_3[2]*E_list[2][2]) 
    #E_z = 0
    return np.array([E_x, E_y, E_z]) + E

def compute_compliant_position4(omega_1,omega_2,omega_3, delta_F, E_list,E, T=0.01):
    
    #E_x = omega_1[0]**(-1) * (T**(2) *(delta_F[0][0] + 2*delta_F[0][1] + delta_F[0][2]) -omega_2[0]*delta_X[0][1] - omega_3[0]*delta_X[0][2]) 
    E_x = 0
    #E_y = omega_1[1]**(-1) * (T**(2) *(delta_F[1][0] + 2*delta_F[1][1] + delta_F[1][2]) -omega_2[1]*delta_X[1][1] - omega_3[1]*delta_X[1][2]) 
    E_y = 0

    E_z =omega_3[2]**(-1) * (T**(2) *(delta_F[2][0] + 2*delta_F[2][1] + delta_F[2][2]) -omega_2[2]*E_list[2][1] - omega_1[2]*E_list[2][2]) 
    #E_z = 0
    return np.array([E_x, E_y, E_z]) + E


def calculate_omega(T,M_d = np.array([1, 1, 1]),B_d = 1*np.array([1, 1, 1]),K_d = 50*np.array([1, 1, 1])):
    omega_1 = 4*M_d + 2* B_d + K_d*T**2
    omega_2 = -8*M_d + 2*K_d*T**2           # 3*1 vectors [omega_0[x],omega_0[y],omega_0[z]]
    omega_3 = 4*M_d-2*B_d*T + K_d*T**2
    return np.array([omega_1,omega_2,omega_3]) #3x3

def plot_result(a,b,c,F_d):
    plt.subplot(121)
    plt.title("Sensed external wrench")
    """
    plt.plot(a[0,:], label="force x [N]")
    plt.plot(a[1,:], label="force y [N]")"""
    plt.plot(a[2,:], label="force z [N]")
    """
    plt.plot(a[3,:], label="torque x [Nm]")
    plt.plot(a[4,:], label="torque y [Nm]")
    plt.plot(a[5,:], label="torque z [Nm]")
    """
    plt.plot(F_d[2,:], label = " desired z-force [N]", color='b',linestyle='dashed')
    plt.xlabel("number of iterations")
    plt.legend()
    plt.subplot(122)
    plt.title("true position")
    plt.plot(c[0,:], label = " x [m]")
    plt.plot(c[1,:], label = " y [m]")
    plt.plot(c[2,:], label = " z [m]")
    plt.plot(b[2,:], label = " compliant z [m]", color='g',linestyle='dashed')
    plt.xlabel("number of iterations")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rospy.init_node("admittance_control")
    robot = ArmInterface()
    robot.move_to_neutral() 

    max_num_it=6000
    T=0.1
    omega = calculate_omega(T)

    F_d =np.array([0,0,15])
    goal_ori = np.asarray(robot.endpoint_pose()['orientation']) #goal = current
    x_d = np.asarray([0.3,0,0.42]) #random goal position

    # ---------- Initialization -------------------
    E = np.asarray([0,0,0])
    delta_F = np.zeros((3,3))    # x[N, N-1, N-2],
    E_list = np.zeros((3,3))    # y[N, N-1, N-2],
                                 # z[N, N-1, N-2] 
    sensor_readings = np.zeros((6,max_num_it))
    x_c_list = np.zeros((3,max_num_it))
    x_list = np.zeros((3,max_num_it))
    F_d_list = np.zeros((3,max_num_it))
                            
    for i in range(max_num_it):
        #for plotting
        sensor_readings[:,i]=np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
        x_c_list[:,i] = x_d + E
        x_list[:,i] = robot.endpoint_pose()['position']
        F_d_list[:,i] = F_d
        
        if i ==3000: 
            x_d =  np.asarray([0.5,0,0.42]) #move 20 cm in the x direction 
        
        update_temp_lists(delta_F,E_list,x_d,F_d,E)
        if i%3==0: 
            E = compute_compliant_position1(omega[0],omega[1],omega[2],delta_F, E_list, E) #update compliant position
        position_control_loop(x_d,E,goal_ori) #control
        
        
        #printing and plotting
        if i%100==0:
            print(i,', pos:',robot.endpoint_pose()['position'],' F_e: ', np.array([delta_F[2][0]]))#' force measured: ',robot.endpoint_effort()['force'])
    plot_result(sensor_readings,x_c_list,x_list,F_d_list)