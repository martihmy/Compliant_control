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


# ------------ Helper functions --------------------------------

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
    

def PD_torque_control(x_d,E, ori, P_pos=1000, P_ori = 50, D_pos=100, D_ori=1): #100,25,50,1
    curr_pos = robot.endpoint_pose()['position'] #400                   300
    curr_ori = np.asarray(robot.endpoint_pose()['orientation'])

    goal_pos = x_d + E
    goal_ori = np.asarray(ori)

    delta_pos = (goal_pos - curr_pos).reshape([3,1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3,1])


    curr_vel = robot.endpoint_velocity()['linear'].reshape([3,1])
    curr_omg = robot.endpoint_velocity()['angular'].reshape([3,1])

    # Desired task-space force using PD law
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel), D_ori*(curr_omg)])
            
    J = robot.zero_jacobian()  
    
    tau = np.dot(J.T,F) # joint torques to be commanded

    robot.set_joint_torques(dict(list(zip(robot.joint_names(), tau)))) # command robot using joint torques




def update_force_list(f_list): #setting forces in x and y = 0
    for i in range(3): #update for x, then y, then z
        f_list[i][2]=f_list[i][1]
        f_list[i][1]=f_list[i][0]
        if i ==2:
            f_list[i][0] = robot.endpoint_effort()['force'][i]-F_d[i]
        else:
            f_list[i][0] = 0
    
def update_x_history(x_history, current_x):
    for i in range(3):
        x_history[i][1]=x_history[i][0]
        x_history[i][0] = current_x[i]

def raw_position_control(x_d,E,ori):
    x_c = x_d + E
    joint_angles = robot.inverse_kinematics(x_c,ori=ori)[1]
    #print(joint_angles)
    robot.exec_position_cmd(joint_angles)
    #robot.move_to_joint_position(joint_angles)



def calculate_x(T,x_list, force_list,M = 1*np.array([1, 1, 1]),B = 60*np.array([1, 1, 1]),K= 10*np.array([1, 1, 1])):
    x_x = (T**(2) * force_list[0][0] + 2* T**(2) * force_list[0][1]+ T**(2) * force_list[0][2]-(2*K[0]*T**(2)-8*M[0])*x_list[0][0]-(4*M[0] -2*B[0]*T+K[0]*T**(2))*x_list[0][1])/(4*M[0]+2*B[0]*T+K[0]*T**(2))
    x_y = (T**2 * force_list[1][0] + 2* T**2 * force_list[1][1]+ T**2 * force_list[1][2]-(2*K[1]*T**2-8*M[1])*x_list[1][0]-(4*M[1] -2*B[1]*T+K[1]*T**2)*x_list[1][1])/(4*M[1]+2*B[1]*T+K[1]*T**2)
    x_z = (T**2 * force_list[2][0] + 2* T**2 * force_list[2][1]+ T**2 * force_list[2][2]-(2*K[2]*T**2-8*M[2])*x_list[2][0]-(4*M[2] -2*B[2]*T+K[2]*T**2)*x_list[2][1])/(4*M[2]+2*B[2]*T+K[2]*T**2)
    return np.array([x_x,x_y,x_z]) 

def plot_result(a,b,c,F_d,x_d):

    time_array = np.arange(len(F_d[0]))*0.001
    

    plt.subplot(121)
    plt.title("Sensed external wrench")
    """
    plt.plot(a[0,:], label="force x [N]")
    plt.plot(a[1,:], label="force y [N]")"""
    plt.plot(time_array, a[2,:], label="force z [N]")
    """
    plt.plot(a[3,:], label="torque x [Nm]")
    plt.plot(a[4,:], label="torque y [Nm]")
    plt.plot(a[5,:], label="torque z [Nm]")
    """
    plt.plot(time_array, F_d[2,:], label = " desired z-force [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    plt.subplot(122)
    plt.title("position")
    plt.plot(time_array, c[0,:], label = "true x [m]")
    plt.plot(time_array, c[1,:], label = "true y [m]")
    plt.plot(time_array, c[2,:], label = "true  z [m]")
    plt.plot(time_array, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(time_array, x_d[2,:], label = "desired z [m]", color='g',linestyle='dashed')
    plt.plot(time_array, b[2,:], label = "compliant z [m]", color='g',linestyle='dotted')

    plt.xlabel("Real time [s]")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rospy.init_node("admittance_control")
    robot = PandaArm()
    robot.move_to_neutral() 

    max_num_it=12000 # 12 seconds
    T=0.001 # Controller loop period [correct in sim]

    F_d =np.array([0,0,0])
    goal_ori = robot.endpoint_pose()['orientation'] #goal = current
    x_d = robot.endpoint_pose()['position']
    #x_d = np.asarray([0.3,0,0.59]) #random goal position 42->46-49

    # ---------- Initialization -------------------

    sensor_readings = np.zeros((6,max_num_it))
    x_c_list = np.zeros((3,max_num_it))
    x_list = np.zeros((3,max_num_it))
    x_d_list = np.zeros((3,max_num_it))
    F_d_list = np.zeros((3,max_num_it))
    f_list = np.zeros((3,3))
    current_x = np.zeros(3)
    x_history = np.zeros((3,3))
                            
    for i in range(max_num_it):
        #for plotting
        sensor_readings[:,i]=np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
        x_d_list[:,i] = x_d
        x_c_list[:,i] = x_d + current_x
        x_list[:,i] = robot.endpoint_pose()['position']
        F_d_list[:,i] = F_d

        if i < 1800:
            x_d[2] -= 0.00005
        
        if i == 1700:
            F_d =np.array([0,0,15])
        
        
        if i > 3000 and i < 7000: 
            x_d[0] += 0.00005#move 20 cm in the x direction 
        
        
        if  i > 1700 and i%3==0: 
            update_force_list(f_list)
            current_x = calculate_x(T,x_history, f_list)
            update_x_history(x_history,current_x)
            
        """chose one of the two position controllers: """
        #raw_position_control(x_d,current_x,goal_ori) #control x_c = x_d + x(k)
        PD_torque_control(x_d,current_x,goal_ori)
        
        #printing and plotting
        if i%100==0:
            print(i,', pos:',robot.endpoint_pose()['position'],' F: ', robot.endpoint_effort()['force'][2])#' force measured: ',robot.endpoint_effort()['force'])
    plot_result(sensor_readings,x_c_list,x_list,F_d_list,x_d_list)


