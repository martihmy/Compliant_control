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
from scipy.spatial.transform import Rotation

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
    

def PD_torque_control(x_d,E, ori, P_pos=1050, P_ori = 50, D_pos=100, D_ori=1): #100,25,50,1
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




def update_force_list(f_list,F_d): #setting forces in x and y = 0
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

def plot_result(force,x_c,pos,F_d,x_d,ori,ori_d,T):

    time_array = np.arange(len(F_d[0]))*T
    

    plt.subplot(121)
    plt.title("Sensed external wrench")
    """
    plt.plot(force[0,:], label="force x [N]")
    plt.plot(force[1,:], label="force y [N]")"""
    plt.plot(time_array, force[2,:], label="force z [N]")
    """
    plt.plot(force[3,:], label="torque x [Nm]")
    plt.plot(force[4,:], label="torque y [Nm]")
    plt.plot(force[5,:], label="torque z [Nm]")
    """
    plt.plot(time_array, F_d[2,:], label = " desired z-force [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(122)
    plt.title("position")
    plt.plot(time_array, pos[0,:], label = "true x [m]")
    plt.plot(time_array, pos[1,:], label = "true y [m]")
    plt.plot(time_array, pos[2,:], label = "true  z [m]")
    plt.plot(time_array, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(time_array, x_d[2,:], label = "desired z [m]", color='g',linestyle='dashed')
    plt.plot(time_array, x_c[2,:], label = "compliant z [m]", color='g',linestyle='dotted')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.subplot(133)
    plt.title("Error in orientation")
    plt.plot(time_array, ori[0,:], label = "true  Ori_x [degrees]")
    plt.plot(time_array, ori[1,:], label = "true  Ori_y [degrees]")
    plt.plot(time_array, ori[2,:], label = "true  Ori_z [degrees]")

    #plt.axhline(y=ori_d[0], label = "desired Ori_x [degrees]", color='b',linestyle='dashed')
    #plt.axhline(y=ori_d[1], label = "desired Ori_y [degrees]", color='C1',linestyle='dashed')
    #plt.axhline(y=ori_d[2], label = "desired Ori_z [degrees]", color='g',linestyle='dashed')

    plt.xlabel("Real time [s]")
    plt.legend()
    """

    plt.show()

def get_ori_degrees():
    quat_as_list = np.array([robot.endpoint_pose()['orientation'].x,robot.endpoint_pose()['orientation'].y,robot.endpoint_pose()['orientation'].z,robot.endpoint_pose()['orientation'].w])
    rot = Rotation.from_quat(quat_as_list)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return np.array([(rot_euler[0]-np.sign(rot_euler[0])*180),rot_euler[1],rot_euler[2]])

def get_ori_degrees_error(ori_d):
    return get_ori_degrees()-ori_d

def get_f_d(i,original_d=np.array([0,0,15])):
    if i < 1500:
        return np.array([0,0,float(i)/100])
    elif i > 2000 and i < 4000:
        new_lambda_d = original_d + np.array([0,0,5*np.sin(i*0.001*2*np.pi)])
        return new_lambda_d
    else:
        return original_d

def get_x_d(i,current_x_d):
    if i > 4500 and i < 6500:
        new_r_d = current_x_d + np.array([0.0001,0,0]) #adding to x
        return new_r_d
    else:
        return current_x_d

if __name__ == "__main__":
    rospy.init_node("admittance_control")
    robot = PandaArm()
    robot.move_to_neutral()
    publish_rate = 250
    rate = rospy.Rate(publish_rate)

    max_num_it=7500
    T = 0.001*(1000/publish_rate) #correct for sim

    F_d =np.array([0,0,0])
    goal_ori = robot.endpoint_pose()['orientation'] #goal = current
    x_d = robot.endpoint_pose()['position']
    #x_d = np.asarray([0.3,0,0.59]) #random goal position 42->46-49

    # ---------- Initialization -------------------

    sensor_readings = np.zeros((6,max_num_it))
    x_c_list = np.zeros((3,max_num_it))
    x_list = np.zeros((3,max_num_it))
    x_d_list = np.zeros((3,max_num_it))
    #x_d_list = np.zeros((6,max_num_it))
    F_d_list = np.zeros((3,max_num_it))
    f_list = np.zeros((3,3))
    current_x = np.zeros(3)
    x_history = np.zeros((3,3))
    
    #For plotting
    ori_degrees_error_history = np.zeros((3,max_num_it))
    desired_ori_degrees = get_ori_degrees()
                            
    for i in range(max_num_it):

        F_d = get_f_d(i)
        x_d = get_x_d(i,x_d)

        #for plotting
        sensor_readings[:,i]=np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
        x_d_list[:,i] = x_d
        x_c_list[:,i] = x_d + current_x
        x_list[:,i] = robot.endpoint_pose()['position']
        F_d_list[:,i] = F_d
        ori_degrees_error_history[:,i] = get_ori_degrees_error(desired_ori_degrees)
        #
        
        if i%3==0: 
            update_force_list(f_list,F_d)
            current_x = calculate_x(T,x_history, f_list)
            update_x_history(x_history,current_x)
            
        """chose one of the two position controllers: """
        #raw_position_control(x_d,current_x,goal_ori) #control x_c = x_d + x(k)
        PD_torque_control(x_d,current_x,goal_ori)
        rate.sleep() #added
        
        #printing and plotting
        if i%100==0:
            print(i,', pos:',robot.endpoint_pose()['position'],' F: ', robot.endpoint_effort()['force'][2])#' force measured: ',robot.endpoint_effort()['force'])
    plot_result(sensor_readings,x_c_list,x_list,F_d_list,x_d_list,ori_degrees_error_history, desired_ori_degrees,T)


