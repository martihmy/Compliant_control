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
                [0, 0, 0]]).reshape([6,3])

S_v = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]]).reshape([6,3])

#should be estimated (this one is chosen at random)
K = np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 5000, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]]).reshape([6,6])

C = np.linalg.inv(K)


K_Plambda = 10*np.identity(3) #random
K_Dlambda = np.identity(3) #random

K_Pr = 100*np.identity(3) #random
K_Dr = np.identity(3) #random

# ------------ Helper functions --------------------------------

# GET STUFF
def get_joint_velocities():
    return np.array([robot.joint_velocity(robot.joint_names()[0]),robot.joint_velocity(robot.joint_names()[1]),robot.joint_velocity(robot.joint_names()[2]),robot.joint_velocity(robot.joint_names()[3]),robot.joint_velocity(robot.joint_names()[4]),robot.joint_velocity(robot.joint_names()[5]),robot.joint_velocity(robot.joint_names()[6])])

def get_v():
    return (np.array([robot.endpoint_velocity()['linear'][0],robot.endpoint_velocity()['linear'][1],robot.endpoint_velocity()['angular'][2]])).reshape([3,1])


def get_r(): #
    return np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],robot.endpoint_pose()['orientation'].z])

def get_lambda():
    return np.array([robot.endpoint_effort()['force'][2],robot.endpoint_effort()['torque'][0],robot.endpoint_effort()['torque'][1]])
    #return np.array([robot.endpoint_effort()['force'][2],0,0])
    #return np.zeros(3) #fake feedback 

def get_r_d(i,current_r_d):
    if i < 3000:
        new_r_d = current_r_d + np.zeros(3) #doing nothing
        return new_r_d
    else:
        return current_r_d

def get_lambda_d(i,current_lambda_d):
    if i < 3000:
        new_lambda_d = current_lambda_d + np.array([0.005,0,0])#.reshape([3,1]))
        return new_lambda_d
    else:
        return current_lambda_d.reshape([3,1])


def get_S_inv(S,C):
    a = np.linalg.inv(np.linalg.multi_dot([S.T,C,S]))
    return np.array(np.linalg.multi_dot([a,S.T,C])).reshape([3,6])
    
def get_K_dot(S_f,S_f_inv,C):
    return np.array(np.linalg.multi_dot([S_f,S_f_inv,np.linalg.inv(C)])).reshape([6,6])

def get_dot(history,iteration,T):
    if iteration > 0:
        return ((history[:,iteration]-history[:,iteration-1]).reshape([3,1])/T).reshape([3,1])
    else:
        return np.zeros(3).reshape([3,1])

def get_ddot(history,iteration,T):
    if iteration > 1:
        return ((get_dot(history,iteration,T) - get_dot(history,iteration-1,T)).reshape([3,1])/T).reshape([3,1])
    else:
        return np.zeros(3).reshape([3,1])


# CALCULATE TORQUE
def calculate_f_lambda(i,T, S_f,C,K_Dlambda,K_Plambda,lambda_d_history, lambda_d):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()])).reshape([3,1])
    return (get_ddot(lambda_d_history,i,T) + np.array(np.dot(K_Dlambda,(get_dot(lambda_d_history,i,T)-lambda_dot))).reshape([3,1]) + np.dot(K_Plambda,(lambda_d.reshape([3,1])-(get_lambda()).reshape([3,1]))).reshape([3,1]))

def calculate_alpha_v(i,T,r_d_history,r_d,K_Pr,K_Dr):
    r_d_dot = get_dot(r_d_history,i,T)
    r_d_ddot = get_ddot(r_d_history,i,T)
    return (r_d_ddot + np.array(np.dot(K_Dr,r_d_dot-get_v())).reshape([3,1])+ np.array(np.dot(K_Pr, r_d.reshape([3,1])-(get_r()).reshape([3,1]))).reshape([3,1]))

def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_inv(S_v,C)
    P_v = np.array(np.dot(S_v,S_v_inv))
    C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
    return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + np.array(np.linalg.multi_dot([C_dot,S_f,f_lambda])).reshape([6,1])

def perform_torque(alpha):
    cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([robot.jacobian(),np.linalg.inv(robot.joint_inertia_matrix()),robot.jacobian().T]))
    alpha_torque = np.array(np.linalg.multi_dot([robot.jacobian().T,cartesian_inertia,alpha])).reshape([7,1])
    external_torque = np.dot(robot.jacobian().T,np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])).reshape([7,1])
    torque = alpha_torque + robot.coriolis_comp().reshape([7,1]) + external_torque
    robot.set_joint_torques(dict(list(zip(robot.joint_names(),torque))))


def plot_result(f_controlled, f_d ,controlled_pose,x_d):

    time_array = np.arange(len(controlled_pose[0]))*0.001
    

    plt.subplot(121)
    plt.title("External wrench")
    
    plt.plot(time_array, f_controlled[0,:], label="force z [N]")
    plt.plot(time_array, f_controlled[1,:], label="torque x [Nm]")
    plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, f_d[0,:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.plot(time_array, f_d[1,:], label="desired torque x [Nm]", color='C1',linestyle='dashed')
    plt.plot(time_array, f_d[2,:], label="desired torque y [Nm]", color='g',linestyle='dashed')

    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(122)
    plt.title("Pose")

    plt.plot(time_array, controlled_pose[0,:], label = "true x [m]")
    plt.plot(time_array, controlled_pose[1,:], label = "true y [m]")
    plt.plot(time_array, controlled_pose[2,:], label = "true  Ori_z [quat]")
    plt.plot(time_array, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(time_array, x_d[2,:], label = "desired Ori_z [m]", color='g',linestyle='dashed')

    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.show()

# MAIN FUNCTION

if __name__ == "__main__":
    rospy.init_node("impedance_control")
    robot = PandaArm()
    robot.move_to_neutral() 

    max_num_it=500 # 12 seconds
    # TO BE INITIALISED BEFORE LOOP
    T = 0.001 #correct for sim

    lambda_d = np.array([0,0,0])#.reshape([3,1])
    lambda_d_history = np.zeros((3,max_num_it))
    r_d = get_r()
    r_d_history = np.zeros((3,max_num_it)) 


    #for plotting
    controlled_pose = np.zeros((3,max_num_it))
    controlled_wrench = np.zeros((3,max_num_it))

    for i in range(max_num_it):
        # IN LOOP:

        lambda_d = get_lambda_d(i,lambda_d)
        lambda_d_history[:,i]=lambda_d

        r_d = get_r_d(i,r_d)
        r_d_history[:,i] = r_d

        f_lambda = calculate_f_lambda(i, T, S_f ,C , K_Dlambda, K_Plambda, lambda_d_history, lambda_d)
        alpha_v = calculate_alpha_v(i,T,r_d_history, r_d,K_Pr,K_Dr)
        alpha = calculate_alpha(S_v,alpha_v,C,S_f,f_lambda)
        perform_torque(alpha)

        # plotting and printing
        if i%100 == 0:
            print(i,') True Force in z: ',robot.endpoint_effort()['force'][2])

        controlled_pose[:,i] = get_r()
        controlled_wrench[:,i] = get_lambda()
    
    plot_result(controlled_wrench,lambda_d_history,controlled_pose,r_d_history)

