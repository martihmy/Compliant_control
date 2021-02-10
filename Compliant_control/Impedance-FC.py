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
#import panda as pd
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=2)

# --------- Constants -----------------------------

S_f = np.array([[0, 0, 1, 0, 0, 0]]).reshape([6,1])

S_v = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]]).reshape([6,5])

#should be estimated (this one is chosen at random)
K = np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 100, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 1]]).reshape([6,6])

C = np.linalg.inv(K)


K_Plambda =30 #random (force gains)
K_Dlambda = K_Plambda*0.633 #K_Plambda*0.633 #random

#Position:
Pp = 120 #proportional gain for position (x and y)
Dp = 12# Pp*0.633 #damping position (x and y)

#Orientation
Po = 20 #proportional gain for orientation
Do = 40#damping_orientation

K_Pr = np.array([[Pp, 0, 0, 0, 0],
                [0, Pp, 0, 0, 0],
                [0, 0, Po, 0, 0],
                [0, 0, 0, Po, 0],
                [0, 0, 0, 0, Po]])

K_Dr = np.array([[Dp, 0, 0, 0, 0],
                [0, Dp, 0, 0, 0],
                [0, 0, Do, 0, 0],
                [0, 0, 0, Do, 0],
                [0, 0, 0, 0, Do]])

# ------------ Helper functions --------------------------------

# GET STUFF
def get_joint_velocities():
    return np.array([robot.joint_velocity(robot.joint_names()[0]),robot.joint_velocity(robot.joint_names()[1]),robot.joint_velocity(robot.joint_names()[2]),robot.joint_velocity(robot.joint_names()[3]),robot.joint_velocity(robot.joint_names()[4]),robot.joint_velocity(robot.joint_names()[5]),robot.joint_velocity(robot.joint_names()[6])])

def get_v():
    return (np.array([robot.endpoint_velocity()['linear'][0],robot.endpoint_velocity()['linear'][1],robot.endpoint_velocity()['angular'][0],robot.endpoint_velocity()['angular'][1],robot.endpoint_velocity()['angular'][2]])).reshape([5,1])


def get_r():
    quat_as_list = np.array([robot.endpoint_pose()['orientation'].x,robot.endpoint_pose()['orientation'].y,robot.endpoint_pose()['orientation'].z,robot.endpoint_pose()['orientation'].w])
    rot = Rotation.from_quat(quat_as_list)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],(rot_euler[0]-np.sign(rot_euler[0])*180),rot_euler[1],rot_euler[2]])


def get_lambda():
    return robot.endpoint_effort()['force'][2]
    #return 0 #fake feedback 


def get_lambda_d(i,original_d=15):
    if i < 1500:
        return  float(i)/100
    elif i > 2000 and i < 4000:
        new_lambda_d = original_d + 5*np.sin(i*0.001*2*np.pi)
        return new_lambda_d
    else:
        return original_d

def get_r_d(i,current_r_d):
    if i > 4500 and i < 6500:
        new_r_d = current_r_d + np.array([0.0001,0,0,0,0]) #adding to x
        return new_r_d
    else:
        return current_r_d

def get_F_d(max_num_it,T):
    a = np.zeros(max_num_it)
    a[0:10]=0.01
    a[20:29]=-0.01
    a[30] = -0.002
    if max_num_it >4001:
        a[1500:1508]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s




def get_S_inv(S,C):
    a = np.linalg.inv(np.linalg.multi_dot([S.T,C,S]))
    return np.array(np.linalg.multi_dot([a,S.T,C]))
    
def get_K_dot(S_f,S_f_inv,C):
    return np.array(np.linalg.multi_dot([S_f,S_f_inv,np.linalg.inv(C)])).reshape([6,6])

def get_dot_list(history,iteration,T):
    if iteration > 0:
        return ((history[:,iteration]-history[:,iteration-1]).reshape([5,1])/T).reshape([5,1])
    else:
        return np.zeros(5).reshape([5,1])

def get_ddot_list(history,iteration,T):
    if iteration > 1:
        return ((get_dot_list(history,iteration,T) - get_dot_list(history,iteration-1,T)).reshape([5,1])/T).reshape([5,1])
    else:
        return np.zeros(5).reshape([5,1])

def get_dot_scalar(history,iteration,T):
    if iteration > 0:
        return ((history[iteration]-history[iteration-1])/T)
    else:
        return 0

def get_ddot_scalar(history,iteration,T):
    if iteration > 1:
        return (get_dot_scalar(history,iteration,T) - get_dot_scalar(history,iteration-1,T))/T
    else:
        return 0

def low_pass(history, iteration):
    if i > 1:
        return (history[i]+history[i-1])/2
    else:
        return history[i]
    

# CALCULATE TORQUE
def calculate_f_lambda(f_d_ddot, f_d_dot, f_d, i,T, S_f,C,K_Dlambda,K_Plambda, z_force):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()]))
    #lambda_a = get_ddot_scalar(lambda_d_history,i,T)
    lambda_a = f_d_ddot
    #lambda_b = np.array(np.dot(K_Dlambda,(get_dot_scalar(lambda_d_history,i,T)-lambda_dot)))
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    #return -(get_ddot_scalar(lambda_d_history,i,T) + np.array(np.dot(K_Dlambda,(get_dot_scalar(lambda_d_history,i,T)-lambda_dot))) + np.dot(K_Plambda,(lambda_d-z_force)))
    return lambda_a, lambda_b, lambda_c

def calculate_alpha_v(i,T,r_d_history,r_d,K_Pr,K_Dr):
    r_d_dot = get_dot_list(r_d_history,i,T)
    r_d_ddot = get_ddot_list(r_d_history,i,T)
    return (r_d_ddot + np.array(np.dot(K_Dr,r_d_dot-get_v())).reshape([5,1])+ np.array(np.dot(K_Pr, r_d.reshape([5,1])-(get_r()).reshape([5,1]))).reshape([5,1]))

def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_inv(S_v,C)
    P_v = np.array(np.dot(S_v,S_v_inv))
    C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
    return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + f_lambda*np.array(np.dot(C_dot,S_f)).reshape([6,1])

def perform_torque(alpha,z_offset):
    cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([robot.jacobian(),np.linalg.inv(robot.joint_inertia_matrix()),robot.jacobian().T]))
    alpha_torque = np.array(np.linalg.multi_dot([robot.jacobian().T,cartesian_inertia,alpha])).reshape([7,1])
    #external_torque = np.dot(robot.jacobian().T,np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])).reshape([7,1])
    external_torque = np.dot(robot.jacobian().T,np.array([0,0,robot.endpoint_effort()['force'][2]-z_offset,0,0,0])).reshape([7,1])
    torque = alpha_torque + robot.coriolis_comp().reshape([7,1])# - external_torque
    robot.set_joint_torques(dict(list(zip(robot.joint_names(),torque))))


def plot_result(f_controlled, f_d ,controlled_pose,x_d,z, f_lambda,T):

    time_array = np.arange(len(controlled_pose[0]))*T
    

    plt.subplot(221)
    plt.title("External force")
    
    plt.plot(time_array, f_controlled[:], label="force z [N]")
    #plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, f_d[:], label="desired force z [N]", color='b',linestyle='dashed')
    #plt.plot(time_array, f_d[1,:], label="desired torque x [Nm]", color='C1',linestyle='dashed')
    #plt.plot(time_array, f_d[2,:], label="desired torque y [Nm]", color='g',linestyle='dashed')

    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(222)
    plt.title("Position")

    plt.plot(time_array, controlled_pose[0,:], label = "true x [m]")
    plt.plot(time_array, controlled_pose[1,:], label = "true y [m]")
    plt.plot(time_array, z[:], label = "true z [m]")

    plt.plot(time_array, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
 


    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(223)
    plt.title("Orientation")

    plt.plot(time_array, controlled_pose[2,:], label = "true  Ori_x [degrees]")
    plt.plot(time_array, controlled_pose[3,:], label = "true  Ori_y [degrees]")
    plt.plot(time_array, controlled_pose[4,:], label = "true  Ori_z [degrees]")

    plt.plot(time_array, x_d[2,:], label = "desired Ori_x [degrees]", color='b',linestyle='dashed')
    plt.plot(time_array, x_d[3,:], label = "desired Ori_y [degrees]", color='C1',linestyle='dashed')
    plt.plot(time_array, x_d[4,:], label = "desired Ori_z [degrees]", color='g',linestyle='dashed')

    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(224)
    plt.title("Applied control")
    plt.plot(time_array, f_lambda[0][:], label="f_lambda (a)")
    plt.plot(time_array, f_lambda[1][:], label="f_lambda (b)")
    plt.plot(time_array, f_lambda[2][:], label="f_lambda (c)")
    plt.plot(time_array, f_lambda[3][:], label="f_lambda (sum)")
    plt.xlabel("Real time [s]")
    plt.legend()
    
     


    plt.show()

# MAIN FUNCTION

if __name__ == "__main__":
    rospy.init_node("impedance_control")
    publish_rate = 250
    rate = rospy.Rate(publish_rate)
    
    robot = PandaArm()
    robot.move_to_neutral() 

    max_num_it=7500
    # TO BE INITIALISED BEFORE LOOP
    T = 0.001*(1000/publish_rate) #correct for sim

    #lambda_d = 10
    lambda_d_history = np.zeros(max_num_it)
    r_d = get_r()
    r_d_history = np.zeros((5,max_num_it)) 


    #for plotting
    controlled_pose = np.zeros((5,max_num_it))
    z_force_history = np.zeros(max_num_it)
    z_position = np.zeros(max_num_it)
    f_lambda_history = np.zeros((4,max_num_it))
    trajectory = np.zeros((3,max_num_it))

    f_d_ddot,f_d_dot, f_d = get_F_d(max_num_it,T)
    wrench_offsets = np.load('/home/martin/trajectory_wrenches.npy') ###
    z_offsets = wrench_offsets[2][:] ###
    #z_offsets = np.zeros(max_num_it)
    for i in range(max_num_it):
        # IN LOOP:
        """
        lambda_d = get_lambda_d(i)
        lambda_d_history[i]=lambda_d
        """
        r_d = get_r_d(i,r_d)
        r_d_history[:,i] = r_d

        z_force = get_lambda()-z_offsets[i] ###
        z_force_history[i] = z_force
        #z_force = low_pass(z_force_history,i) #####

        a,b,c = calculate_f_lambda(f_d_ddot[i], f_d_dot[i], f_d[i], i, T, S_f ,C , K_Dlambda, K_Plambda, z_force)
        f_lambda = (a+b+c)
        alpha_v = calculate_alpha_v(i,T,r_d_history, r_d,K_Pr,K_Dr)
        alpha = calculate_alpha(S_v,alpha_v,C,S_f,-f_lambda)
        perform_torque(alpha,z_offsets[i])
        rate.sleep()


        # plotting and printing
        if i%100 == 0:
            print(i,'= ',T*i,' [s]   ) Force in z: ',robot.endpoint_effort()['force'][2])
            print('f_lambda: ',f_lambda)
            print('')

        controlled_pose[:,i] = get_r()
        z_position[i] = robot.endpoint_pose()['position'][2]

        f_lambda_history[0][i] = a
        f_lambda_history[1][i] = b
        f_lambda_history[2][i] = c
        f_lambda_history[3][i] = f_lambda

        trajectory[:,i] = np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],robot.endpoint_pose()['position'][2]])#
    
    #np.save('trajectory.npy',trajectory)#
    plot_result(z_force_history,f_d,controlled_pose,r_d_history,z_position, f_lambda_history,T)

