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


# is adjusted continously
K = 50 * np.identity(6)
B = K*0.633
M = np.identity(6)

#
K_v = 0.5*np.identity(6)

P = np.identity(6)
gamma = np.identity(18)



max_num_it = 500

def get_derivative_of_vector(history,iteration,T):
    size = history.shape[0]
    if iteration > 0:
        #return ((history[:,iteration]-history[:,iteration-1]).reshape([size,1])/T).reshape([size,1])
        return np.subtract(history[:,iteration],history[:,iteration-1])/T
    else:
        return np.zeros(size)#.reshape([size,1])

def get_xi(goal_ori, p_d, x_d_dot, x_d_ddot, v_history, i, T):
    E = -get_delta_x(goal_ori, p_d)
    E_dot = -get_x_dot_delta(x_d_dot, two_dim = False)
    E_ddot = -get_x_ddot_delta(x_d_ddot,v_history,i,T)
    E_diag = np.diagflat(E)
    E_dot_diag = np.diagflat(E_dot)
    E_ddot_diag = np.diagflat(E_ddot)
    return np.block([E_diag,E_dot_diag,E_ddot_diag])

def get_lambda_dot(gamma,xi,K_v,P,F_d):
    return np.linalg.multi_dot([-np.linalg.inv(gamma),xi.T,np.linalg.inv(K_v),P,get_F_ext(two_dim=True)-F_d.reshape([6,1])])

def update_MBK_hat(lam,M,B,K):
    return  M + np.diagflat(lam[0:6]), B + np.diagflat(lam[6:12]), K + np.diagflat(lam[12:18])
    


def get_W(inv = False):
    if inv == True:
        return np.linalg.inv(np.linalg.multi_dot([robot.jacobian(),np.linalg.inv(robot.joint_inertia_matrix()),robot.jacobian().T]))
        
    else:
        return np.linalg.multi_dot([robot.jacobian(),np.linalg.inv(robot.joint_inertia_matrix()),robot.jacobian().T])


def get_F_ext(two_dim = False):
    if two_dim == True:
        return np.array([0,0,robot.endpoint_effort()['force'][2],0,0,0]).reshape([6,1])
    else:
        return np.array([0,0,robot.endpoint_effort()['force'][2],0,0,0])

def get_p(two_dim=False):
    if two_dim == True:
        return robot.endpoint_pose()['position'].reshape([3,1])
    else:
        return robot.endpoint_pose()['position']

def get_x_dot():
    a = np.append(robot.endpoint_velocity()['linear'],robot.endpoint_velocity()['angular'])
    b = np.zeros(6)
    return b
    #return np.append(robot.endpoint_velocity()['linear'],robot.endpoint_velocity()['angular'])

def get_delta_x(goal_ori, p_d, two_dim = False):
    delta_pos = p_d - robot.endpoint_pose()['position']
    delta_ori = quatdiff_in_euler(np.asarray(robot.endpoint_pose()['orientation']), goal_ori)    
    if two_dim == True:
        return np.array([np.append(delta_pos,delta_ori)]).reshape([6,1])

    else:
        return np.append(delta_pos,delta_ori)

def get_x_dot_delta(x_d_dot, two_dim = True):
    if two_dim == True:
        return (x_d_dot - get_x_dot()).reshape([6,1])
    else:
        return x_d_dot - get_x_dot()


def get_x_ddot_delta(x_d_ddot,v_history,i,T):
    a = get_derivative_of_vector(v_history,i,T)
    return x_d_ddot-a

def get_desired_trajectory(iterations,T):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    if iterations > 300:
        a[2,0:5]=-0.00002/T**2
        a[2,295:300]=0.00002/T**2
    if iterations > 6500:
        a[0,4500:4510]=0.00001/T**2
        a[0,6490:6500]=-0.00001/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p



def perform_torque(M, B, K, x_d_ddot, x_d_dot, p_d, goal_ori):
    a = np.linalg.multi_dot([robot.jacobian().T,get_W(inv=True),np.linalg.inv(M)])
    b = np.array([np.dot(M,x_d_ddot)]).reshape([6,1]) + np.array([np.dot(B,get_x_dot_delta(x_d_dot))]).reshape([6,1]) + np.array([np.dot(K,get_delta_x(goal_ori,p_d,two_dim = True))]).reshape([6,1])
    c = robot.coriolis_comp().reshape([7,1])
    d = (np.identity(6)-np.dot(get_W(inv=True),np.linalg.inv(M))).reshape([6,6])
    total_torque = np.array([np.dot(a,b)]).reshape([7,1]) + c #+ np.array([np.linalg.multi_dot([robot.jacobian().T,d,get_F_ext()])]).reshape([7,1])
    robot.set_joint_torques(dict(list(zip(robot.joint_names(),total_torque))))

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

def get_F_d(max_num_it,T): #current
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s[2,0]=15
    """
    a[2,0:100] = 0.0005/T**2
    a[2,100:200] = - 0.0005/T**2
    if max_num_it > 1100:
        a[2,500:550] = 0.0002/T**2
    if max_num_it >4001:
        a[2,1500:1550]=-0.0002/T**2
        it = 2000
        while it <= 4000:
            a[2,it]= (-9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1

        a[2,4001]=0.0001/T**2
    """
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

def plot_result(p,p_d, delta_x, F_ext,F_d, z_stiffness, T):

    time_array = np.arange(len(p[0]))*T
    

    plt.subplot(221)
    plt.title("External force")
    plt.plot(time_array, F_ext[2], label="force z [N]")
    plt.plot(time_array, F_d[2], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(222)
    plt.title("Position")
    plt.plot(time_array, p[0,:], label = "true x [m]")
    plt.plot(time_array, p[1,:], label = "true y [m]")
    plt.plot(time_array, p[2,:], label = "true z [m]")

    plt.plot(time_array, p_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, p_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(time_array, p_d[2,:], label = "desired z [m]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(223)
    plt.title("Orientation error")
    plt.plot(time_array, delta_x[3], label = "error  Ori_x [degrees]")
    plt.plot(time_array, delta_x[4], label = "error  Ori_y [degrees]")
    plt.plot(time_array, delta_x[5], label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(224)
    plt.title("Stiffness along z")
    plt.plot(time_array, z_stiffness, label = "stiffness in z")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.show()





if __name__ == "__main__":
    rospy.init_node("impedance_control")
    publish_rate = 250
    rate = rospy.Rate(publish_rate)
    
    robot = PandaArm()
    robot.move_to_neutral() 

    
    # TO BE INITIALISED BEFORE LOOP
    T = 0.001*(1000/publish_rate) #correct for sim

    #lambda_d = 10
    

    lam = np.zeros(18)
    #for plotting
    p_history = np.zeros((3,max_num_it))
    v_history = np.zeros((6,max_num_it))

    delta_x_history = np.zeros((6,max_num_it))

    F_ext_history = np.zeros((6,max_num_it))


    z_stiffness_history = np.zeros(max_num_it)

    x_d_ddot, x_d_dot, p_d = get_desired_trajectory(max_num_it,T)
    goal_ori = np.asarray(robot.endpoint_pose()['orientation'])
    F_d = get_F_d(max_num_it,T)

    for i in range(max_num_it):
        # update state-lists
        p_history[:,i] = get_p()
        delta_x_history[:,i] = get_delta_x(goal_ori,p_d[:,i])
        F_ext_history[:,i] = get_F_ext()
        v_history[:,i] = get_x_dot() #positioning is important

        # adapt M,B and K
        xi = get_xi(goal_ori, p_d[:,i], x_d_dot[:,i], x_d_ddot[:,i], v_history, i, T)        
        lam = lam.reshape([18,1]) + get_lambda_dot(gamma,xi,K_v,P,F_d[:,i]).reshape([18,1]) 
        M_hat,B_hat,K_hat = update_MBK_hat(lam,M,B,K)
        perform_torque(M_hat, B_hat, K_hat, x_d_ddot[:,i], x_d_dot[:,i], p_d[:,i], goal_ori)
        rate.sleep()


        # plotting and printing
        z_stiffness_history[i]=K_hat[2][2]
        if i%100 == 0:
            print(i,'/',max_num_it,' = ',T*i,' [s]   ) Force in z: ',F_ext_history[2,i])
            print(K_hat[2][2])
            print('')


        


        #trajectory[:,i] = np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],robot.endpoint_pose()['position'][2]])#
    
    #np.save('trajectory.npy',trajectory)#
    plot_result(p_history, p_d, delta_x_history, F_ext_history, F_d, z_stiffness_history, T)

