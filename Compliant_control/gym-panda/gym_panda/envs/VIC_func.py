import gym
#from gym import ...

#! /usr/bin/env python
import copy
from copy import deepcopy
import rospy
import threading
import quaternion
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
#from franka_interface import ArmInterface
#from panda_robot import PandaArm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import signal

np.set_printoptions(precision=2)


"""



This is a HYBRID MOTION/FORCE CONTROLLER based on chapter 9.4.1 in the Handbook of Robotics 

It is doing force control along the z-axis, and motion control of the orientation and x and y positions.


About the code:

1] Due to the faulted joint velocities (read from rostopics), the more noisy, 
    numerically derived derivatives of the joint position are prefered to be 
        used in the controller { get_v(..., numerically = True) }

2]  Due to the faulted joint velocities, a more noisy (and precise) estimate of lambda_dot is considered.
            This is calculated in the function 'get_lambda_dot(...)' in an unorthodox matter

3] The default desired motion- and force-trajectories are now made in a time-consistent matter, so that the PUBLISH RATE can be altered without messing up the desired behaviour. 
    The number of iterations is calculated as a function of the controller's control-cycle, T: (max_num_it = duration(=15 s) / T)

4] IN THE FIRST LINE OF CODE BELOW "if __name__ == "__main__":" YOU HAVE TO SET SIM EQUAL TO "True" OR "False"
            - if True: starting position = neutral
            - if False: starting position = cartboard, Fz = (-) robot.endpoint_effort...

5] The time step (T) is now being explicitly calculated for each iteration due to its stochastic nature

"""


# --------- Parameters -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles
cartboard = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

"""Functions for generating desired MOTION trajectories"""

def generate_desired_trajectory(robot,max_num_it,T,move_down=True, move_in_x=True):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((3,max_num_it))
    
    s[:,0]= robot.endpoint_pose()['position']
    """
    if move_down:
        a[2,0:10]=-0.5
        a[2,30:40]= 0.5
    """
    s[2,0] -= 0.2 #0.1

    if move_in_x:
	a[0,int(max_num_it*4/10):int(max_num_it*6/10)]=0.05
        a[0,int(max_num_it*6/10):int(max_num_it*8/10)]=-0.05



    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:3,i-1]*T
    return a,v,s



"""Functions for generating desired FORCE trajectories"""

def generate_Fd_steep(max_num_it,Fd,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    v[2,:] = 10
    for i in range(max_num_it):
        if i>0:

            s[2,i]=min(s[2,i-1]+v[2,i-1]*T,Fd)
    return s

# Generate a constant desired force [STABLE]
def generate_Fd_constant(max_num_it,Fd):
    s = np.zeros((6,max_num_it)) 
    s[2,:] = Fd
    return s

# ------------ Helper functions --------------------------------
"""
# Compute difference between quaternions and return Euler angles as difference
def quatdiff_in_euler_radians(quat_curr, quat_des):
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
    return -des_mat.dot(vec)


# Calculate the numerical derivative of a each row in a vector
def get_derivative_of_vector(history,iteration,time_per_iteration):
    size = history.shape[0]
    if iteration > 0:
        T = float(time_per_iteration[iteration]-time_per_iteration[iteration-1])
        #return ((history[:,iteration]-history[:,iteration-1]).reshape([size,1])/T).reshape([size,1])
        if T>0:
            return np.subtract(history[:,iteration],history[:,iteration-1])/float(T)
    
    return np.zeros(size)#.reshape([size,1])

# Fetch the psudoinverse of S_f/S_v as in equation (9.34) in chapter 9.3 of The Handbook of Robotics
def get_S_inv(S,C):
    a = np.linalg.inv(np.linalg.multi_dot([S.T,C,S]))
    return np.array(np.linalg.multi_dot([a,S.T,C]))

# Fetch K' as in equation (9.49) in chapter 9.3 of The Handbook of Robotics
def get_K_dot(S_f,S_f_inv,C):
    return np.array(np.linalg.multi_dot([S_f,S_f_inv,np.linalg.inv(C)])).reshape([6,6])

# Fetch a simplified, less noisy estimate of the derivative of the external force in z 
def get_lambda_dot(S_f_inv,h_e_hist,i,time_per_iteration):
    h_e_dot = get_derivative_of_vector(h_e_hist,i,time_per_iteration)
    cap = 50
    if abs(h_e_dot[2]) > cap:
        h_e_dot[2] = np.sign(h_e_dot[2])*cap
    return np.dot(S_f_inv,h_e_dot)

# Fetch joint velocities
def get_joint_velocities(joint_names):
    dict = robot.joint_velocities()
    return np.array([ dict[joint_names[0]],dict[joint_names[1]],dict[joint_names[2]],dict[joint_names[3]],dict[joint_names[4]],dict[joint_names[5]],dict[joint_names[6]]])

# Calculate the error in position and orientation (in the subspace subject to motion control)
def get_delta_r(ori,goal_ori, p, p_d, two_dim = True):
    delta_pos = p_d - p[:2]
    delta_ori = quatdiff_in_euler_radians(np.asarray(ori), goal_ori)    
    if two_dim == True:
        return np.array([np.append(delta_pos,delta_ori)]).reshape([5,1])

    else:
        return np.append(delta_pos,delta_ori)

"""
    # Calculate the numerical derivative of a each row in a vector
def get_derivative_of_vector(history,iteration,time_per_iteration):
    size = history.shape[0]
    if iteration > 0:
        T = float(time_per_iteration[int(iteration)]-time_per_iteration[int(iteration-1)])
        if T > 0:
            return np.subtract(history[:,iteration],history[:,iteration-1])/T
    
    return np.zeros(size)

    # Return the error in linear and angular velocities
def get_x_dot_delta(x_d_dot,x_dot, two_dim = True):
    if two_dim == True:
        return (x_d_dot - x_dot).reshape([6,1])
    else:
        return x_d_dot - x_dot

    # Return the error in linear and angular acceleration
def get_x_ddot_delta(x_d_ddot,v_history,i,time_per_iteration):
    a = get_derivative_of_vector(v_history,i,time_per_iteration)
    return x_d_ddot-a

# Saturation-function
def ensure_limits(lower,upper,matrix):   
    for i in range(6):
        if matrix[i,i] > upper:
            matrix[i,i] = upper
        elif matrix[i,i] < lower:
            matrix[i,i] = lower
    return matrix
# ------------ Main functions (adaptive stiffness and damping) --------------------------------------

# Calculate lambda_dot as in equation (50) in [Huang1992] 
def get_lambda_dot(gamma,xi,K_v,P,F_d,F_ext_2D,i,time_per_iteration,T):
    if i > 0:
        T = float(time_per_iteration[i]-time_per_iteration[i-1])
    return np.linalg.multi_dot([-np.linalg.inv(gamma),xi.T,np.linalg.inv(K_v),P,F_ext_2D-F_d.reshape([6,1])])*T


# Get xi as it is described in equation (44) in [Huang1992]
def get_xi(x_dot, x_d_dot, x_d_ddot, delta_x,  x_dot_history, i, time_per_iteration):
    E = -delta_x
    E_dot = -get_x_dot_delta(x_d_dot,x_dot, two_dim = False)
    E_ddot = -get_x_ddot_delta(x_d_ddot,x_dot_history,i,time_per_iteration)
    E_diag = np.diagflat(E)
    E_dot_diag = np.diagflat(E_dot)
    E_ddot_diag = np.diagflat(E_ddot)
    return np.block([E_ddot_diag,E_dot_diag,E_diag])


def update_MBK_hat(lam,B,K,B_hat_limits,K_hat_limits):

    B_hat = B + np.diagflat(lam[6:12])
    K_hat = K + np.diagflat(lam[12:18])
    
    B_hat = ensure_limits(B_hat_limits[0],B_hat_limits[1],B_hat)
    K_hat = ensure_limits(K_hat_limits[0],K_hat_limits[1],K_hat)
    return B_hat, K_hat

# Compute difference between quaternions and return Euler angles as difference
def quatdiff_in_euler_degrees(quat_curr, quat_des):
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
    return -des_mat.dot(vec)*(180/np.pi)




# -------------- Helper functions (calculating torque) ------------------

def from_three_to_six_dim(matrix):
    return np.block([[matrix,np.zeros((3,3))],[np.zeros((3,3)),matrix]])

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

def get_K_Pt_dot(R_d,K_pt,R_e):
    return np.array([0.5*np.linalg.multi_dot([R_d,K_pt,R_d.T])+0.5*np.linalg.multi_dot([R_e,K_pt,R_e.T])])

def get_K_Pt_ddot(p_d,R_d,K_pt,delta_x):
    return np.array([0.5*np.linalg.multi_dot([skew(delta_x[:3]),R_d,K_pt,R_d.T])])

def E_quat(quat_n,quat_e): 
    return np.dot(quat_n,np.identity(3))-skew(quat_e)

def get_K_Po_dot(quat_n,quat_e,R_e,K_po): 
    return np.array([2*np.linalg.multi_dot([E_quat(quat_n,quat_e).T,R_e,K_po,R_e.T])])

def get_h_delta(K_pt_dot,K_pt_ddot,p_delta,K_po_dot,quat_e):
    f_delta_t = np.array([np.dot(K_pt_dot,p_delta)])
    m_delta_t = np.array([np.dot(K_pt_ddot,p_delta)])
    null = np.zeros((3,1))
    m_delta_o = np.array([np.dot(K_po_dot,quat_e)])
    
    return np.array([np.append(f_delta_t.T,m_delta_t.T)]).T + np.array([np.append(null.T,m_delta_o.T)]).T


def get_W(jacobian,robot_inertia, inv = False):
    W = np.linalg.multi_dot([jacobian,np.linalg.inv(robot_inertia),jacobian.T])
    if inv == True:
        return np.linalg.inv(W)
    else:
        return W

# -------------- Main functions  (calculating torque)  --------------------

def perform_torque_DeSchutter(robot,M, B, K, x_d_ddot, x_d_dot,x_dot,delta_x, p_d, Rot_e,Rot_d,F_ext_2D,jacobian, robot_inertia, coriolis_comp, joint_names): # must include Rot_d
    Rot_e_bigdim = from_three_to_six_dim(Rot_e)
    Rot_e_dot = np.dot(skew(x_dot[3:]),Rot_e) #not a 100 % sure about this one
    Rot_e_dot_bigdim = from_three_to_six_dim(Rot_e_dot)
    
    
    quat = quaternion.from_rotation_matrix(np.dot(Rot_e.T,Rot_d)) #orientational displacement represented as a unit quaternion
    quat_e_e = np.array([quat.x,quat.y,quat.z]) # vector part of the unit quaternion in the frame of the end effector
    quat_e = np.dot(Rot_e.T,quat_e_e) # ... in the base frame
    quat_n = quat.w
        
    p_delta = delta_x[:3]

    K_Pt_dot = get_K_Pt_dot(Rot_d,K[:3,:3],Rot_e)
    K_Pt_ddot = get_K_Pt_ddot(p_d,Rot_d,K[:3,:3],delta_x)
    K_Po_dot = get_K_Po_dot(quat_n,quat_e,Rot_e,K[3:,3:])

    h_delta_e = np.array(np.dot(Rot_e_bigdim,get_h_delta(K_Pt_dot,K_Pt_ddot,p_delta,K_Po_dot,quat_e))).reshape([6,1])
    h_e_e = np.array(np.dot(Rot_e_bigdim,F_ext_2D))

    a_d_e = np.dot(Rot_e_bigdim,x_d_ddot).reshape([6,1])
    v_d_e = np.dot(Rot_e_bigdim,x_d_dot).reshape([6,1])
    alpha_e = a_d_e + np.dot(np.linalg.inv(M),(np.dot(B,v_d_e.reshape([6,1])-np.dot(Rot_e_bigdim,x_dot).reshape([6,1]))+h_delta_e-h_e_e)).reshape([6,1])
    alpha = np.dot(Rot_e_bigdim.T,alpha_e).reshape([6,1])+np.dot(Rot_e_dot_bigdim.T,np.dot(Rot_e_bigdim,x_dot)).reshape([6,1])
    torque = np.linalg.multi_dot([jacobian.T,get_W(jacobian, robot_inertia, inv=True),alpha]).reshape((7,1)) + np.array(coriolis_comp.reshape((7,1))) + np.dot(jacobian.T,F_ext_2D).reshape((7,1))
    robot.set_joint_torques(dict(list(zip(joint_names,torque))))



# -------------- Plotting ------------------------

def plot_run(data,list_of_limits):

    adjusted_time_per_iteration = data[11,:] - data[11,0]
    new_list = np.zeros(len(data[0]))
    new_list[0]=adjusted_time_per_iteration[1] # just so that the first element isn't 0
    for i in range(len(adjusted_time_per_iteration)):
        if i >0:
            new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]

    plt.subplot(241)
    plt.title("External force")
    plt.plot(adjusted_time_per_iteration, data[0], label="force z [N]")
    plt.plot(adjusted_time_per_iteration, data[1], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(242)
    plt.title("Position")
    plt.plot(adjusted_time_per_iteration, data[2], label = "true x [m]")
    plt.plot(adjusted_time_per_iteration, data[3], label = "true y [m]")
    plt.plot(adjusted_time_per_iteration, data[4], label = "true z [m]")
    plt.plot(adjusted_time_per_iteration, data[5], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, data[6], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, data[7], label = "desired z [m]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

    
    plt.subplot(243)
    plt.title("Orientation error")
    plt.plot(adjusted_time_per_iteration, data[8], label = "error  Ori_x [degrees]")
    plt.plot(adjusted_time_per_iteration, data[9], label = "error  Ori_y [degrees]")
    plt.plot(adjusted_time_per_iteration, data[10], label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(244)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(245)
    plt.title("Varying adaptive rate of damping in z (learning)")
    plt.axhline(y=list_of_limits[1], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[12], label = "adaptive rate")
    plt.axhline(y=list_of_limits[0], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(246)
    plt.title("Varying adaptive rate of stiffness in z (learning)")
    plt.axhline(y=list_of_limits[3], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[13], label = "adaptive rate")
    plt.axhline(y=list_of_limits[2], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(247)
    plt.title("Varying stiffness in x and y (learning)")
    plt.axhline(y=list_of_limits[5], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[16], label = "stiffness over time")
    plt.axhline(y=list_of_limits[4], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    plt.subplot(248)
    plt.title("Varying damping and stiffness in z (result of learning) ")
    #plt.axhline(y=list_of_limits[7], label = 'upper bound', color='b', linestyle = 'dashed')
    plt.plot(data[14], label = "damping in z")
    #plt.axhline(y=list_of_limits[6], label = 'lower bound', color='b', linestyle = 'dashed')
    #plt.axhline(y=list_of_limits[9], label = 'upper bound', color='C1', linestyle = 'dashed')
    plt.plot(data[15], label = "stiffness in z")
    #plt.axhline(y=list_of_limits[8], label = 'lower bound', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()

    print('\a') #make a sound
    plt.show()

# move to neutral or alternative starting position (Dependent on sim/not sim)
