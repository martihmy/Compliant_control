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

def generate_desired_trajectory(robot,max_num_it,T,move_in_x=True):
    a = np.zeros((5,max_num_it))
    v = np.zeros((5,max_num_it))
    s = np.zeros((2,max_num_it))
    
    s[:,0]= robot.endpoint_pose()['position'][0:2]

    if move_in_x:
        a[0,int(max_num_it*4/10):int(max_num_it*6/10)]=0.05
        a[0,int(max_num_it*6/10):int(max_num_it*8/10)]=-0.05

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:2,i-1]*T
    return a,v,s



"""Functions for generating desired FORCE trajectories"""

def generate_Fd_steep(max_num_it,T,f_d):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    v[0]=10


    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T
            if s[i]>3:
                s[i] = f_d
                v[i]=0

    return a,v,s


# Generate a constant desired force [STABLE]
def generate_Fd_constant(max_num_it):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)+3
    return a,v,s

# ------------ Helper functions --------------------------------

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



# ------------ Main functions --------------------------------------

# Calculate f_lambda (part of equation 9.62) as in equation (9.65) in chapter 9.3 of The Handbook of Robotics
def get_f_lambda(f_d_ddot, f_d_dot, f_d, i,time_per_iteration, S_f,C,K_Dlambda,K_Plambda, z_force,h_e_hist,jacobian,joint_v,joint_names,sim):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    if sim: 
       #lambda_dot = get_lambda_dot(S_f_inv,h_e_hist,i,time_per_iteration)
       lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,jacobian,joint_v]))
    else: 
        lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,jacobian,joint_v])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
    lambda_a = f_d_ddot
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    return max(lambda_a + lambda_b + lambda_c,0)

# Calculate alpha_v (part of equation 9.62) as on page 213 in chapter 9.3 of The Handbook of Robotics
def calculate_alpha_v(i, ori, goal_ori, r_d_ddot, r_d_dot, p,p_d,K_Pr,K_Dr,v):
    return (r_d_ddot.reshape([5,1]) + np.array(np.dot(K_Dr,r_d_dot.reshape([5,1])-v)).reshape([5,1])+ np.array(np.dot(K_Pr,get_delta_r(ori,goal_ori,p,p_d))).reshape([5,1]))


# Calculate alpha (part of equation 9.16) as in equation (9.62) in chapter 9.3 of The Handbook of Robotics
def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_inv(S_v,C)
    P_v = np.array(np.dot(S_v,S_v_inv))
    C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
    return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + f_lambda*np.array(np.dot(C_dot,S_f)).reshape([6,1])





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



#Return only the force in z
def get_Fz(sim=False):
    if sim:
        return robot.endpoint_effort()['force'][2]
    else:
        return -robot.endpoint_effort()['force'][2]

# -------------- Main functions --------------------




# -------------- Plotting ------------------------

def plot_result(time_per_iteration,force_z_raw,x_c,pos,F_d_raw,x_d,ori_error,T,publish_rate,joints,joints_d):
    print('')
    print('Constructing plot...')
    print('')
    
    force_z = force_z_raw - force_z_raw[0] #remove offset
    Fz_d_raw = F_d_raw[2]
    Fz_d = Fz_d_raw- Fz_d_raw[0] #remove offset
    
    adjusted_time_per_iteration = time_per_iteration - time_per_iteration[0]
    new_list = np.zeros(len(force_z_raw))
    new_list[0]=T#adjusted_time_per_iteration[0]
    for i in range(len(adjusted_time_per_iteration)):
        if i >0:
            new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]
            


    plt.subplot(221)
    plt.title("External force")
    plt.plot(adjusted_time_per_iteration, force_z, label="force z [N]")
    plt.plot(adjusted_time_per_iteration, Fz_d, label = " desired z-force [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(222)
    plt.title("Positional adjustments in z")
    plt.plot(adjusted_time_per_iteration, pos[2,:], label = "true  z [m]")
    plt.plot(adjusted_time_per_iteration, x_d[2,:], label = "desired z [m]",linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, x_c[2,:], label = "compliant z [m]",linestyle='dotted')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(223)
    plt.title("position in x and y")
    plt.plot(adjusted_time_per_iteration, pos[0,:], label = "true x [m]")
    plt.plot(adjusted_time_per_iteration, pos[1,:], label = "true y [m]")
    plt.plot(adjusted_time_per_iteration, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    
    plt.subplot(224)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.axhline(y=1/float(publish_rate), label = 'desired time-step', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
    """


    plt.subplot(224)
    plt.title("joint positions")
    plt.plot(adjusted_time_per_iteration, joints[0], label = "0")
    plt.plot(adjusted_time_per_iteration, joints[1], label = "1")
    plt.plot(adjusted_time_per_iteration, joints[2], label = "2")
    plt.plot(adjusted_time_per_iteration, joints[3], label = "3")
    plt.plot(adjusted_time_per_iteration, joints[4], label = "4")
    plt.plot(adjusted_time_per_iteration, joints[5], label = "5")
    plt.plot(adjusted_time_per_iteration, joints[6], label = "6")
    plt.plot(adjusted_time_per_iteration, joints_d[0], label = "0d", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[1], label = "1d", color='C1',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[2], label = "2d", color='C2',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[3], label = "3d", color='C3',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[4], label = "4d", color='C4',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[5], label = "5d", color='C5',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, joints_d[6], label = "6d", color='C6',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    """
    plt.subplot(224)
    plt.title("Error in orientation")
    plt.plot(adjusted_time_per_iteration, ori_error[0,:], label = "true  Ori_x [degrees]")
    plt.plot(adjusted_time_per_iteration, ori_error[1,:], label = "true  Ori_y [degrees]")
    plt.plot(adjusted_time_per_iteration, ori_error[2,:], label = "true  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    
    plt.show()
    print('Plot should be showing now')

# move to neutral or alternative starting position (Dependent on sim/not sim)
def move_to_start(alternative_position, sim):
    if sim:
        robot.move_to_neutral()
    else:
        robot.move_to_joint_positions(alternative_position)

def fetch_states(robot,sim):
    x,Fz_raw = robot.fetch_states_admittance()
    if sim:
        Fz = Fz_raw
    else:
        Fz = -Fz_raw

    return x,Fz



def perform_action(action,B,K,increment):
        #indexes of action space:

        #                       Damping (B)
        #                   ---------------------#
        #                   0       1       2   # (0-2): increase K
        # Stiffness (K)     3       4       5   # (3-5): don't change K
        #                   6       7       8   # (6-8): decrease K
        #                   ---------------------#
        #               (0,3,6): decrease B
        #                       (1,4,7): don't change B
        #                               (2,5,8): increase B
    if action < 3:
        K = K + increment
        if action == 0:
            B = B - increment
        if action == 2:
            B = B + increment
    
    elif action in range(3,5):
        if action == 3:
            B = B - increment
        if action == 5:
            B = B + increment

    elif action > 5:
        K = K - increment
        if action == 6:
            B = B - increment
        if action == 8:
            B = B + increment

    return B,K


