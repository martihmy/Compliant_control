#! /usr/bin/env python
import copy
from copy import deepcopy
import rospy
import threading
import quaternion
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
from franka_interface import ArmInterface
from panda_robot import PandaArm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
np.set_printoptions(precision=2)
"""

This is a FORCE-BASED VARIABLE IMPEDANCE CONTROLLER based on [Huang1992: Compliant Motion Control of Robots by Using Variable Impedance]

To achieve force tracking, the apparent stiffness (K) and damping (B) is dynamically adjusted through functions dependent on the error in position, velocity and force




About the code/controller:

1] Only stiffness and damping in the 'z'-direction is adaptive, the rest are static

2] Due to the faulted joint velocities (read from rostopics), the more noisy, 
    numerically derived derivatives of the joint position are prefered to be 
        used in the controller { get_x_dot(..., numerically = True) }

3] You can now choose between perform_torque_Huang1992() and perform_torque_DeSchutter()
    - DeSchutter's control-law offers geometrically consistent stiffness and is more computationally expensive

4] The default desired motion- and force-trajectories are now made in a time-consistent matter, so that the PUBLISH RATE can be altered without messing up the desired behaviour. 
    The number of iterations is calculated as a function of the controller's control-cycle, T: (max_num_it = duration(=15 s) / T)

5] IN THE FIRST LINE OF CODE BELOW "if __name__ == "__main__":" YOU HAVE TO SET SIM EQUAL TO "True" OR "False"
            - if True: starting position = neutral
            - if False: starting position = cartboard, Fz = (-) robot.endpoint_effort...
"""
# --------- Constants -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles

cartboard = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

# Stiffness
Kp = 30
Kpz = 50 #initial value (adaptive)
Ko = 900
K = np.array([[Kp, 0, 0, 0, 0, 0],
                [0, Kp, 0, 0, 0, 0],
                [0, 0, Kpz, 0, 0, 0],
                [0, 0, 0, Ko, 0, 0],
                [0, 0, 0, 0, Ko, 0],
                [0, 0, 0, 0, 0, Ko]])

# Damping 
Bp = Kp/30
Bpz = Bp # #initial value (adaptive)
Bo = 10
B = np.array([[Bp, 0, 0, 0, 0, 0],
                [0, Bp, 0, 0, 0, 0],
                [0, 0, Bpz, 0, 0, 0],
                [0, 0, 0, Bo, 0, 0],
                [0, 0, 0, 0, Bo, 0],
                [0, 0, 0, 0, 0, Bo]])

# Apparent inertia
Mp = 10
Mo = 10
M_diag = np.array([Mp,Mp,Mp,Mo,Mo,Mo])
M = np.diagflat(M_diag)

# Constant matrices appearing in equation (50) of [Huang1992]
K_v = np.identity(6)
P = np.identity(6)
gamma = np.identity(18)

#gamma_M = 12 
gamma_B = 0.001*10 #2    # The damping's rate of adaptivity (high value = slow changes)
gamma_K = 0.0005/10 #1    # The stiffness' rate of adaptivity (high value = slow changes)
#gamma[2,2] = gamma_M
gamma[8,8] = gamma_B
gamma[14,14] = gamma_K

duration = 15 #seconds SHOULD NOT BE ALTERED


"""Functions for generating desired MOTION trajectories"""

#1  Generate a desired trajectory for the manipulator to follow
def generate_desired_trajectory(iterations,T):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    
    if iterations > 300:
        a[2,0:100]=-0.00001/T**2
        a[2,250:350]=0.00001/T**2
        
    if iterations > 6500:
        a[0,4500:4510]=0.00001/T**2
        a[0,6490:6500]=-0.00001/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p

#2  Generate a desired trajectory for the manipulator to follow
def generate_desired_trajectory_express(iterations,T):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    
    if iterations > 175:
        a[2,0:50]=-0.00002/T**2
        a[2,50:100]=0.00002/T**2
        
    if iterations > 3250:
        a[0,2250:2255]=0.00002/T**2
        a[0,3245:3250]=-0.00002/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p

#3  Generate a (time-consistent) desired motion trajectory
def generate_desired_trajectory_tc(iterations,T,move_in_x=False, move_down=True):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    
    if move_down:
        a[2,0:int(iterations/75)]=-0.25
        a[2,int(iterations*2/75):int(iterations/25)]= 0.25


    if move_in_x:
        a[0,int(max_num_it*4/10):int(max_num_it*5/10)]=0.015
        a[0,int(max_num_it*7/10):int(max_num_it*8/10)]=-0.015

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p

"""Functions for generating desired FORCE trajectories"""

#1  Generate a desired force trajectory that takes offset into consideration
def generate_F_d_robot(max_num_it,T,sim=False):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s[2,0]= get_Fz(sim)
    a[2,0:100] = 0.0005/T**2
    a[2,100:200] = - 0.0005/T**2

    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

#2  Generate an efficient desired force trajectory 
def generate_F_d_express(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    
    a[2,0:50] = 0.0010/T**2
    a[2,100:150] = - 0.0010/T**2
    if max_num_it > 275:
        a[2,250:275] = 0.0008/T**2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    if max_num_it >2001:
        a[2,750:775]=-0.0008/T**2                                                                                       
        it = 1000
        while it <= 2000:
            a[2,it]= (-9*(np.pi**2)*(T/4)**2*np.sin(2*it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1
        a[2,2001]=0.0001/T**2                                                                                                                                                                                                                                 
    
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

#3  Generate a (time-consistent) desired force trajectory 
def generate_Fd_advanced(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))                                                                                                                                                          
    
    a[2,0:int(max_num_it/75)] = 62.5
    a[2,int(max_num_it/37.5):int(max_num_it/25)] = - 62.5                                     
    if max_num_it > 275:
        a[2,int(max_num_it/15):int(max_num_it*11/150)] = 50
    if max_num_it >2001:
        a[2,int(max_num_it/5):int(max_num_it*31/150)]=-50
        it = int(max_num_it*4/15)
        while it <= int(max_num_it*8/15):
            a[2,it]= (-9*(np.pi**2)*(T/4)**2*np.sin(2*it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1
        a[2,int(max_num_it*8/15+1)]=6.25
    
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

# generate a STEEP desired force trajectory [STABLE]
def generate_Fd_steep(max_num_it,T,sim=False):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    s[2,0]= get_Fz(sim)
    v[2,0]=2.5
    a[2,max_num_it/15:3*max_num_it/15] = - 1.25

    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

# ------------ Helper functions --------------------------------


# Calculate the numerical derivative of a each row in a vector
def get_derivative_of_vector(history,iteration,time_per_iteration):
    size = history.shape[0]
    if iteration > 0:
        T = float(time_per_iteration[int(iteration)]-time_per_iteration[int(iteration-1)])
        if T > 0:
            return np.subtract(history[:,iteration],history[:,iteration-1])/T
    
    return np.zeros(size)


# Saturation-function
def ensure_limits(lower,upper,matrix):   
    for i in range(6):
        if matrix[i,i] > upper:
            matrix[i,i] = upper
        elif matrix[i,i] < lower:
            matrix[i,i] = lower


# Return the cartesian (task-space) inertia of the manipulator [alternatively the inverse of it]
def get_W(jacobian,robot_inertia, inv = False):
    W = np.linalg.multi_dot([jacobian,np.linalg.inv(robot_inertia),jacobian.T])
    if inv == True:
        return np.linalg.inv(W)
    else:
        return W



# Return the external forces (everything except for z-force is set to 0 due to offsets)
def construct_F_ext(Fz,two_dim = False):
    if two_dim == True:
        return np.array([0,0,Fz,0,0,0]).reshape([6,1])
    else:

        return np.array([0,0,Fz,0,0,0])


#Return only the force in z
def get_Fz(sim=False):
    if sim:
        return robot.endpoint_effort()['force'][2]
    else:
        return -robot.endpoint_effort()['force'][2]


# Return the position and (relative) orientation 
def get_x(goal_ori):
    pos_x = robot.endpoint_pose()['position']
    rel_ori = quatdiff_in_euler_radians(np.asarray(robot.endpoint_pose()['orientation']), goal_ori)  #used to be opposite
    return np.append(pos_x,rel_ori)


# Return the linear and angular velocities
# Numerically = True -> return the derivarive of the state-vector
# Numerically = False -> read values from rostopic (faulty in sim when interacting with the environment)
def get_x_dot(x_hist,i,time_per_iteration, numerically=False):
    if numerically == True:
        return get_derivative_of_vector(x_hist,i,time_per_iteration)
    else:
        return np.append(robot.endpoint_velocity()['linear'],robot.endpoint_velocity()['angular'])



# Return the error in position and orientation
def get_delta_x(x, p_d, two_dim = False):
    delta_pos = p_d - x[:3]
    delta_ori = x[3:]
    if two_dim == True:
        return np.array([np.append(delta_pos,delta_ori)]).reshape([6,1])

    else:
        return np.append(delta_pos,delta_ori)


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


# Return the cartesian (task-space) position [only used in offline planning of the desired trajectory]
def get_p(two_dim=False):
    if two_dim == True:
        return robot.endpoint_pose()['position'].reshape([3,1])
    else:
        return robot.endpoint_pose()['position']


# Compute difference between quaternions and return Euler angle in radians as difference
def quatdiff_in_euler_radians(quat_curr, quat_des):
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec
    return -des_mat.dot(vec)

# -------------- Main functions --------------------

# Get xi as it is described in equation (44) in [Huang1992]
def get_xi(x_dot, x_d_dot, x_d_ddot, delta_x,  x_dot_history, i, time_per_iteration):
    E = -delta_x
    E_dot = -get_x_dot_delta(x_d_dot,x_dot, two_dim = False)
    E_ddot = -get_x_ddot_delta(x_d_ddot,x_dot_history,i,time_per_iteration)
    E_diag = np.diagflat(E)
    E_dot_diag = np.diagflat(E_dot)
    E_ddot_diag = np.diagflat(E_ddot)
    return np.block([E_ddot_diag,E_dot_diag,E_diag])


# Calculate lambda_dot as in equation (50) in [Huang1992] 
def get_lambda_dot(gamma,xi,K_v,P,F_d,F_ext_2D,i,time_per_iteration,T):
    if i > 0:
        T = float(time_per_iteration[i]-time_per_iteration[i-1])
    return np.linalg.multi_dot([-np.linalg.inv(gamma),xi.T,np.linalg.inv(K_v),P,F_ext_2D-F_d.reshape([6,1])])*T


# Return the updated (adapted) Inertia, Damping and Stiffness matrices.
def update_MBK_hat(lam,M,B,K):
    M_hat = M # + np.diagflat(lam[0:6]) M is chosen to be constant 
    B_hat = B + np.diagflat(lam[6:12])
    K_hat = K + np.diagflat(lam[12:18])
    #ensure_limits(1,5000,M_hat)
    ensure_limits(1,300,B_hat)
    ensure_limits(10,1000,K_hat)
    return M_hat, B_hat, K_hat


# Calculate and perform the torque as in equation (10) in [Huang1992]
def perform_torque_Huang1992(M, B, K, x_d_ddot, x_d_dot,x,x_dot, p_d, F_ext_2D, jacobian,robot_inertia,joint_names):
    a = np.linalg.multi_dot([jacobian.T,get_W(jacobian,robot_inertia,inv=True),np.linalg.inv(M)])
    b = np.array([np.dot(M,x_d_ddot)]).reshape([6,1]) + np.array([np.dot(B,get_x_dot_delta(x_d_dot,x_dot))]).reshape([6,1]) + np.array([np.dot(K,get_delta_x(x,p_d,two_dim = True))]).reshape([6,1])
    c = robot.coriolis_comp().reshape([7,1])
    d = (np.identity(6)-np.dot(get_W(jacobian,robot_inertia,inv=True),np.linalg.inv(M))).reshape([6,6])
    total_torque = np.array([np.dot(a,b)]).reshape([7,1]) + c + np.array([np.linalg.multi_dot([jacobian.T,d,F_ext_2D])]).reshape([7,1])
    robot.set_joint_torques(dict(list(zip(joint_names,total_torque))))


"""
    TESTING AREA (Functions needed to run an adaptive version of DeSchutter's impedance controller)
    [with geometrically consistent stiffness]
"""
def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

def from_three_to_six_dim(matrix):
    return np.block([[matrix,np.zeros((3,3))],[np.zeros((3,3)),matrix]])

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

def perform_torque_DeSchutter(M, B, K, x_d_ddot, x_d_dot,x_dot,delta_x, p_d, Rot_e,Rot_d,F_ext_2D,jacobian, robot_inertia, joint_names,sim): # must include Rot_d
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
    torque = np.linalg.multi_dot([jacobian.T,get_W(jacobian, robot_inertia, inv=True),alpha]).reshape((7,1)) + np.array(robot.coriolis_comp().reshape((7,1))) + np.dot(jacobian.T,F_ext_2D).reshape((7,1))
    robot.set_joint_torques(dict(list(zip(joint_names,torque))))
"""
    TESTING AREA
"""
# -------------- Plotting ------------------------

def plot_result(time_per_iteration,p,p_d, x_dot,delta_x, F_ext_raw,F_d_raw, z_dynamics,M,B,K,publish_rate):

    
    Fz_ext_raw = F_ext_raw[2]
    Fz_ext = Fz_ext_raw -Fz_ext_raw[0]#remove offset
    Fz_d_raw = F_d_raw[2] 
    Fz_d = Fz_d_raw - Fz_d_raw[0]#remove offset

    adjusted_time_per_iteration = time_per_iteration - time_per_iteration[0]
    new_list = np.zeros(len(p[0]))
    new_list[0]=adjusted_time_per_iteration[0]
    for i in range(len(adjusted_time_per_iteration)):
        if i >0:
            new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]

    plt.subplot(231)
    plt.title("External force")
    plt.plot(adjusted_time_per_iteration, Fz_ext, label="force z [N]")
    plt.plot(adjusted_time_per_iteration, Fz_d, label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(232)
    plt.title("Position")
    plt.plot(adjusted_time_per_iteration, p[0,:], label = "true x [m]")
    plt.plot(adjusted_time_per_iteration, p[1,:], label = "true y [m]")
    plt.plot(adjusted_time_per_iteration, p[2,:], label = "true z [m]")

    plt.plot(adjusted_time_per_iteration, p_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, p_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, p_d[2,:], label = "desired z [m]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(233)
    plt.title("Orientation error in Euler")
    plt.plot(adjusted_time_per_iteration, delta_x[3]*(180/np.pi), label = "error  Ori_x [degrees]")
    plt.plot(adjusted_time_per_iteration, delta_x[4]*(180/np.pi), label = "error  Ori_y [degrees]")
    plt.plot(adjusted_time_per_iteration, delta_x[5]*(180/np.pi), label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(234)
    plt.title("Adaptive dynamics along the z-axis")
    plt.plot(adjusted_time_per_iteration, z_dynamics[0], label = "inertia (M_z)")
    plt.plot(adjusted_time_per_iteration, z_dynamics[1], label = "damping (B_z)")
    plt.plot(adjusted_time_per_iteration, z_dynamics[2], label = "stiffness (K_z)")
    plt.axhline(y=M[2][2], label = "initial inertia (M_z)", color='b',linestyle='dashed')
    plt.axhline(y=B[2][2], label = "initial damping (B_z)", color='C1',linestyle='dashed')
    plt.axhline(y=K[2][2], label = "initial stiffness (K_z)", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.subplot(235)
    plt.title("velocity used in calculation")
    plt.plot(adjusted_time_per_iteration, x_dot[0], label = "vel x")
    plt.plot(adjusted_time_per_iteration,x_dot[1], label = "vel y")
    plt.plot(adjusted_time_per_iteration,x_dot[2], label = "vel z")
    plt.plot(adjusted_time_per_iteration,x_dot[3], label = "ang x")
    plt.plot(adjusted_time_per_iteration,x_dot[4], label = "ang y")
    plt.plot(adjusted_time_per_iteration,x_dot[5], label = "ang z")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(236)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.axhline(y=1/float(publish_rate), label = 'desired time-step', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
    """
    plt.subplot(235)
    plt.title("velocity read from rostopic")
    plt.plot(adjusted_time_per_iteration, v[0], label = "vel x")
    plt.plot(adjusted_time_per_iteration, v[1], label = "vel y")
    plt.plot(adjusted_time_per_iteration, v[2], label = "vel z")
    plt.plot(adjusted_time_per_iteration, v[3], label = "ang x")
    plt.plot(adjusted_time_per_iteration, v[4], label = "ang y")
    plt.plot(adjusted_time_per_iteration, v[5], label = "ang z")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(236)
    plt.title("numerically calculated velocity")
    plt.plot(adjusted_time_per_iteration, v_num[0], label = "vel x")
    plt.plot(adjusted_time_per_iteration, v_num[1], label = "vel y")
    plt.plot(adjusted_time_per_iteration, v_num[2], label = "vel z")
    plt.plot(adjusted_time_per_iteration, v_num[3], label = "ang x")
    plt.plot(adjusted_time_per_iteration, v_num[4], label = "ang y")
    plt.plot(adjusted_time_per_iteration, v_num[5], label = "ang z")
    plt.xlabel("Real time [s]")
    plt.legend()
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

    plt.show()


# move to neutral or alternative starting position (Dependent on sim/not sim)
def move_to_start(alternative_position, sim):
    if sim:
        robot.move_to_neutral()
    else:
        robot.move_to_joint_positions(alternative_position)


def fetch_states(i,time_per_iteration, p_d, x_history, x_dot_history, sim):
    
    
    x = get_x(goal_ori)
    x_history[:,i] = x
    p = x[:3]
    jacobian = robot.zero_jacobian()
    robot_inertia = robot.joint_inertia_matrix()
    Fz = get_Fz(sim)
    F_ext = construct_F_ext(Fz)
    F_ext_2D = F_ext.reshape([6,1])
    if sim:
        x_dot = get_x_dot(x_history,i,time_per_iteration, numerically=True)
    else:
        x_dot = get_x_dot(x_history,i,time_per_iteration, numerically=False)
    x_dot_history[:,i] = x_dot
    delta_x = get_delta_x(x, p_d)
    Rot_e = robot.endpoint_pose()['orientation_R']
        

    return Rot_e, p,x,x_dot, x_history, x_dot_history, delta_x, jacobian,robot_inertia,Fz,F_ext,F_ext_2D

if __name__ == "__main__":

    # ---------- Initialization -------------------
    sim = True
    rospy.init_node("impedance_control")
    robot = ArmInterface() #PandaArm()
    joint_names = robot.joint_names()
    publish_rate = 50
    rate = rospy.Rate(publish_rate)
    T = 0.001*(1000/publish_rate)
    max_num_it = duration*publish_rate
    move_to_start(cartboard,sim)


    # List used to contain data needed for calculation of the torque output 
    lam = np.zeros(18)
    #v_history = np.zeros((6,max_num_it))
    x_dot_history = np.zeros((6,max_num_it))

    # Lists providing data for plotting
    p_history = np.zeros((3,max_num_it))
    v_history_num = np.zeros((6,max_num_it))
    x_history = np.zeros((6,max_num_it))
    delta_x_history = np.zeros((6,max_num_it))
    F_ext_history = np.zeros((6,max_num_it))
    z_dynamics_history = np.zeros((3,max_num_it))

    time_per_iteration = np.zeros(max_num_it)

    # Specify the desired behaviour of the robot
    x_d_ddot, x_d_dot, p_d = generate_desired_trajectory_tc(max_num_it,T,move_in_x = True)
    goal_ori = np.asarray(robot.endpoint_pose()['orientation']) # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
    Rot_d = robot.endpoint_pose()['orientation_R'] # used by the DeSchutter implementation
    F_d = generate_Fd_steep(max_num_it,T,sim)

    
    # ----------- The control loop  -----------   
    for i in range(max_num_it):
        # update state-lists
        time_per_iteration[i]=rospy.get_time()
        Rot_e,p, x, x_dot,x_history,x_dot_history, delta_x, jacobian, robot_inertia, Fz, F_ext, F_ext_2D = fetch_states(i,time_per_iteration, p_d[:,i],x_history,x_dot_history,sim)

 

        #v_history_num[:,i] = get_x_dot(x_history,i,T, numerically=True) # only for plotting 
        #v_history[:,i] = get_x_dot(x_history,i,T) # for calculating error in acceleration


        # adapt M,B and K
        xi = get_xi(x_dot, x_d_dot[:,i], x_d_ddot[:,i], delta_x, x_dot_history, i, time_per_iteration)        
        lam = lam.reshape([18,1]) + get_lambda_dot(gamma,xi,K_v,P,F_d[:,i],F_ext_2D, i,time_per_iteration,T).reshape([18,1])
        M_hat,B_hat,K_hat = update_MBK_hat(lam,M,B,K)



        # Apply the resulting torque to the robot
        """CHOOSE ONE OF THE TWO CONTROLLERS BELOW"""
        perform_torque_Huang1992(M_hat, B_hat, K_hat, x_d_ddot[:,i], x_d_dot[:,i],x, x_dot, p_d[:,i],F_ext_2D, jacobian, robot_inertia,joint_names)
        #perform_torque_DeSchutter(M_hat, B_hat, K_hat, x_d_ddot[:,i], x_d_dot[:,i],x_dot, delta_x,p_d[:,i], Rot_e, Rot_d,F_ext_2D, jacobian, robot_inertia,joint_names, sim)
        


        # plotting and printing
        z_dynamics_history[0][i]=M_hat[2][2]
        z_dynamics_history[1][i]=B_hat[2][2]
        z_dynamics_history[2][i]=K_hat[2][2]

        p_history[:,i] = p 
        x_history[:,i] = x 
        delta_x_history[:,i] = delta_x
        F_ext_history[:,i] = F_ext


        # Live printing to screen when the controller is running
        if i%100 == 0:
            print(i,'/',max_num_it,' = ',T*i,' [s]   ) Force in z: ',F_ext_history[2,i])
            print(K_hat[2][2])
            print('')
        
        rate.sleep()

    #Uncomment the block below to save plotting-data 
    """
    np.save('VIC_p_d.npy',p_d)
    np.save('VIC_p.npy',p_history)
    np.save('VIC_Fz_d.npy',F_d)
    np.save('VIC_Fz.npy',F_ext_history[2])
    np.save('VIC_delta_x.npy',delta_x_history) #orientation error in radians
    np.save('VIC_adaptive_gains.npy',z_dynamics_history)
    """

    plot_result(time_per_iteration, p_history, p_d, x_dot_history, delta_x_history, F_ext_history, F_d, z_dynamics_history,M,B,K,publish_rate)



