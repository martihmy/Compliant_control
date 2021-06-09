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

This is an ADMITTANCE CONTROLLER based on [Lahr2016: Understanding the implementation of Impedance Control in Industrial Robots]


It is computing a compliant position (x_c = x_d + E) based on the force error (F_d - F_ext) and a desired inertia, damping and stiffness (M,B,K).
The compliant position x_c is fed to a position controller.



About the code/controller:

1] The manipulator is doing some jerky movements due to the noisiness of force measurements it is acting on 

2] The default desired motion- and force-trajectories are now made in a time-consistent matter, so that the PUBLISH RATE can be altered without messing up the desired behaviour. 
    The number of iterations is calculated as a function of the controller's control-cycle, T: (max_num_it = duration(=15 s) / T)

3] IN THE FIRST LINE OF CODE BELOW "if __name__ == "__main__":" YOU HAVE TO SET SIM EQUAL TO "True" OR "False"
            - if True: starting position = neutral
            - if False: starting position = cartboard, Fz = (-) robot.endpoint_effort...

4] The time step (T) is now being explicitly calculated for each iteration due to its stochastic nature"""



# --------- Parameters -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles
cartboard = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

"""Functions for generating desired MOTION trajectories"""

#1  Generate a desired motion-trajectory
def generate_desired_trajectory(iterations,T):
    a = np.zeros((3,iterations))
    v = np.zeros((3,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = robot.endpoint_pose()['position']
    
    if iterations > 300:
        a[2,0:100]=-0.00001/T**2
        a[2,100:200]=0.00001/T**2
        
    if iterations > 6500:
        a[0,4500:4510]=0.00001/T**2
        a[0,6490:6500]=-0.00001/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:,i-1]*T
    return p

#2  Generate a (time-consistent) desired motion-trajectory
def generate_desired_trajectory_tc(robot,iterations,T,sim,move_in_x=False, move_down=False): #admittance
    a = np.zeros((3,iterations+100))
    v = np.zeros((3,iterations+100))
    p = np.zeros((3,iterations+100))
    p[:,0] = robot.endpoint_pose()['position']
    max_num_it = iterations
    if move_down:
        a[2,0:int(max_num_it/75)]=-0.625
        a[2,int(max_num_it/75):int(max_num_it*2/75)]=0.625
        
    if move_in_x:
	if sim:
	    a[0,int(max_num_it*4/10):int(max_num_it*6/10)]=0.05
            a[0,int(max_num_it*6/10):int(max_num_it*8/10)]=-0.05
	else:
	    a[0,int(max_num_it*4/10):int(max_num_it*6/10)]=0.0125
            a[0,int(max_num_it*6/10):int(max_num_it*8/10)]=-0.0125



    for i in range(max_num_it+100):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:,i-1]*T
    return p



"""Functions for generating desired FORCE trajectories"""

#1  Generate a SMOOTH desired force-trajectory [STABLE]
def generate_Fd_smooth(robot,max_num_it,T,sim=False):
    a = np.zeros((6,max_num_it+100))
    v = np.zeros((6,max_num_it+100)) #+100 to add a buffer when updating the state-space
    s = np.zeros((6,max_num_it+100)) 
    s[2,0]= robot.get_Fz(sim)
    a[2,0:max_num_it/15] = 5
    a[2,max_num_it/15:2*max_num_it/15] = - 5

    for i in range(max_num_it+100):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s


#2  Generate an ADVANCED (time-consistent) desired force trajectory 
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

# Generate a constant desired force [STABLE]
def generate_F_d_constant(max_num_it,Fd):
    s = np.zeros((6,max_num_it))
    s[2,:]= Fd
    return s

def generate_F_d_steep(robot, max_num_it,T,sim):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    #s[2,0]= robot.get_Fz(sim)
    v[2,0] = 0.05
    for i in range(max_num_it):
        if i>0:
            #v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=min(s[2,i-1]+v[2,i-1]*T,3)
    return s

def generate_F_d_steep(max_num_it,T,f_d):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    #s[2,0]= robot.get_Fz(sim)
    v[2,0] = 10
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]#+a[2,i-1]*T
            s[2,i]=min(s[2,i-1]+v[2,i-1]*T,f_d)
    return s

# ------------ Helper functions --------------------------------



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

# Low-pass filter
def real_time_filter(value,z,b):
    filtered_value, z = signal.lfilter(b, 1, [value], zi=z)
    return filtered_value,z


# Update the list of the last three recorded force errors
def update_F_error_list(F_error_list,F_d,Fz,sim): #setting forces in x and y = 0
    for i in range(3): #update for x, then y, then z
        F_error_list[i][2]=F_error_list[i][1]
        F_error_list[i][1]=F_error_list[i][0]
        if i ==2:
            F_error_list[i][0] = Fz-F_d[i]
        else:
            F_error_list[i][0] = 0


# Update the list of the last three E-calculations
def update_E_history(E_history, E):
    for i in range(3):
        E_history[i][1]=E_history[i][0]
        E_history[i][0] = E[i]
    return E_history

# Calculate E (as in 'step 8' of 'algorithm 2' in Lahr2016 [Understanding the implementation of Impedance Control in Industrial Robots] )
def calculate_E(i,time_per_iteration,E_history, F_e_history,M,B,K):
    if i < 1:
        return np.array([0,0,0])
    T = time_per_iteration[i]-time_per_iteration[i-1]
    x_z = (T**2 * F_e_history[2][0] + 2* T**2 * F_e_history[2][1]+ T**2 * F_e_history[2][2]-(2*K*T**2-8*M)*E_history[2][0]-(4*M -2*B*T+K*T**2)*E_history[2][1])/(4*M+2*B*T+K*T**2)
    return np.array([0,0,x_z]) 

# Perform position control with the compliant position (x_c = x_d + E) as input
def perform_joint_position_control(robot,x_d,E,ori):
    x_c = x_d + E
    desired_joint_angles = robot.inverse_kinematics(x_c,ori=ori)[1]
    #joint_angles = robot.joint_ordered_angles()

    robot.exec_position_cmd(desired_joint_angles)










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


