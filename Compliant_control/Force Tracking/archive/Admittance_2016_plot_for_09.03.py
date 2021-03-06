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

This is an ADMITTANCE CONTROLLER based on [Lahr2016: Understanding the implementation of Impedance Control in Industrial Robots]


It is computing a compliant position (x_c = x_d + E) based on the force error (F_d - F_ext) and a desired inertia, damping and stiffness (M,B,K).
The compliant position x_c is fed to a position controller.



About the code/controller:

1] The manipulator is doing quite jerky movements due to the noisiness of force measurements it is acting on 

3] The default desired motion- and force-trajectories are now made in a time-consistent matter, so that the PUBLISH RATE can be altered without messing up the desired behaviour. 
    The number of iterations is calculated as a function of the controller's control-cycle, T: (max_num_it = duration(=15 s) / T)

"""



# --------- Parameters -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles
#new_start = {'panda_joint1': 1.938963389436404, 'panda_joint2': 0.6757504724282993, 'panda_joint3': -0.43399745125475564, 'panda_joint4': -2.0375275954865573, 'panda_joint5': -0.05233040021194351, 'panda_joint6': 3.133254153457202, 'panda_joint7': 1.283328743909796}

"""Functions for generating desired MOTION trajectories"""

#1  Generate a desired motion-trajectory
def generate_desired_trajectory(iterations,T):
    a = np.zeros((3,iterations))
    v = np.zeros((3,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = robot.endpoint_pose()['position']
    
    if iterations > 300:
        a[2,0:50]=-0.00001/T**2
        a[2,50:100]=0.00001/T**2
        
    if iterations > 6500:
        a[0,4500:4510]=0.00001/T**2
        a[0,6490:6500]=-0.00001/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:,i-1]*T
    return p

#2  Generate a (time-consistent) desired motion-trajectory
def generate_desired_trajectory_tc(iterations,T,move_in_x=False): #admittance
    a = np.zeros((3,iterations))
    v = np.zeros((3,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = robot.endpoint_pose()['position']

    a[2,0:int(max_num_it/75)]=-0.625
    a[2,int(max_num_it/75):int(max_num_it*2/75)]=0.625
        
    if move_in_x:
        a[0,int(max_num_it*3/5):int(max_num_it*451/750)]=1.25
        a[0,int(max_num_it*649/750):int(max_num_it*13/15)]=-1.25
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:,i-1]*T
    return p

"""Functions for generating desired FORCE trajectories"""

#1  Generate a desired force-trajectory 
def generate_F_d(max_num_it,T):
    a = np.zeros((6,max_num_it))
    v = np.zeros((6,max_num_it))
    s = np.zeros((6,max_num_it))
    
    s[2,0] = 15

    
    for i in range(max_num_it):
        if i>0:
            v[2,i]=v[2,i-1]+a[2,i-1]*T
            s[2,i]=s[2,i-1]+v[2,i-1]*T

    return s

#2  Generate a (time-consistent) desired force trajectory 
def generate_F_d_tc(max_num_it,T):
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





# -------------- Main functions --------------------

# Update the list of the last three recorded force errors
def update_F_error_list(F_error_list,F_d): #setting forces in x and y = 0
    for i in range(3): #update for x, then y, then z
        F_error_list[i][2]=F_error_list[i][1]
        F_error_list[i][1]=F_error_list[i][0]
        if i ==2:
            F_error_list[i][0] = robot.endpoint_effort()['force'][i]-F_d[i]
        else:
            F_error_list[i][0] = 0


# Update the list of the last three E-calculations
def update_E_history(E_history, E):
    for i in range(3):
        E_history[i][1]=E_history[i][0]
        E_history[i][0] = E[i]

# Calculate E (as in 'step 8' of 'algorithm 2' in Lahr2016 [Understanding the implementation of Impedance Control in Industrial Robots] )
def calculate_E(T,E_history, F_e_history,M = 1*np.array([1, 1, 1]),B =20*np.array([1, 1, 1]),K= 5*np.array([1, 1, 1])):
    x_x = (T**(2) * F_e_history[0][0] + 2* T**(2) * F_e_history[0][1]+ T**(2) * F_e_history[0][2]-(2*K[0]*T**(2)-8*M[0])*E_history[0][0]-(4*M[0] -2*B[0]*T+K[0]*T**(2))*E_history[0][1])/(4*M[0]+2*B[0]*T+K[0]*T**(2))
    x_y = (T**2 * F_e_history[1][0] + 2* T**2 * F_e_history[1][1]+ T**2 * F_e_history[1][2]-(2*K[1]*T**2-8*M[1])*E_history[1][0]-(4*M[1] -2*B[1]*T+K[1]*T**2)*E_history[1][1])/(4*M[1]+2*B[1]*T+K[1]*T**2)
    x_z = (T**2 * F_e_history[2][0] + 2* T**2 * F_e_history[2][1]+ T**2 * F_e_history[2][2]-(2*K[2]*T**2-8*M[2])*E_history[2][0]-(4*M[2] -2*B[2]*T+K[2]*T**2)*E_history[2][1])/(4*M[2]+2*B[2]*T+K[2]*T**2)
    return np.array([x_x,x_y,x_z]) 

# Perform position control with the compliant position (x_c = x_d + E) as input
def perform_joint_position_control(x_d,E,ori):
    x_c = x_d + E
    joint_angles = robot.inverse_kinematics(x_c,ori=ori)[1]
    robot.exec_position_cmd(joint_angles)

def quatdiff_in_euler(quat_curr, quat_des):
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





# -------------- Plotting ------------------------

def plot_result(force,x_c,pos,F_d,x_d,ori_error,T):

    time_array = np.arange(len(F_d[0]))*T
    
    
    plt.subplot(121)
    plt.title("Sensed external wrench")
    plt.plot(time_array, force[2,:], label="force z [N]")
    plt.plot(time_array, F_d[2,:], label = " desired z-force [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.subplot(122)
    plt.title("position")
    #plt.plot(time_array, pos[0,:], label = "true x [m]")
    #plt.plot(time_array, pos[1,:], label = "true y [m]")
    plt.plot(time_array, pos[2,:], label = "true  z [m]")
    #plt.plot(time_array, x_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    #plt.plot(time_array, x_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.plot(time_array, x_d[2,:], label = "desired z [m]",linestyle='dashed')
    plt.plot(time_array, x_c[2,:], label = "compliant z [m]",linestyle='dotted')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.subplot(133)
    plt.title("Error in orientation")
    plt.plot(time_array, ori_error[0,:], label = "true  Ori_x [degrees]")
    plt.plot(time_array, ori_error[1,:], label = "true  Ori_y [degrees]")
    plt.plot(time_array, ori_error[2,:], label = "true  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    
    plt.show()



# -------------- Running the controller ---------------------

if __name__ == "__main__":

    # ---------- Initialization -------------------

    rospy.init_node("admittance_control")
    robot = PandaArm()
    
    publish_rate = 250
    rate = rospy.Rate(publish_rate)
    T = 0.001*(1000/publish_rate) # The control loop's time step
    #duration = 15
    #max_num_it = int(duration/T)
    #robot.move_to_joint_positions(new_start)
    robot.move_to_neutral() # Move the manipulator to its neutral position (starting position)

    max_num_it=500 # Duration of the run
    # Full run = 7500 iterations 

     
    # List used to contain data needed for calculation of the torque output 
    F_error_list = np.zeros((3,3))
    E = np.zeros(3)
    E_history = np.zeros((3,3))
    
    # Lists providing data for plotting
    x_history = np.zeros((3,max_num_it))
    x_c_history = np.zeros((3,max_num_it))
    F_ext_history = np.zeros((6,max_num_it))
    orientation_error_history = np.zeros((3,max_num_it))
    #desired_ori_degrees = get_ori_degrees()

    
    
    # Specify the desired behaviour of the robot
    goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
    x_d = generate_desired_trajectory(max_num_it,T)
    F_d = generate_F_d(max_num_it,T)


    # ----------- The control loop  -----------                       
    for i in range(max_num_it):
        
        
        # Update the compliant position every X'th iteration 
        if i%2==0: 
            update_F_error_list(F_error_list,F_d[:,i])
            E = calculate_E(T,E_history, F_error_list)
            update_E_history(E_history,E)
            
        """chose one of the two position controllers: """
        #perform_joint_position_control(x_d[:,i],E,goal_ori)
        PD_torque_control(x_d[:,i],E,goal_ori)
        
        rate.sleep()
        
        
        # Live printing to screen when the controller is running
        if i%100==0: 
            print(i,', pos:',robot.endpoint_pose()['position'],' F: ', robot.endpoint_effort()['force'][2])#' force measured: ',robot.endpoint_effort()['force'])

        # Collecting data for plotting
        F_ext_history[:,i]=np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])
        x_c_history[:,i] = x_d[:,i] + E
        x_history[:,i] = robot.endpoint_pose()['position']
        orientation_error_history[:,i] = quatdiff_in_euler_degrees(robot.endpoint_pose()['orientation'], goal_ori)

    
    #Uncomment the block below to save plotting-data 
    """
    np.save('Admittance_x_d.npy',x_d)
    np.save('Admittance_x_c.npy',x_c_history)
    np.save('Admittance_x.npy',x_history)
    np.save('Admittance_Fz_d.npy',F_d)
    np.save('Admittance_Fz.npy',F_ext_history[2])
    np.save('Admittance_ori_error.npy',orientation_error_history) #orientation error in degrees
    """


    # Plotting the full result of the run         
    plot_result(F_ext_history,x_c_history,x_history,F_d,x_d,orientation_error_history,T)

