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

"""
# --------- Constants / Parameters -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles
#new_start = {'panda_joint1': 1.938963389436404, 'panda_joint2': 0.6757504724282993, 'panda_joint3': -0.43399745125475564, 'panda_joint4': -2.0375275954865573, 'panda_joint5': -0.05233040021194351, 'panda_joint6': 3.133254153457202, 'panda_joint7': 1.283328743909796}


# this array is specifying the force-control-subspace (only doing force control in z)
S_f = np.array([[0, 0, 1, 0, 0, 0]]).reshape([6,1])

# This array is specifying the motion-control-subspace (x, y, ori_x, ori_y, ori_z)
S_v = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]]).reshape([6,5])

# Stiffness of the interaction [should be estimated (this one is chosen at random)]
K = np.array([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 100, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 1]]).reshape([6,6])
C = np.linalg.inv(K)


#max_num_it=500 # Duration of the run 
# Full run = 7500 iterations 

K_Plambda = 45 #force gains
K_Dlambda = K_Plambda*0.07 #K_Plambda*0.633 #random

#Position control dynamics:
Pp = 120 #proportional gain for position (x and y)
Dp = Pp*0.1*0.5*0.5 #damping position (x and y)

#Orientation control dynamics
Po =120 #80#40 #proportional gain for orientation
Do = 40#20 #damping_orientation

K_Pr = np.array([[Pp, 0, 0, 0, 0], # Stiffness matrix
                [0, Pp, 0, 0, 0],
                [0, 0, Po, 0, 0],
                [0, 0, 0, Po, 0],
                [0, 0, 0, 0, Po]])

K_Dr = np.array([[Dp, 0, 0, 0, 0], # Damping matrix
                [0, Dp, 0, 0, 0],
                [0, 0, Do, 0, 0],
                [0, 0, 0, Do, 0],
                [0, 0, 0, 0, Do]])

duration = 15 #seconds


"""Functions for generating desired MOTION trajectories"""

#1 Generate some desired trajectory (position and orientation)
def generate_desired_trajectory(max_num_it,T):
    a = np.zeros((5,max_num_it))
    v = np.zeros((5,max_num_it))
    s = np.zeros((2,max_num_it))
    
    s[:,0]= get_p()
    if max_num_it>6500:
        a[0,4500:4510]=0.00001/T**2
        a[0,6490:6500]=-0.00001/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:2,i-1]*T
    return a,v,s

#2 Generate some (time-consistent) desired trajectory (position and orientation)
def generate_desired_trajectory_tc(max_num_it,T,move_in_x=False):
    a = np.zeros((5,max_num_it))
    v = np.zeros((5,max_num_it))
    s = np.zeros((2,max_num_it))
    
    s[:,0]= get_p()

    if move_in_x:
        a[0,int(max_num_it*3/5):int(max_num_it*451/750)]=1.25
        a[0,int(max_num_it*649/750):int(max_num_it*13/15)]=-1.25

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:2,i-1]*T
    return a,v,s


"""Functions for generating desired FORCE trajectories"""


#1 Generate some desired force trajectory
def generate_F_d(max_num_it,T): 
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    a[0:100] = 0.0005/T**2
    a[100:200] = - 0.0005/T**2
    if max_num_it > 1100:
        a[500:550] = 0.0002/T**2
    if max_num_it >4001:
        a[1500:1550]=-0.0002/T**2
        it = 2000
        while it <= 4000:
            a[it]= (-9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1

        a[4001]=0.0001/T**2

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s

#2 Generate a (time-consistent) desired force trajectory 
def generate_F_d_tc(max_num_it,T): 
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    
    a[0:int(max_num_it/75)] = 62.5
    a[int(max_num_it/37.5):int(max_num_it/25)] = - 62.5
    if max_num_it > 275:
        a[int(max_num_it/15):int(max_num_it*11/150)] = 50
    if max_num_it >2001:
        a[int(max_num_it/5):int(max_num_it*31/150)]=-50
        it = int(max_num_it*4/15)
        while it <= int(max_num_it*8/15):
            a[it]= (-9*(np.pi**2)*(T/4)**2*np.sin(2*it*T/4*2*np.pi+np.pi/2))/T**2
            it+=1
        a[int(max_num_it*8/15+1)]=6.25
    
    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

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


# Fetch joint velocities
def get_joint_velocities():
    return np.array([robot.joint_velocity(robot.joint_names()[0]),robot.joint_velocity(robot.joint_names()[1]),robot.joint_velocity(robot.joint_names()[2]),robot.joint_velocity(robot.joint_names()[3]),robot.joint_velocity(robot.joint_names()[4]),robot.joint_velocity(robot.joint_names()[5]),robot.joint_velocity(robot.joint_names()[6])])


# Fetch linear and angular velocities (subject to motion control)
#def get_v():
 #   return (np.array([robot.endpoint_velocity()['linear'][0],robot.endpoint_velocity()['linear'][1],robot.endpoint_velocity()['angular'][0],robot.endpoint_velocity()['angular'][1],robot.endpoint_velocity()['angular'][2]])).reshape([5,1])


# Return the linear and angular velocities
# Numerically = True -> return the derivarive of the state-vector
# Numerically = False -> read values from rostopic (faulty in sim when interacting with the environment)
def get_v(x_hist,i,T, numerically=False, two_dim=True):
    if numerically == True:
        if two_dim == True:
            return get_derivative_of_vector(x_hist,i,T).reshape([5,1])
        else:
            return get_derivative_of_vector(x_hist,i,T)

    else:
        if two_dim == True:
            return np.array([np.append(get_cartesian_v()[:2],robot.endpoint_velocity()['angular'])]).reshape([5,1])
        else:
            return np.append(get_cartesian_v()[:2],robot.endpoint_velocity()['angular'])

# Return the position and (relative) orientation 
def get_x(goal_ori):
    pos_x = robot.endpoint_pose()['position'][:2]
    rel_ori = quatdiff_in_euler_radians(goal_ori, np.asarray(robot.endpoint_pose()['orientation']))
    return np.append(pos_x,rel_ori)


# Fetch the linear (cartesian) velocities
def get_cartesian_v():
    return np.array([robot.endpoint_velocity()['linear'][0],robot.endpoint_velocity()['linear'][1],robot.endpoint_velocity()['linear'][2]])


# Fetch the joint angles
def get_joint_angles():
    return np.array([robot.joint_angle(robot.joint_names()[0]),robot.joint_angle(robot.joint_names()[1]),robot.joint_angle(robot.joint_names()[2]),robot.joint_angle(robot.joint_names()[3]),robot.joint_angle(robot.joint_names()[4]),robot.joint_angle(robot.joint_names()[5]),robot.joint_angle(robot.joint_names()[6])])


# Fetch the position (in the subspace subject to motion control)
def get_p():
    return np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1]])


# Fetch the estimated external force in z 
def get_lambda():
    return robot.endpoint_effort()['force'][2]
    #return 0 #fake feedback 


# Fetch the estimated external forces and torques (h_e / F_ext)
def get_h_e():
    return np.array([0,0,robot.endpoint_effort()['force'][2],0,0,0])


# Fetch a simplified, less noisy estimate of the derivative of the external force in z 
def get_lambda_dot(S_f_inv,h_e_hist,i,T):
    h_e_dot = get_derivative_of_vector(h_e_hist,i,T)/30#40
    cap = 20#50
    if abs(h_e_dot[2]) > cap:
        h_e_dot[2] = np.sign(h_e_dot[2])*cap
    return np.dot(S_f_inv,h_e_dot)


# Fetch the psudoinverse of S_f/S_v as in equation (9.34) in chapter 9.3 of The Handbook of Robotics
def get_S_inv(S,C):
    a = np.linalg.inv(np.linalg.multi_dot([S.T,C,S]))
    return np.array(np.linalg.multi_dot([a,S.T,C]))


# Fetch K' as in equation (9.49) in chapter 9.3 of The Handbook of Robotics
def get_K_dot(S_f,S_f_inv,C):
    return np.array(np.linalg.multi_dot([S_f,S_f_inv,np.linalg.inv(C)])).reshape([6,6])


# Calculate the numerical derivative of a each row in a vector
def get_derivative_of_vector(history,iteration,T):
    size = history.shape[0]
    if iteration > 0:
        #return ((history[:,iteration]-history[:,iteration-1]).reshape([size,1])/T).reshape([size,1])
        return np.subtract(history[:,iteration],history[:,iteration-1])/T
    else:
        return np.zeros(size)#.reshape([size,1])


# Calculate the error in position and orientation (in the subspace subject to motion control)
def get_delta_r(goal_ori, p_d, two_dim = True):
    delta_pos = p_d - robot.endpoint_pose()['position'][:2]
    delta_ori = quatdiff_in_euler_radians(np.asarray(robot.endpoint_pose()['orientation']), goal_ori)    
    if two_dim == True:
        return np.array([np.append(delta_pos,delta_ori)]).reshape([5,1])

    else:
        return np.append(delta_pos,delta_ori)


# ------------  Calculation of torque -----------------

# Calculate f_lambda (part of equation 9.62) as in equation (9.65) in chapter 9.3 of The Handbook of Robotics
def calculate_f_lambda(f_d_ddot, f_d_dot, f_d, i,T, S_f,C,K_Dlambda,K_Plambda, z_force,h_e_hist):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    #lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
    lambda_dot = get_lambda_dot(S_f_inv,h_e_hist,i,T)
    lambda_a = f_d_ddot
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    return lambda_a + lambda_b + lambda_c

# Get the subproducts of f_lambda (for plotting/troubleshooting)
def get_f_lambda_subproducts(f_d_ddot, f_d_dot, f_d, i,T, S_f,C,K_Dlambda,K_Plambda, z_force,h_e_hist):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    #lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
    lambda_dot = get_lambda_dot(S_f_inv,h_e_hist,i,T)
    lambda_a = f_d_ddot
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    return lambda_a, lambda_b, lambda_c, (lambda_a + lambda_b + lambda_c)

# Calculate alpha_v (part of equation 9.62) as on page 213 in chapter 9.3 of The Handbook of Robotics
def calculate_alpha_v(i,T, goal_ori, r_d_ddot, r_d_dot, p_d,K_Pr,K_Dr,v):
    return (r_d_ddot.reshape([5,1]) + np.array(np.dot(K_Dr,r_d_dot.reshape([5,1])-v)).reshape([5,1])+ np.array(np.dot(K_Pr,get_delta_r(goal_ori,p_d))).reshape([5,1]))

# Calculate alpha (part of equation 9.16) as in equation (9.62) in chapter 9.3 of The Handbook of Robotics
def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_inv(S_v,C)
    P_v = np.array(np.dot(S_v,S_v_inv))
    C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
    return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + f_lambda*np.array(np.dot(C_dot,S_f)).reshape([6,1])

# Calculate and perform the torque as in equation (9.16) in chapter 9.2 of The Handbook of Robotics
def perform_torque(alpha):
    cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([robot.jacobian(),np.linalg.inv(robot.joint_inertia_matrix()),robot.jacobian().T]))
    alpha_torque = np.array(np.linalg.multi_dot([robot.jacobian().T,cartesian_inertia,alpha])).reshape([7,1])
    #external_torque = np.dot(robot.jacobian().T,np.append(robot.endpoint_effort()['force'],robot.endpoint_effort()['torque'])).reshape([7,1])
    external_torque = np.dot(robot.jacobian().T,np.array([0,0,robot.endpoint_effort()['force'][2],0,0,0])).reshape([7,1])
    torque = alpha_torque + robot.coriolis_comp().reshape([7,1]) - external_torque
    robot.set_joint_torques(dict(list(zip(robot.joint_names(),torque))))

# Plot the result of the run 

def plot_result(fz, fz_d ,p, p_d, ori_error, f_lambda,T, lambda_dot,f_d_dot, v_rostopic, v_num):

    time_array = np.arange(len(p[0]))*T
    

    plt.subplot(231)
    plt.title("External force")
    plt.plot(time_array, fz[:], label="force z [N]")
    plt.plot(time_array, fz_d[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(232)
    plt.title("Position")
    plt.plot(time_array, p[0,:], label = "true x [m]")
    plt.plot(time_array, p[1,:], label = "true y [m]")
    plt.plot(time_array, p[2,:], label = "true z [m]")
    plt.plot(time_array, p_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(time_array, p_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(233)
    plt.title("Orientation error")
    plt.plot(time_array, ori_error[0,:], label = "error  Ori_x [degrees]")
    plt.plot(time_array, ori_error[1,:], label = "error  Ori_y [degrees]")
    plt.plot(time_array, ori_error[2,:], label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(234)
    plt.title("Estimated force-derivative (lambda_dot)")
    plt.plot(time_array, lambda_dot, label = "lambda_dot")
    plt.plot(time_array, f_d_dot, label = "lambda_dot_desired",linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.subplot(235)
    plt.title("Force control")
    plt.plot(time_array, f_lambda[0][:], label="f_lambda (f_ddot-related)")
    plt.plot(time_array, f_lambda[1][:], label="f_lambda (f_dot-related)")
    plt.plot(time_array, f_lambda[2][:], label="f_lambda (f-related)")
    plt.plot(time_array, f_lambda[3][:], label="f_lambda (sum)")
    plt.xlabel("Real time [s]")
    plt.legend()

    

    plt.subplot(236)
    plt.title("velocity from ROSTOPIC vs numerically derived")
    plt.plot(time_array, v_rostopic[1], label='velocity in x [ROSTOPIC]',linestyle='dashed', color='C3')
    plt.plot(time_array, v_num[1], label='velocity in x [numeric]',color='C3')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.show()



# -------------- Running the controller ---------------------

if __name__ == "__main__":

    # ---------- Initialization -------------------
    rospy.init_node("impedance_control")
    robot = PandaArm()
    publish_rate = 250
    rate = rospy.Rate(publish_rate)
    T = 0.001*(1000/publish_rate)
    max_num_it = int(duration/T)
    #robot.move_to_joint_positions(new_start)
    robot.move_to_neutral() 

    # List used to contain data needed for calculation of the torque output 
    h_e_hist = np.zeros((6,max_num_it))
    x_history = np.zeros((5,max_num_it))


    # Lists providing data for plotting
    ori_error = np.zeros((3,max_num_it))
    z_force_history = np.zeros(max_num_it)
    f_lambda_history = np.zeros((4,max_num_it))
    lambda_dot_history = np.zeros(max_num_it)
    trajectory = np.zeros((3,max_num_it))
    #joint_vel_list = np.zeros((7,max_num_it))
    #joint_vel_hist_II = np.zeros((7,max_num_it))
    #joint_angle_list = np.zeros((7,max_num_it))
    v_rostopic = np.zeros((5,max_num_it))
    v_num = np.zeros((5,max_num_it))

    # Specify the desired behaviour of the robot
    r_d_ddot, r_d_dot, p_d = generate_desired_trajectory_tc(max_num_it,T, move_in_x=True)
    f_d_ddot,f_d_dot, f_d = generate_F_d_tc(max_num_it,T)
    goal_ori = np.asarray(robot.endpoint_pose()['orientation']) # goal orientation = current (initial) orientation [remains the same the entire duration of the run]


    # ----------- The control loop  -----------   
    for i in range(max_num_it):
        
        # Fetching necessary data 
        z_force = get_lambda()
        h_e_hist[:,i] = get_h_e()
        x_history[:,i] = get_x(goal_ori)
        v = get_v(x_history,i,T, numerically=True)

        # Calculating the parameters that together make up the outputted torque 
        f_lambda = calculate_f_lambda(f_d_ddot[i], f_d_dot[i], f_d[i], i, T, S_f ,C , K_Dlambda, K_Plambda, z_force, h_e_hist)
        alpha_v= calculate_alpha_v(i,T,goal_ori, r_d_ddot[:,i], r_d_dot[:,i], p_d[:,i], K_Pr,K_Dr,v)
        alpha = calculate_alpha(S_v,alpha_v,C,S_f,-f_lambda)

        # Apply the resulting torque to the robot 
        perform_torque(alpha)
        rate.sleep()


        # Live printing to screen when the controller is running
        if i%100 == 0:
            print(i,' /',max_num_it,'= ',T*i,' [s]   ) Force in z: ',robot.endpoint_effort()['force'][2])
            print('f_lambda: ',f_lambda)
            print('')

        # Collecting data for plotting
        trajectory[:,i] = np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],robot.endpoint_pose()['position'][2]])
        ori_error[:,i] = (180/np.pi)*quatdiff_in_euler_radians(np.asarray(robot.endpoint_pose()['orientation']), goal_ori)
        z_force_history[i] = z_force
        #joint_angle_list[:,i] = get_joint_angles()
        #joint_vel_list[:,i]= get_joint_velocities()
        #joint_vel_hist_II[:,i] = get_derivative_of_vector(joint_angle_list,i,T)
        v_rostopic[:,i] = get_v(x_history,i,T, numerically=False, two_dim=False)
        v_num[:,i] = get_v(x_history,i,T, numerically=True, two_dim=False)
        f_lambda_history[0][i],f_lambda_history[1][i],f_lambda_history[2][i], f_lambda_history[3][i] = get_f_lambda_subproducts(f_d_ddot[i], f_d_dot[i], f_d[i], i, T, S_f ,C , K_Dlambda, K_Plambda, z_force, h_e_hist)
        lambda_dot_history[i] = get_lambda_dot(get_S_inv(S_f,C),h_e_hist,i,T)


        
    #Uncomment the block below to save plotting-data 
    """
    np.save('HMFC_p_d.npy',p_d)
    np.save('HMFC_p.npy',trajectory)
    np.save('HMFC_Fz_d.npy',f_d)
    np.save('HMFC_Fz.npy',z_force_history)
    np.save('HMFC_ori_error.npy',ori_error) #orientation error in degrees
    """


    # Plotting the full result of the run 
    plot_result(z_force_history,f_d,trajectory, p_d, ori_error, f_lambda_history, T, lambda_dot_history, f_d_dot, v_rostopic, v_num)

