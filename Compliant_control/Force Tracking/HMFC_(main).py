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

4] IN THE FIRST LINE OF CODE BELOW "if __name__ == "__main__":" YOU HAVE TO SET SIM EQUAL TO "True" OR "False"
            - if True: starting position = neutral
            - if False: starting position = cartboard, Fz = (-) robot.endpoint_effort...

5] The time step (T) is now being explicitly calculated for each iteration due to its stochastic nature

"""
# --------- Constants / Parameters -----------------------------

#print(robot.joint_ordered_angles()) #Read the robot's joint-angles

cartboard = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

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
K_Dlambda = K_Plambda*0.001 #K_Plambda*0.633 #random

#Position control dynamics:
Pp = 60 #proportional gain for position (x and y)
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
        a[0,int(max_num_it*4/10):int(max_num_it*5/10)]=0.015
        a[0,int(max_num_it*7/10):int(max_num_it*8/10)]=-0.015

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:2,i-1]*T
    return a,v,s


"""Functions for generating desired FORCE trajectories"""

#1  Generate a SMOOTH desired force trajectory that takes offset into consideration [STABLE]
def generate_Fd_smooth(max_num_it,T,sim=False):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    s[0]= get_lambda(sim)
    a[0:max_num_it/15] = 5
    a[max_num_it/15:2*max_num_it/15] = - 5

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s



#3 Generate an advanced (time-consistent) desired force trajectory 
def generate_Fd_advanced(max_num_it,T): 
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

# generate a STEEP desired force trajectory [STABLE]
def generate_Fd_steep(max_num_it,T,sim=False):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    s[0]= get_lambda(sim)
    v[0]=2.5
    a[max_num_it/15:3*max_num_it/15] = - 1.25

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s

# Generate a constant desired force [STABLE]
def generate_Fd_constant(max_num_it,T,sim=False):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    s[0]= get_lambda(sim)+3

    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]*T
            s[i]=s[i-1]+v[i-1]*T

    return a,v,s


# Generate a HMFC-friendly force trajecotry [STABLE]    
def generate_Fd_jump_and_steep(max_num_it,T,sim=False):
    a = np.zeros(max_num_it)
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    s[0]= get_lambda(sim)+2.5
   #a[0:10] = 0.005/T**2
    v[0]=2
    a[max_num_it/15:2*max_num_it/15] = -2

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
def get_joint_velocities(joint_names):
    dict = robot.joint_velocities()
    return np.array([ dict[joint_names[0]],dict[joint_names[1]],dict[joint_names[2]],dict[joint_names[3]],dict[joint_names[4]],dict[joint_names[5]],dict[joint_names[6]]])
    #return np.array([robot.joint_velocity(robot.joint_names()[0]),robot.joint_velocity(robot.joint_names()[1]),robot.joint_velocity(robot.joint_names()[2]),robot.joint_velocity(robot.joint_names()[3]),robot.joint_velocity(robot.joint_names()[4]),robot.joint_velocity(robot.joint_names()[5]),robot.joint_velocity(robot.joint_names()[6])])


# Fetch linear and angular velocities (subject to motion control)
#def get_v():
 #   return (np.array([robot.endpoint_velocity()['linear'][0],robot.endpoint_velocity()['linear'][1],robot.endpoint_velocity()['angular'][0],robot.endpoint_velocity()['angular'][1],robot.endpoint_velocity()['angular'][2]])).reshape([5,1])


# Return the linear and angular velocities
# Numerically = True -> return the derivarive of the state-vector
# Numerically = False -> read values from rostopic (faulty in sim when interacting with the environment)
def get_v(x_hist,i,time_per_iteration, numerically=False, two_dim=True):
    if numerically == True:
        if two_dim == True:
            return get_derivative_of_vector(x_hist,i,time_per_iteration).reshape([5,1])
        else:
            return get_derivative_of_vector(x_hist,i,time_per_iteration)

    else:
        if two_dim == True:
            return np.array([np.append(get_cartesian_v()[:2],robot.endpoint_velocity()['angular'])]).reshape([5,1])
        else:
            return np.append(get_cartesian_v()[:2],robot.endpoint_velocity()['angular'])

# Return the position and (relative) orientation 
def get_x(p,ori,goal_ori):
    pos_x = p[:2]
    rel_ori = quatdiff_in_euler_radians(goal_ori, np.asarray(ori))
    return np.append(pos_x,rel_ori)

# Fetch the linear (cartesian) velocities
def get_cartesian_v():
    return robot.endpoint_velocity()['linear']


# Fetch the joint angles
def get_joint_angles():
    return np.array([robot.joint_angle(robot.joint_names()[0]),robot.joint_angle(robot.joint_names()[1]),robot.joint_angle(robot.joint_names()[2]),robot.joint_angle(robot.joint_names()[3]),robot.joint_angle(robot.joint_names()[4]),robot.joint_angle(robot.joint_names()[5]),robot.joint_angle(robot.joint_names()[6])])


# Fetch the position (in the subspace subject to motion control) [only used for offline trajectory planning]
def get_p():
    return robot.endpoint_pose()['position'][0:2]


# Fetch the estimated external force in z 
def get_lambda(sim=False):
    if sim:
        return robot.endpoint_effort()['force'][2]
    else:
        return -robot.endpoint_effort()['force'][2]
    #return 0 #fake feedback 


# Fetch the estimated external forces and torques (h_e / F_ext)
def construct_h_e(Fz):
    return np.array([0,0,Fz,0,0,0])



# Fetch a simplified, less noisy estimate of the derivative of the external force in z 
def get_lambda_dot(S_f_inv,h_e_hist,i,time_per_iteration):
    h_e_dot = get_derivative_of_vector(h_e_hist,i,time_per_iteration)
    cap = 50
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
def get_derivative_of_vector(history,iteration,time_per_iteration):
    size = history.shape[0]
    if iteration > 0:
        T = float(time_per_iteration[iteration]-time_per_iteration[iteration-1])
        #return ((history[:,iteration]-history[:,iteration-1]).reshape([size,1])/T).reshape([size,1])
        if T>0:
            return np.subtract(history[:,iteration],history[:,iteration-1])/float(T)
    
    return np.zeros(size)#.reshape([size,1])


# Calculate the error in position and orientation (in the subspace subject to motion control)
def get_delta_r(ori,goal_ori, p, p_d, two_dim = True):
    delta_pos = p_d - p[:2]
    delta_ori = quatdiff_in_euler_radians(np.asarray(ori), goal_ori)    
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
    return max(lambda_a + lambda_b + lambda_c,0)

# Get the subproducts of f_lambda (for plotting/troubleshooting)
def get_f_lambda_subproducts(f_d_ddot, f_d_dot, f_d, i,time_per_iteration, S_f,C,K_Dlambda,K_Plambda, z_force,h_e_hist,jacobian,joint_names,sim):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    if sim: 
       lambda_dot = get_lambda_dot(S_f_inv,h_e_hist,i,time_per_iteration)
    else: 
        lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,jacobian,get_joint_velocities(joint_names)])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
    lambda_a = f_d_ddot
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    return lambda_dot, lambda_a, lambda_b, lambda_c, max(lambda_a + lambda_b + lambda_c,0)

    """def get_f_lambda_subproducts(f_d_ddot, f_d_dot, f_d, i,T, S_f,C,K_Dlambda,K_Plambda, z_force,h_e_hist):
    S_f_inv = get_S_inv(S_f,C)
    K_dot = get_K_dot(S_f,S_f_inv,C)
    #lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,robot.jacobian(),get_joint_velocities()])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
    lambda_dot = get_lambda_dot(S_f_inv,h_e_hist,i,T)
    lambda_a = f_d_ddot
    lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
    lambda_c = np.dot(K_Plambda,(f_d-z_force))
    return lambda_a, lambda_b, lambda_c, (max(lambda_a + lambda_b + lambda_c,0))
    """

# Calculate alpha_v (part of equation 9.62) as on page 213 in chapter 9.3 of The Handbook of Robotics
def calculate_alpha_v(i, ori, goal_ori, r_d_ddot, r_d_dot, p,p_d,K_Pr,K_Dr,v):
    return (r_d_ddot.reshape([5,1]) + np.array(np.dot(K_Dr,r_d_dot.reshape([5,1])-v)).reshape([5,1])+ np.array(np.dot(K_Pr,get_delta_r(ori,goal_ori,p,p_d))).reshape([5,1]))

# Calculate alpha (part of equation 9.16) as in equation (9.62) in chapter 9.3 of The Handbook of Robotics
def calculate_alpha(S_v, alpha_v,C,S_f,f_lambda):
    S_v_inv = get_S_inv(S_v,C)
    P_v = np.array(np.dot(S_v,S_v_inv))
    C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
    return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + f_lambda*np.array(np.dot(C_dot,S_f)).reshape([6,1])

# Calculate and perform the torque as in equation (9.16) in chapter 9.2 of The Handbook of Robotics
def perform_torque(alpha,sim,jacobian,h_e,joint_names):
    cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([jacobian,np.linalg.inv(robot.joint_inertia_matrix()),jacobian.T]))
    alpha_torque = np.array(np.linalg.multi_dot([jacobian.T,cartesian_inertia,alpha])).reshape([7,1])
    external_torque = np.dot(jacobian.T,h_e).reshape([7,1])
    torque = alpha_torque + robot.coriolis_comp().reshape([7,1]) - external_torque
    robot.set_joint_torques(dict(list(zip(joint_names,torque))))

# Plot the result of the run 

def plot_result(time_per_iteration, fz_raw, fz_d_raw ,p, p_d, ori_error, f_lambda,T, lambda_dot,f_d_dot,publish_rate):#, v_rostopic, v_num):

    fz = fz_raw - fz_raw[0] #remove offset
    fz_d = fz_d_raw - fz_raw[0] #remove offset


    adjusted_time_per_iteration = time_per_iteration - time_per_iteration[0]
    new_list = np.zeros(len(p[0]))
    new_list[0]=1/float(publish_rate) #adjusted_time_per_iteration[0]
    for i in range(len(adjusted_time_per_iteration)):
        if i >0:
            new_list[i] = adjusted_time_per_iteration[i]-adjusted_time_per_iteration[i-1]

    plt.subplot(231)
    plt.title("External force")
    plt.plot(adjusted_time_per_iteration, fz[:], label="force z [N]")
    plt.plot(adjusted_time_per_iteration, fz_d[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()


    plt.subplot(232)
    plt.title("Position")
    plt.plot(adjusted_time_per_iteration, p[0,:], label = "true x [m]")
    plt.plot(adjusted_time_per_iteration, p[1,:], label = "true y [m]")
    plt.plot(adjusted_time_per_iteration, p[2,:], label = "true z [m]")
    plt.plot(adjusted_time_per_iteration, p_d[0,:], label = "desired x [m]", color='b',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, p_d[1,:], label = "desired y [m]", color='C1',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    
    plt.subplot(233)
    plt.title("Orientation error")
    plt.plot(adjusted_time_per_iteration, ori_error[0,:], label = "error  Ori_x [degrees]")
    plt.plot(adjusted_time_per_iteration, ori_error[1,:], label = "error  Ori_y [degrees]")
    plt.plot(adjusted_time_per_iteration, ori_error[2,:], label = "error  Ori_z [degrees]")
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(234)
    plt.title("Estimated force-derivative (lambda_dot)")
    plt.plot(adjusted_time_per_iteration, lambda_dot, label = "lambda_dot")
    plt.plot(adjusted_time_per_iteration, f_d_dot, label = "lambda_dot_desired",linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    
    plt.subplot(235)
    plt.title("Force control")
    plt.plot(adjusted_time_per_iteration, f_lambda[0][:], label="f_lambda (f_ddot-related)")
    plt.plot(adjusted_time_per_iteration, f_lambda[1][:], label="f_lambda (f_dot-related)")
    plt.plot(adjusted_time_per_iteration, f_lambda[2][:], label="f_lambda (f-related)")
    plt.plot(adjusted_time_per_iteration, f_lambda[3][:], label="f_lambda (sum)")
    plt.xlabel("Real time [s]")
    plt.legend()

    
    plt.subplot(236)
    plt.title("Time per iteration")
    plt.plot(new_list, label = "time per iteration")
    plt.axhline(y=1/float(publish_rate), label = 'desired time-step', color='C1', linestyle = 'dashed')
    plt.xlabel("iterations")
    plt.legend()
    """
    plt.subplot(236)
    plt.title("velocity from ROSTOPIC vs numerically derived")
    plt.plot(adjusted_time_per_iteration, v_rostopic[1], label='velocity in x [ROSTOPIC]',linestyle='dashed')
    plt.plot(adjusted_time_per_iteration, v_num[1], label='velocity in x [numeric]')
    plt.xlabel("Real time [s]")
    plt.legend()
    """

    plt.show()

def fetch_states(goal_ori, x_history,h_e_hist,i,time_per_iteration, sim):
    
    z_force = get_lambda(sim)
    h_e = construct_h_e(z_force)#
    h_e_hist[:,i] = h_e
    ori = robot.endpoint_pose()['orientation']
    p = robot.endpoint_pose()['position']
    x = get_x(p,ori,goal_ori)
    x_history[:,i] = x
    jacobian = robot.zero_jacobian()
    if sim:
        v = get_v(x_history,i,time_per_iteration, numerically=True)
    else:
        v = get_v(x_history,i,time_per_iteration, numerically=False)
    return z_force, h_e,h_e_hist,p,ori,x,x_history,jacobian,v

# move to neutral or alternative starting position (Dependent on sim/not sim)
def move_to_start(alternative_position, sim):
    if sim:
        robot.move_to_neutral()
    else:
        robot.move_to_joint_positions(alternative_position)


# -------------- Running the controller ---------------------

if __name__ == "__main__":

    # ---------- Initialization -------------------
    sim = True
    rospy.init_node("impedance_control")
    robot = ArmInterface()
    joint_names=robot.joint_names()
    publish_rate = 40
    rate = rospy.Rate(publish_rate)
    T = 0.001*(1000/publish_rate)
    max_num_it = int(duration*publish_rate)
    move_to_start(cartboard,sim)


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

    time_per_iteration = np.zeros(max_num_it)

    # Specify the desired behaviour of the robot
    r_d_ddot, r_d_dot, p_d = generate_desired_trajectory_tc(max_num_it,T, move_in_x=True)
    f_d_ddot,f_d_dot, f_d = generate_Fd_constant(max_num_it,T,sim)#generate_Fd_jump_and_steep(max_num_it,T,sim)
    goal_ori = np.asarray(robot.endpoint_pose()['orientation']) # goal orientation = current (initial) orientation [remains the same the entire duration of the run]


    # ----------- The control loop  -----------   
    for i in range(max_num_it):
        
        # Fetching necessary data 
        time_per_iteration[i]=rospy.get_time()
        z_force, h_e,h_e_hist,p,ori,x,x_history,jacobian,v = fetch_states(goal_ori, x_history,h_e_hist,i,time_per_iteration, sim)
        
        """
        z_force = get_lambda(sim) #DOES NOT WORK
        h_e = get_h_e(sim)
        p,x = get_x(goal_ori)
        v = get_v(x_history,i,T, numerically=True)
        jacobian = robot.zero_jacobian()

        h_e_hist[:,i] = h_e
        x_history[:,i] = x
        """
        """
        z_force = get_lambda(sim) #DO WORK
        h_e = get_h_e(sim)
        h_e_hist[:,i] = h_e
        p,x_history[:,i] = get_x(goal_ori)
        v = get_v(x_history,i,T, numerically=True)
        jacobian = robot.zero_jacobian()
        """
        # Calculating the parameters that together make up the outputted torque 
        #f_lambda = calculate_f_lambda(f_d_ddot[i], f_d_dot[i], f_d[i], i, T, S_f ,C , K_Dlambda, K_Plambda, z_force, h_e_hist)
        
        lambda_dot,lambda_a, lambda_b, lambda_c, f_lambda = get_f_lambda_subproducts(f_d_ddot[i], f_d_dot[i], f_d[i], i, time_per_iteration, S_f ,C , K_Dlambda, K_Plambda, z_force, h_e_hist,jacobian,joint_names,sim)
        alpha_v= calculate_alpha_v(i,ori,goal_ori, r_d_ddot[:,i], r_d_dot[:,i],p, p_d[:,i], K_Pr,K_Dr,v)
        alpha = calculate_alpha(S_v,alpha_v,C,S_f,-f_lambda)


        
        # Apply the resulting torque to the robot 
        perform_torque(alpha,sim,jacobian,h_e,joint_names)
        


        # Live printing to screen when the controller is running
        if i%100 == 0:
            print(i,' /',max_num_it,'= ',T*i,' [s]   ) Force in z: ',z_force)
            print('f_lambda: ',f_lambda)
            print('')

        # Collecting data for plotting
        trajectory[:,i] = p
        ori_error[:,i] = (180/np.pi)*x[2:]
        z_force_history[i] = z_force
        #joint_angle_list[:,i] = get_joint_angles() #slow
        #joint_vel_list[:,i]= get_joint_velocities() # slow
        #joint_vel_hist_II[:,i] = get_derivative_of_vector(joint_angle_list,i,T)
        #v_rostopic[:,i] = get_v(x_history,i,T, numerically=False, two_dim=False)
        #v_num[:,i] = get_v(x_history,i,T, numerically=True, two_dim=False)
        f_lambda_history[0][i],f_lambda_history[1][i],f_lambda_history[2][i], f_lambda_history[3][i] =  lambda_a, lambda_b, lambda_c, f_lambda
        lambda_dot_history[i] = lambda_dot
        rate.sleep()


        
    #Uncomment the block below to save plotting-data 
    """
    np.save('HMFC_p_d.npy',p_d)
    np.save('HMFC_p.npy',trajectory)
    np.save('HMFC_Fz_d.npy',f_d)
    np.save('HMFC_Fz.npy',z_force_history)
    np.save('HMFC_ori_error.npy',ori_error) #orientation error in degrees
    """


    # Plotting the full result of the run 
    plot_result(time_per_iteration,z_force_history,f_d,trajectory, p_d, ori_error, f_lambda_history, T, lambda_dot_history, f_d_dot,publish_rate)#, v_rostopic, v_num)

