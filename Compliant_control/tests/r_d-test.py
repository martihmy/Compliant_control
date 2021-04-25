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

def get_p():
    return robot.endpoint_pose()['position'][0:2]

def get_r():
    quat_as_list = np.array([robot.endpoint_pose()['orientation'].x,robot.endpoint_pose()['orientation'].y,robot.endpoint_pose()['orientation'].z,robot.endpoint_pose()['orientation'].w])
    rot = Rotation.from_quat(quat_as_list)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return np.array([robot.endpoint_pose()['position'][0],robot.endpoint_pose()['position'][1],(rot_euler[0]-np.sign(rot_euler[0])*180),rot_euler[1],rot_euler[2]])

def get_r_d(max_num_it):
    a = np.zeros((5,max_num_it))
    v = np.zeros((5,max_num_it))
    s = np.zeros((5,max_num_it))
    
    s[:,0]= get_r()
    if max_num_it>6500:
        a[0,4500:4510]=0.00001
        a[0,6490:6500]=-0.00001
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]
            s[:,i]=s[:,i-1]+v[:,i-1]
    return a,v,s

def generate_desired_trajectory(max_num_it,T,move_in_x=True):
    a = np.zeros((5,max_num_it))
    v = np.zeros((5,max_num_it))
    s = np.zeros((2,max_num_it))
    
    s[:,0]= get_p()

    if move_in_x:
        a[0,int(max_num_it*4/10):int(max_num_it*5/10)]=0.015*2
        a[0,int(max_num_it*7/10):int(max_num_it*8/10)]=-0.015*2

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            s[:,i]=s[:,i-1]+v[:2,i-1]*T
    return a,v,s

def plot_result(r_d,T):

    time_array = np.arange(len(r_d[0]))*T
    

    plt.subplot(111)
    plt.title("External force")
    
    #plt.plot(time_array, f_controlled[:], label="force z [N]")
    #plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, r_d[0,:], label="r_d x ", color='b',linestyle='dashed')
    plt.plot(time_array, r_d[1,:], label="r_d y ", color='C1',linestyle='dashed')
    #plt.plot(time_array, f_d[2,:], label="desired torque y [Nm]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.subplot(132)
    plt.title("dot")
    plt.plot(time_array, f_d_dot[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(133)
    plt.title("ddot")
    plt.plot(time_array, f_d_ddot[:], label="desired force z [N]", color='b',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.show()
# MAIN FUNCTION
if __name__ == "__main__":
    rospy.init_node("impedance_control")
    publish_rate = 50
    duration = 10
    rate = rospy.Rate(publish_rate)
    
    robot = PandaArm()
    robot.move_to_neutral() 

    max_num_it= int(publish_rate*duration)
    # TO BE INITIALISED BEFORE LOOP
    T = 0.001*(1000/publish_rate) #correct for sim
    a,v,s = generate_desired_trajectory_tc(max_num_it,T)
    plot_result(s,T)


