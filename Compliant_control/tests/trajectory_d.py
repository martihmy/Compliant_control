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


def get_p(two_dim=False):
    if two_dim == True:
        return robot.endpoint_pose()['position'].reshape([3,1])
    else:
        return robot.endpoint_pose()['position']

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


def plot_result(r_d, v_d, a_d,T):

    time_array = np.arange(len(r_d[0]))*T
    

    plt.subplot(111)
    plt.title("desired trajectory")
    plt.plot(time_array, r_d[0,:], label="r_d x ", color='b',linestyle='dashed')
    plt.plot(time_array, r_d[1,:], label="r_d y ", color='C1',linestyle='dashed')
    plt.plot(time_array, r_d[2,:], label="r_d z", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.subplot(132)
    plt.title("desired vel")
    plt.plot(time_array, v_d[0,:], label="v_d x ", color='b')
    plt.plot(time_array, v_d[1,:], label="v_d y ", color='C1')
    plt.plot(time_array, v_d[2,:], label="v_d z", color='g')
    plt.xlabel("Real time [s]")
    plt.legend()

    plt.subplot(133)
    plt.title("desired acc")
    plt.plot(time_array, a_d[0,:], label="a_d x ", color='b')
    plt.plot(time_array, a_d[1,:], label="a_d y ", color='C1')
    plt.plot(time_array, a_d[2,:], label="a_d z", color='g')
    plt.xlabel("Real time [s]")
    plt.legend()
    """
    plt.show()


def generate_desired_trajectory_express(iterations,T):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    
    if iterations > 175:
        a[2,0:50]=-0.00002/T**2
        a[2,125:175]=0.00002/T**2
        
    if iterations > 3250:
        a[0,2250:2255]=0.00002/T**2
        a[0,3245:3250]=-0.00002/T**2
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p

def generate_desired_trajectory_time_consistent_VIC(iterations,T):
    a = np.zeros((6,iterations))
    v = np.zeros((6,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = get_p()
    

    a[2,0:int(iterations/75)]=-1.25
    a[2,int(iterations*2/75):int(iterations/25)]= 1.25
    #a[2,int(iterations/30):int(iterations*7/150)]= 1.25        
    
    a[0,int(iterations*3/5):int(iterations*451/750)]=1.25
    a[0,int(iterations*649/750):int(iterations*13/15)]=-1.25

    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:3,i-1]*T
    return a,v,p



def generate_desired_trajectory_tc(iterations,T,move_in_x=False): #admittance
    a = np.zeros((3,iterations))
    v = np.zeros((3,iterations))
    p = np.zeros((3,iterations))
    p[:,0] = robot.endpoint_pose()['position']

    a[2,0:int(max_num_it/75)]=-0.625/5
    a[2,int(max_num_it/75):int(max_num_it*2/75)]=0.625/5
        
    if move_in_x:
        a[0,int(max_num_it*3/5):int(max_num_it*451/750)]=1.25/5
        a[0,int(max_num_it*649/750):int(max_num_it*13/15)]=-1.25/5
    for i in range(max_num_it):
        if i>0:
            v[:,i]=v[:,i-1]+a[:,i-1]*T
            p[:,i]=p[:,i-1]+v[:,i-1]*T
    return a,v,p

# MAIN FUNCTION
if __name__ == "__main__":
    rospy.init_node("impedance_control")
    publish_rate = 250
    rate = rospy.Rate(publish_rate)
    
    robot = PandaArm()
    robot.move_to_neutral() 

    duration = 15
    T = 0.001*(1000/publish_rate) #correct for sim
    max_num_it= int(duration / T)
    # TO BE INITIALISED BEFORE LOOP
    
    a,v,s = generate_desired_trajectory_tc(max_num_it,T)
    plot_result(s,v,a,T)


