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


def plot_result(f_d,f_d_dot,f_d_ddot, T):

    time_array = np.arange(len(f_d))*T
    

    plt.subplot(131)
    plt.title("External force")
    
    #plt.plot(time_array, f_controlled[:], label="force z [N]")
    #plt.plot(time_array, f_controlled[2,:], label="torque y [Nm]")
    plt.plot(time_array, f_d[:], label="desired force z [N]", color='b',linestyle='dashed')
    #plt.plot(time_array, f_d[1,:], label="desired torque x [Nm]", color='C1',linestyle='dashed')
    #plt.plot(time_array, f_d[2,:], label="desired torque y [Nm]", color='g',linestyle='dashed')
    plt.xlabel("Real time [s]")
    plt.legend()

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

    plt.show()


def get_F_d_steep(max_num_it,T): #current
    a = np.zeros(max_num_it)
    a[0:10]=0.01
    a[20:29]=-0.01
    a[30] = -0.002
    if max_num_it >4001:
        a[1500:1508]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s


def get_F_d(max_num_it,T):
    a = np.zeros(max_num_it)
    a[0:10]=0.001
    if max_num_it >4001:
        a[1500:1510]=-0.001
        it = 2000
        while it <= 4000:
            a[it]= -9*(np.pi**2)*(T/4)**2*np.sin(it*T/4*2*np.pi+np.pi/2)
            it+=1

        a[4001]=0.0001
    v = np.zeros(max_num_it)
    s = np.zeros(max_num_it)
    for i in range(max_num_it):
        if i>0:
            v[i]=v[i-1]+a[i-1]
            s[i]=s[i-1]+v[i-1]

    return a,v,s

# MAIN FUNCTION

if __name__ == "__main__":
    max_num_it=7500
    # TO BE INITIALISED BEFORE LOOP
    T = 0.001*(1000/250) #correct for sim


    f_d_ddot,f_d_dot, f_d = get_F_d(max_num_it,T)


    plot_result(f_d,f_d_dot,f_d_ddot,T)

