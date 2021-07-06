
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


import math


# Same as above + DUALenv
data = np.array([[2.44,  0.302,  3.012],
                [5.444,  0.365,  3.491],
                [16.357,  0.676,  3.878],
                [15.976,  0.547,  3.821],
                [14.664,  0.457,  3.476],
                [14.254,  0.443,  3.678],
                [12.941,  0.465,  3.29],
                [10.161,  0.498,  3.665],
                [10.154,  0.413,  3.2],
                [11.041,  0.539,  3.202],
                [11.855,  0.407,  4.717],
                [12.13,  0.429,  4.983]])
rewards = np.array([6.139,9.168,1.503,2.789,4.896,6.955,11.022,8.988,5.598,7.568,3.796])

F = data[:,0]
pos = data[:,1]
vel = data[:,2]



#PLOTTING

plt.figure(figsize=(10,6))
plt.suptitle('Difficulty of modeling non-homogeneous transition dynamics', size =16)

plt.subplot(121)
plt.title('Variance in the GP models')
plt.xlabel("number of model- and policy optimizations")
xaxis = range(1,math.ceil(len(F))+1)
plt.xticks(xaxis)


plt.plot(list(range(1,len(F)+1)),F, label=r'$\sigma^2$ in force model')
plt.plot(list(range(1,len(pos)+1)),pos, label=r'$\sigma^2$ in position model')
plt.plot(list(range(1,len(vel)+1)),vel, label=r'$\sigma^2$ in velocity model')

plt.legend()

plt.subplot(122)
plt.title('Fluctuation in policy')

plt.xlabel("number of model- and policy optimizations")
xaxis = range(1,math.ceil(len(F))+1)
plt.xticks(xaxis)

plt.plot(range(1,len(rewards)+1),rewards/len(rewards), label='average expected reward')
plt.legend()
plt.show()