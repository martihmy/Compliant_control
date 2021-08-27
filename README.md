# Compliant Control

This repository is based on justagist's [PandaSimulator](https://github.com/justagist/panda_simulator). The folder Compliant_control contains the work related to my master's thesis "Learning Compliant Robotic Manipulation". In the folder, three compliant force controllers are implemented for the Panda robot: Admittance Control (AC), Hybrid Force/Motion Control (HFMC) and Force-based Variable Impedance Control (VIC). Nrontsis' repository [PILCO](https://github.com/nrontsis/PILCO) provides functionality for the model-based Reinforcement Learning algorithm PILCO. Compliant_control provides compatibility between the PILCO-implementation and the implemented force controllers, constituting a PILCO-framework for learning control-strategies.


# Setting Up the Environment


## Setup of the extended Franka Control Interface (FCI)

The extended FCI can be set up in an Ubuntu 18.04 system in the following steps:


  - Install ROS Melodic Morenia

  
 - Build libfranka and franka_ros
  
  -  Set up the real-time kernel
  
  - Build [PandaSimulator](https://github.com/justagist/panda_simulator) and [PandaRobot](https://github.com/justagist/panda_robot) 


Both the building of libfranka and franka_ros and the setting up of the real-time kernel, are described in detail in https://frankaemika.github.io/docs/installation_linux.html. The process of building panda_simulator and panda_robot is done according to the respective README files on Github. franka_ros_interface and franka_panda_description are both installed when performing the fourth step.

## Setup of the PILCO compatibility

All compatibility related to PILCO can be achieved through the following steps:



  1) Install the [PILCO](https://github.com/nrontsis/PILCO) implementation
  
  2) Install OpenAI Gym
  
  3) Install execnet
  


The PILCO implementation should preferable be installed in a fresh Conda environment with Python >= 3.7. The installation itself is performed by running the following two commands in the terminal: 


- git clone https://github.com/nrontsis/PILCO && cd PILCO}

- python setup.py develop


Step 2 and 3 are performed by similarly running



- pip install gym

in the python 2 environment, and  

- pip install execnet


in the Conda environment. 

## Setup of the implemented controllers

In order to run the implemented controllers, this package (https://github.com/martihmy/Compliant_control), is to replace panda_simulator. Additionally, the four python scripts in the folder Modifications, is to replace the files with the same names in the PILCO-, franka_ros_interface- and panda_robot packages respectively. 
