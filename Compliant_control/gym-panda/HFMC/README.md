# Hybrid Force/Motion Control (HFMC)

The scripts of this folder are running the HFMC through Gym, enabling use of modern RL techniques. 

In PILCO_HFMC_DualEnv.py, two models and policies are learned instead of one. In PILCO_HFMC_tripleEnv.py, three models and policies are learned. The shift(s) must be specified in the gym-environment, Compliant_control/gym-panda/HFMC/gym_panda/envs/HFMC_Env.py.

Loading is not working properly. In contrast to the sampled data itself, the models and policies are not equal when being loaded again. 
