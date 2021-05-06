import numpy as np
import gpflow
import tensorflow as tf
#from gpflow import autoflow
#from gpflow import settings
from pilco.models.pilco import PILCO
from pilco.rewards import ExponentialReward


def save_pilco_model(pilco_object,X1,X,Y,target,W_diag,path,rbf=True):
    for i,m in enumerate(pilco_object.mgpr.models):
        tf.saved_model.save(m,path + '/model'+ str(i) )

    if rbf:
        for i,m in enumerate(pilco_object.controller.models):
            tf.saved_model.save(m,path + '/control_model'+ str(i) )
    else:
        print('W = ',pilco_object.controller.W)
        print('b = ',pilco_object.controller.B)
        #np.savetxt(path + '/controller_W.csv', pilco_object.controller.W, delimiter=',')
        #np.savetxt(path + '/controller_b.csv', pilco_object.controller.b, delimiter=',')


    np.savetxt(path + '/X1.csv', X1, delimiter=',')
    np.savetxt(path + '/X.csv', X, delimiter=',')
    np.savetxt(path + '/Y.csv', Y, delimiter=',')
    #does this work?
    np.savetxt(path + '/target.csv', target, delimiter=',')
    np.savetxt(path + '/W_diag.csv', W_diag, delimiter=',')


def load_pilco_model(path,controller,horizon, rbf=True):
    #saved_data = tf.saved_model.load(path)
    X1 = np.loadtxt(path + '/X1.csv', delimiter=',')
    X = np.loadtxt(path + '/X.csv', delimiter=',')
    Y = np.loadtxt(path + '/Y.csv', delimiter=',')
    target = np.loadtxt(path + '/target.csv', delimiter=',')
    W_diag = np.loadtxt(path + '/W_diag.csv', delimiter=',')
    

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim 
 
    norm_env_m = np.mean(X1[:,:state_dim],0)
    norm_env_std = np.std(X1[:,:state_dim], 0)

    m_init =  np.transpose(X[0,:-control_dim,None])
    S_init =  0.5 * np.eye(state_dim)

    reward = ExponentialReward(state_dim=state_dim, t=np.divide(target - norm_env_m, norm_env_std),W=np.diag(W_diag))
    pilco = PILCO((X,Y),horizon=horizon, controller=controller, reward=reward,m_init=m_init, S_init=S_init)
    for i,m in enumerate(pilco.mgpr.models):
        m = tf.saved_model.load(path + '/model'+ str(i))
    
    if rbf:
        for i,m in enumerate(pilco.controller.models):
            m = tf.saved_model.load(path + '/control_model'+ str(i))
    
    return pilco,X1, m_init, S_init, state_dim, X, Y, target, W_diag
