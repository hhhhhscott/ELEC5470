# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
#import copy
# import argparse


def transmission_passive(libopt,d,signal,x,f,h):
    index=(x==1)
    N=libopt.N
    K=libopt.K[index]
    K2=K**2
    inner=f.conj()@h[:,index]
    inner2=np.abs(inner)**2
    g=signal
    mean=np.mean(g,axis=1)
    g_bar=K@mean
    var=np.var(g,axis=1)
    var[var<1e-3]=1e-3
    var_sqrt=var**0.5
    eta=np.min(libopt.transmitpower*inner2/K2/var)
    eta_sqrt=eta**0.5
    b=K*eta_sqrt*var_sqrt*inner.conj()/inner2   #p_m
    noise_power=libopt.sigma*libopt.transmitpower
    n=(np.random.randn(N,d)+1j*np.random.randn(N,d))/(2)**0.5*noise_power**0.5
    x_signal=np.tile(b/var_sqrt,(d,1)).T*(g-np.tile(mean,(d,1)).T)
    y=h[:,index]@x_signal+n
    w=np.real((f.conj()@y/eta_sqrt+g_bar))/sum(K)
    # w=np.real((f.conj()@y/eta_sqrt+g_bar))
    return w

def transmission_active(libopt,d,signal,x,f,theta,h,h_d,h_UR,h_RB,G): #没写完 

    # theta shape [L]
    # f shape [N]
    # h_d shape [N,M]
    # h_UR [L,M]
    # h_RB [N,L]
    # G [N, L, M]
    # 


    index=(x==1)

    N=libopt.N

    L=libopt.L

    K=libopt.K[index]

    K2=K**2

    # inner=f.conj()@h[:,index]

    # inner2=np.abs(inner)**2

    P_T = libopt.transmitpower

    P_A = libopt.transmitpower_A

    power_s = libopt.sigma*libopt.transmitpower

    power_d = libopt.sigma*libopt.transmitpower


    # g = signal

    # mean = np.mean(g,axis=1)

    # g_bar = K@mean

    # var = np.var(g,axis=1)

    # var[var<1e-3]=1e-3

    # var_sqrt=var**0.5    
    
    # min_term2 = (P_A-)/()



    
    inner=f.conj()@h[:,index]
    inner2=np.abs(inner)**2
    #print(inner[index])
    g=signal
    #mean and variance
    mean=np.mean(g,axis=1)
    g_bar=K@mean
    
    
    
    var=np.var(g,axis=1)
    
    var[var<1e-3]=1e-3
    
    var_sqrt=var**0.5
    Theta = np.diag(theta)
    Theta_h_Rm_norm2_m = np.sum(np.abs(Theta @ h_UR[:,index]) ** 2, axis=0)
    h_R_m = np.sum(np.abs(h_UR[:,index]) ** 2, axis = 0)
    


    min_1 = np.min(P_T*inner2/K2/var)
    min_2 = (P_A-power_d*(np.sum(np.abs(theta) ** 2))) / np.sum(K2*var*(Theta_h_Rm_norm2_m-h_R_m)/inner2)
    eta = np.min([min_1,min_2])
    eta_sqrt=eta**0.5

    b=K*eta_sqrt*var_sqrt*inner.conj()/inner2
    
    
    
    n_s=(np.random.randn(N,d)+1j*np.random.randn(N,d))/(2)**0.5*power_s**0.5
    n_d=(np.random.randn(L,d)+1j*np.random.randn(L,d))/(2)**0.5*power_d**0.5
    

    x_signal=np.tile(b/var_sqrt,(d,1)).T*(g-np.tile(mean,(d,1)).T)


    # y=h[:,index]@x_signal+f.conj()@h_RB@Theta@n_d+f.conj()@n_s
    # w=np.real(f.conj()@y/eta_sqrt+g_bar)/sum(K)

    y=f.conj()@h[:,index]@x_signal+f.conj()@h_RB@Theta@n_d/100.0+f.conj()@n_s
    w=np.real(y/eta_sqrt+g_bar)/sum(K)

    # y=f.conj()@h[:,index]@x_signal+f.conj()@n_s
    # w=np.real(y/eta_sqrt+g_bar)/sum(K)
    
    


    return w

if __name__ == '__main__':
   pass
    
    