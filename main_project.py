# -*- coding: utf-8 -*-
import time
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
import argparse
import torch
#from torch import nn
#from Nets import MLP,CNNCifar,CNNMnist
import flow_project

import MIMO
from optlib_project import Gibbs, Gibbs_active

#bachmark scripts, dependent on cvxpy
import DC_DS
import DC_RIS
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
print(os.environ["CUDA_VISIBLE_DEVICES"])

def initial():
    libopt = argparse.ArgumentParser()
    libopt.add_argument('--M', type=int, default=40, help='total # of devices')
    libopt.add_argument('--N', type=int, default=5, help='# of BS antennas')
    libopt.add_argument('--L', type=int, default=50, help='RIS Size')
    
    
    # optimization parameters
    libopt.add_argument('--nit', type=int, default=100, help='I_max,# of maximum SCA loops')                   
    libopt.add_argument('--Jmax', type=int, default=50, help='# of maximum Gibbs Outer loops')
    libopt.add_argument('--threshold', type=float, default=1e-2, help='epsilon,SCA early stopping criteria')
    libopt.add_argument('--tau', type=float, default=50, help=r'\tau, the SCA regularization term')  
    
    
    
    # simulation parameters
    libopt.add_argument('--trial', type=int, default=3, help='# of Monte Carlo Trials')  
    libopt.add_argument('--SNR', type=float, default=90.0, help='noise variance/0.1W in dB')  
    libopt.add_argument('--verbose', type=int, default=1, help=r'whether output or not')      
    libopt.add_argument('--set', type=int, default=2, help=r'=1 if concentrated devices+ euqal dataset;\
                        =2 if two clusters + unequal dataset')
    libopt.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # learning parameters
    libopt.add_argument('--gpu', type=int, default=0, help=r'Use Which Gpu')
    libopt.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    libopt.add_argument('--local_bs', type=int, default=0, help="0 for no effect,Local Bath size B")
    libopt.add_argument('--lr', type=float, default=0.01, help="learning rate,lambda")
    libopt.add_argument('--momentum', type=float, default=0.9, help="SGD momentum, used only for multiple local updates")
    libopt.add_argument('--epochs', type=int, default=500, help="rounds of training,T")
    
    args = libopt.parse_args()
    return args
    

if __name__ == '__main__':

    
    libopt = initial()
    libopt.transmitpower=0.1
    libopt.transmitpower_A = libopt.transmitpower
    libopt.lr=0.005
    libopt.epochs=300
    libopt.local_bs=50
    np.random.seed(libopt.seed)
    print(libopt)
    filename='./store/trial_{}_M_{}_N_{}_L_{}_SNR_{}_Tau_{}_set_{}_local_bs_{}.npz'.format(libopt.trial,
                            libopt.M,
                            libopt.N,libopt.L,
                            libopt.SNR,libopt.tau,libopt.set,libopt.local_bs)   
    print('save result to: \n {}'.format(filename))
    libopt.alpha_direct=3.76; # User-BS Path loss exponent
    fc=915*10**6 #carrier frequency, wavelength lambda=3.0*10**8/fc
    BS_Gain=10**(5.0/10) #BS antenna gain
    RIS_Gain=10**(5.0/10) #RIS antenna gain
    User_Gain=10**(0.0/10) #User antenna gain
    d_RIS=1.0/10 #dimension length of RIS element/wavelength
    libopt.BS=np.array([-50,0,10])
    libopt.RIS=np.array([0,0,10])
        
    libopt.range=20
    x0=np.ones([libopt.M],dtype=int)
    
    SCA_Gibbs_active=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    SCA_Gibbs_passive=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    DC_NORIS_set=np.ones([libopt.trial,])*np.nan
    Alt_Gibbs=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    DG_NORIS=np.ones([libopt.trial,])*np.nan
    

    result_list = []
    




    # Defining algorithm

    # TODO1: Active RIS + Noise(SCA+Gibbs)
    # TODO2: Passive RIS + Noise(SCA+Gibbs)
    # TOOD3: No RIS + Noise(DC+Gibbs)

    # TODO4: Active RIS + Noise-free(SCA+Gibbs)
    # TODO5: Active RIS + Noise + Select All device(SCA)
    



    # TODO1_DONE!: Active RIS + Noise(SCA+Gibbs)
    # TODO5_DONE!: Active RIS + Noise + Select All device(SCA)
    def Active_RIS(libopt,h_d,h_UR,h_RB,G,x,DS):
        print("Running Active RIS + Noise(SCA+Gibbs)")
        start = time.time()
        [x,obj,f,theta] = Gibbs_active(libopt,h_d,h_UR,h_RB,G,x,DS)
        end = time.time()
        print("Running time: {} seconds".format(end - start))     
        return x,obj,f,theta

    # TODO2_DONE!: Passive RIS + Noise(SCA+Gibbs),
    def Passive_RIS(libopt,h_d,G,x):
        print("Running Passive RIS + Noise(SCA+Gibbs)")
        start = time.time()
        [x,obj,f,theta] = Gibbs(libopt,h_d,G,x,True,True)        
        end = time.time()
        print("Running time: {} seconds".format(end - start))   
        return x,obj,f,theta
    
    # TOOD3_DONE!: No RIS + Noise(DC+Gibbs)
    def No_RIS(libopt,h_d):
        print("Running No RIS + Noise(DC+Gibbs)")
        gamma = [15]
        obj_new_NORIS,x_store_NORIS,f_store_NORIS=DC_DS.DC_NORIS(libopt,h_d,gamma,libopt.verbose)
        DC_NORIS_set[i]=obj_new_NORIS[0]
        return obj_new_NORIS,x_store_NORIS,f_store_NORIS
    
    # TODO4: No code needed

    
    for i in range(libopt.trial):
        libopt.device = torch.device('cuda:{}'.format(libopt.gpu))
        print(libopt.device)
        print('This is the {0}-th trial'.format(i))
        
        # ref=(1e-10)**0.5
        ref = 1.0
        sigma_n=np.power(10,-libopt.SNR/10)
        libopt.sigma=sigma_n/ref**2
        
        
        if libopt.set==1:
            libopt.K=np.ones(libopt.M,dtype=int)*int(30000.0/libopt.M)
            print(sum(libopt.K))
            libopt.dx2=np.random.rand(int(libopt.M-np.round(libopt.M/2)))*libopt.range-libopt.range#[100,100+range]
        else:
            libopt.K=np.random.randint(1000,high=2001,size=(int(libopt.M)))
            lessuser_size=int(libopt.M/2)
            libopt.K2=np.random.randint(100,high=201,size=(lessuser_size))
            libopt.lessuser=np.random.choice(libopt.M,size=lessuser_size, replace=False)
            libopt.K[libopt.lessuser]=libopt.K2
            print(sum(libopt.K))
            libopt.dx2=np.random.rand(int(libopt.M-np.round(libopt.M/2)))*libopt.range+100


        libopt.dx1=np.random.rand(int(np.round(libopt.M/2)))*libopt.range-libopt.range #[-range,0]
        libopt.dx=np.concatenate((libopt.dx1,libopt.dx2))
        libopt.dy=np.random.rand(libopt.M)*20-10
        libopt.d_UR=((libopt.dx-libopt.RIS[0])**2+(libopt.dy-libopt.RIS[1])**2+libopt.RIS[2]**2
                     )**0.5
        libopt.d_RB=np.linalg.norm(libopt.BS-libopt.RIS)
        libopt.d_RIS=libopt.d_UR+libopt.d_RB
        libopt.d_direct=((libopt.dx-libopt.BS[0])**2+(libopt.dy-libopt.BS[1])**2+libopt.BS[2]**2
                         )**0.5             
        libopt.PL_direct=BS_Gain*User_Gain*(3*10**8/fc/4/np.pi/libopt.d_direct)**libopt.alpha_direct
        libopt.PL_RIS=BS_Gain*User_Gain*RIS_Gain*libopt.L**2*d_RIS**2/4/np.pi\
        *(3*10**8/fc/4/np.pi/libopt.d_UR)**2*(3*10**8/fc/4/np.pi/libopt.d_RB)**2
        #channels
        h_d=(np.random.randn(libopt.N,libopt.M)+1j*np.random.randn(libopt.N,libopt.M))/2**0.5
        h_d=h_d@np.diag(libopt.PL_direct**0.5)/ref
        h_RB=(np.random.randn(libopt.N,libopt.L)+1j*np.random.randn(libopt.N,libopt.L))/2**0.5
        h_UR=(np.random.randn(libopt.L,libopt.M)+1j*np.random.randn(libopt.L,libopt.M))/2**0.5
        h_UR=h_UR@np.diag(libopt.PL_RIS**0.5)/ref
        
        
        G=np.zeros([libopt.N,libopt.L,libopt.M],dtype = complex)
        for j in range(libopt.M):
            G[:,:,j]=h_RB@np.diag(h_UR[:,j])
        x=x0

        # Finish exp setting

        

        # Start Running Different Algorithm

        # TODO1: Active RIS + Noise(SCA+Gibbs)
        active_RIS_noise = True


        # TODO2: Passive RIS + Noise(SCA+Gibbs)
        passive_RIS_noise = True


        # TOOD3: No RIS + Noise(DC+Gibbs)
        no_RIS_noise = False


        # TODO4: Noise-free
        active_RIS_noise_free = True


        # TODO5: Active RIS + Noise + Select All device(SCA)
        active_RIS_noise_wo_device = True


        if active_RIS_noise:
            [x_active,obj_active,f_active,theta_active]=Active_RIS(libopt,h_d,h_UR,h_RB,G,x,DS=True)     
        else:
            [x_active,obj_active,f_active,theta_active] = [0,0,0,0]



        if passive_RIS_noise:
            [x_passive,obj_passive,f_passive,theta_passive]=Passive_RIS(libopt,h_d,G,x)
        else:
            [x_passive,obj_passive,f_passive,theta_passive] = [0,0,0,0]



        if no_RIS_noise:
            [x_no,obj_no,f_no]=No_RIS(libopt,h_d)
        else:
            [x_no,obj_no,f_no] = [0,0,0]
            
        
        if active_RIS_noise_free:
            pass
            # no optimization needed
        
        if active_RIS_noise_wo_device:
            libopt.Jmax = 1
            [x_active_noDS,obj_active_noDS,f_active_noDS,theta_active_noDS]=Active_RIS(libopt,h_d,h_UR,h_RB,G,x,DS=True)     
        else:
            [x_active_noDS,obj_active_noDS,f_active_noDS,theta_active_noDS] = [0,0,0,0]

        
        dic={}



        dic['x_active']=copy.deepcopy(x_active)
        dic['f_active']=copy.deepcopy(f_active)
        dic['theta_active']=copy.deepcopy(theta_active)



        dic['x_passive']=copy.deepcopy(x_passive)
        dic['f_passive']=copy.deepcopy(f_passive)
        dic['theta_passive']=copy.deepcopy(theta_passive)

        dic['x_no']=copy.deepcopy(x_no)
        dic['f_no']=copy.deepcopy(f_no)

        # noise-free no setting need

        dic['x_active_noDS']=copy.deepcopy(x_active_noDS)
        dic['f_active_noDS']=copy.deepcopy(f_active_noDS)
        dic['theta_active_noDS']=copy.deepcopy(theta_active_noDS)



    
        # Start training according previous wireless setting by different algorithm

        
        print('lr{} batch{} ep{}'.format(libopt.lr,libopt.local_bs,libopt.epochs))



        start = time.time()
        result,_=flow_project.learning_flow_project(libopt,active_RIS_noise,passive_RIS_noise,no_RIS_noise,active_RIS_noise_free,active_RIS_noise_wo_device,
                                    dic,h_d,h_UR,h_RB,G)
        end = time.time()
        print("Running time: {} seconds".format(end - start))

        result_list.append(result)
    np.savez(filename, vars(libopt), result_list)
    # For test
        # Jmax = libopt.Jmax
        # M = libopt.M
        # N = libopt.N
        # L = libopt.L
        # obj_passive=np.zeros(Jmax+1)
        # f_passive=np.ones([N,Jmax+1],dtype = complex)
        # theta_passive=np.ones([L,Jmax+1],dtype = complex)
        # x_passive=np.ones([Jmax+1,M],dtype=int)