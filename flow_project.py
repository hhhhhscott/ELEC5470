# -*- coding: utf-8 -*-


#import matplotlib
#matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import copy
import numpy as np
# import argparse
np.set_printoptions(precision=6,threshold=1e3)
import torch
# from torch import nn
# from Gibbs_main import initial

#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
#from torchvision import models
#import torch.nn.functional as F


from Nets import CNNMnist
from AirComp_project import transmission_active, transmission_passive
import train_script
# import mse_lib
# from torch.utils.data import DataLoader
# from sklearn.datasets import make_blobs
# from torch.utils.data.sampler import SubsetRandomSampler

def FedAvg_grad(w_glob,grad,device):
    ind=0
    w_return=copy.deepcopy(w_glob)
    
    for item in w_return.keys():
        a=np.array(w_return[item].size())
        if len(a):
            b=np.prod(a)
            w_return[item]=copy.deepcopy(w_return[item])-torch.from_numpy(
                    np.reshape(grad[ind:ind+b],a)).float().to(device)
            ind=ind+b
    return w_return



def Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
              train_images,train_labels,test_images,test_labels,trans_mode,
              x,f,theta,h_d,h_UR,h_RB,G):

    len_active=len(idxs_users)
    loss_train = []
    accuracy_test=[]
    loss_test_set=[]
    
    net_glob.eval()
    acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
    accuracy_test.append(acc_test)
    net_glob.train()


    h=np.zeros([libopt.N,libopt.M],dtype=complex)

    if trans_mode != 0:
        for i in range(libopt.M):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta
    else:
        h = None
        pass

    # given h, 那个关于theta的函数
        

    for iter in range(libopt.epochs):
        #print('Overall Epoch: {}'.format(iter))
        grad_store_per_iter=np.zeros([len_active,d])
        #w_locals=[]
        loss_locals = []
        ind=0
        for idx in idxs_users:
            #print('Active User: {}'.format(idx))
            #print(len(dict_users[idx]))
            #print(int(libopt.K[idx]))
            if libopt.local_bs==0:
                size=int(libopt.K[idx])
            else:
                size=min(int(libopt.K[idx]),libopt.local_bs)
                
            w,loss,gradient=train_script.local_update(libopt,d,copy.deepcopy(net_glob).to(libopt.device),
                                         train_images,train_labels,idx,size)
        
            
            loss_locals.append(copy.deepcopy(loss))
            copyg=copy.deepcopy(gradient)
            copyg[np.isnan(copyg)]=1e2
            
            copyg[copyg>1e2]=1e2
            copyg[copyg<-1e2]=-1e2
            grad_store_per_iter[ind,:]=copyg
            ind=ind+1
        if trans_mode==0:
            grad=np.average(copy.deepcopy(grad_store_per_iter),axis=0,weights=
                            libopt.K[idxs_users]/sum(libopt.K[idxs_users]))
        elif trans_mode==1: # passive ris
            grad=transmission_passive(libopt,d,copy.deepcopy(grad_store_per_iter),
                                    x,f,h)
        elif trans_mode==2: # TODO active ris
            grad=transmission_active(libopt,d,copy.deepcopy(grad_store_per_iter),
                                    x,f,theta,h,h_d,h_UR,h_RB,G)
            pass

        
        grad[grad>1e2]=1e2
        grad[grad<-1e2]=-1e2
#        print(grad)
        w_glob=copy.deepcopy(FedAvg_grad(w_glob,libopt.lr*grad,libopt.device))
        net_glob.load_state_dict(w_glob)
        #loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if libopt.verbose:
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
        accuracy_test.append(acc_test)
        loss_test_set.append(loss_test)
        net_glob.train()
    return loss_train,accuracy_test,loss_test_set

def learning_flow_project(libopt,active_RIS_noise,passive_RIS_noise,no_RIS_noise,active_RIS_noise_free,active_RIS_noise_wo_device,
                  dic,h_d,h_UR,h_RB,G):
    
    # 1
    x_active=dic['x_active']
    f_active=dic['f_active']
    theta_active=dic['theta_active']

    # 2
    x_passive=dic['x_passive']
    f_passive=dic['f_passive']
    theta_passive=dic['theta_passive']

    # 3
    x_no=dic['x_no']
    f_no=dic['f_no']

    # 4
    # # noise-free no setting is needed

    # 5
    x_active_noDS=dic['x_active_noDS']
    f_active_noDS=dic['f_active_noDS']
    theta_active_noDS=dic['theta_active_noDS']


    
    
    torch.manual_seed(libopt.seed)
    result={}
    

    train_images,train_labels,test_images,test_labels=train_script.Load_FMNIST_IID(libopt.M,libopt.K)
    net_glob = CNNMnist(num_classes=10,num_channels=1,batch_norm=True).to(libopt.device)

    w_glob = net_glob.state_dict()
    w_0=copy.deepcopy(w_glob)
    
    
    d=0
    for item in w_glob.keys():
        d=d+int(np.prod(w_glob[item].shape))

    # TODO Need to verify the active RIS condition
    if active_RIS_noise:
        print('Active RIS case is running')
        x = x_active[libopt.Jmax]
        f = f_active[:,libopt.Jmax]
        theta=theta_active[:,libopt.Jmax]
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x==1]
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,2,
                                               x,f,theta,h_d,h_UR,h_RB,G)
        result['loss_train1']=np.asarray(loss_train)
        result['accuracy_test1']=np.asarray(accuracy_test)
        result['loss_test1']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test1'][len(result['accuracy_test1'])-1]))
    
    # Done
    if passive_RIS_noise:
        print('Passive RIS case is running')
        x = x_passive[libopt.Jmax]
        f = f_passive[:,libopt.Jmax]
        theta=theta_passive[:,libopt.Jmax]
        
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
#    
        
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x==1]
        loss_train1,accuracy_test1,loss_test1=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 x,f,theta,h_d,h_UR,h_RB,G)
        result['loss_train2']=np.asarray(loss_train1)
        result['accuracy_test2']=np.asarray(accuracy_test1)
        result['loss_test2']=np.asarray(loss_test1)
        print('result {}'.format(result['accuracy_test2'][len(result['accuracy_test2'])-1]))


    # Done
    if no_RIS_noise:
        print('No RIS case is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
        x = x_no[:,0]
        f = f_no[:,0]
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x==1]
        loss_train3,accuracy_test3,loss_test3=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 x,f,None,h_d,h_UR,h_RB,G
                                                 )
        result['loss_train3']=np.asarray(loss_train3)
        result['accuracy_test3']=np.asarray(accuracy_test3)
        result['loss_test3']=np.asarray(loss_test3)
        print('result {}'.format(result['accuracy_test3'][len(result['accuracy_test3'])-1]))
    

    # Done
    if active_RIS_noise_free:
        print('Noiseless Case is running')
        idxs_users=range(libopt.M)
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,0,
                                               None,None,None,h_d,h_UR,h_RB,G)
        result['loss_train4']=np.asarray(loss_train)
        result['accuracy_test4']=np.asarray(accuracy_test)
        result['loss_test4']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test4'][len(result['accuracy_test4'])-1]))


    # TODO
    if active_RIS_noise_wo_device:
        print("Active RIS w/o device selection case is running")   
        x = x_active_noDS[libopt.Jmax]
        f=f_active_noDS[:,libopt.Jmax]
        theta=theta_active_noDS[:,libopt.Jmax]
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x==1]
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,2,
                                               x,f,theta,h_d,h_UR,h_RB,G)
        result['loss_train5']=np.asarray(loss_train)
        result['accuracy_test5']=np.asarray(accuracy_test)
        result['loss_test5']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test5'][len(result['accuracy_test5'])-1])) 
    return result,d


if __name__ == '__main__':
    pass