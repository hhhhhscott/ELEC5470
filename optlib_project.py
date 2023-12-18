# -*- coding: utf-8 -*-
import copy
import numpy as np
import cvxpy as cp


np.set_printoptions(precision=6,threshold=1e3)
from scipy.optimize import minimize

def norm_2_2(x):
    return np.sum(np.abs(x) ** 2)


def sca_fmincon(libopt,h_d,G,f,theta,x,K2,RISON):
    N=libopt.N
    L=libopt.L
    I=sum(x)
    tau=libopt.tau
    if theta is None:
        theta=np.ones([L],dtype=complex)
    if not RISON:
        theta=np.zeros([L],dtype=complex)
    result=np.zeros(libopt.nit)
    h=np.zeros([N,I],dtype=complex)
    for i in range(I):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta
        
    if f is None:
        f=h[:,0]/np.linalg.norm(h[:,0])
   
    obj=min(np.abs(np.conjugate(f)@h)**2/K2)
    threshold=libopt.threshold
    
    for it in range(libopt.nit):
        obj_pre=copy.deepcopy(obj)
        a=np.zeros([N,I],dtype=complex)
        b=np.zeros([L,I],dtype=complex)
        c=np.zeros([1,I],dtype=complex)
        F_cro=np.outer(f,np.conjugate(f));
        for i in range(I):
            # a[:,i]=tau*K2[i]*f+np.outer(h[:,i],np.conjugate(h[:,i]))@f
            # if RISON:

            #     b[:,i]=tau*K2[i]*theta+G[:,:,i].conj().T@F_cro@h[:,i]
            #     c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]*(L+1)+2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])
            # else:
            #     c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]
            a[:,i]=tau*f+np.outer(h[:,i],np.conjugate(h[:,i]))@f
            if RISON:

                b[:,i]=tau*theta+G[:,:,i].conj().T@F_cro@h[:,i]
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*(L+1)+2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])
            else:
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau
        
        
        #print(c.shape)
        
        fun=lambda mu: np.real(2*np.linalg.norm(a@mu)+2*np.linalg.norm(b@mu,ord=1)-c@mu)
        
        cons = ({'type': 'eq', 'fun': lambda mu:  K2@mu-1})
        bnds=((0,None) for i in range(I))
        res = minimize(fun, 1/K2,   bounds=tuple(bnds), constraints=cons)
        if ~res.success:
            pass
            #print('Iteration: {}, solution:{} obj:{:.6f}'.format(it,res.x,res.fun[0]))
            #print(res.message)
            #return
        fn=a@res.x
        thetan=b@res.x
        fn=fn/np.linalg.norm(fn)
#        thetan=thetan/np.abs(thetan)
        if RISON:
            thetan=thetan/np.abs(thetan)
            theta=thetan
        f=fn
        for i in range(I):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta
        obj=min(np.abs(np.conjugate(f)@h)**2/K2)
        result[it]=copy.deepcopy(obj)
        if libopt.verbose>2:
            print('  Iteration {} Obj {:.6f} Opt Obj {:.6f}'.format(it,result[it],res.fun[0]))
        if np.abs(obj-obj_pre)/min(1,abs(obj))<=threshold:
            break
        
        #print(res)
    if libopt.verbose>1:
        print(' SCA Take {} iterations with final obj {:.6f}'.format(it+1,result[it]))
    result=result[0:it]
    return f,theta,result




def find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,f0,theta0,RISON):
    N=libopt.N
    L=libopt.L
    M=libopt.M
    if sum(x)==0:
        obj=np.inf
        
        theta=np.ones([L],dtype=complex)
        f=h_d[:,0]/np.linalg.norm(h_d[:,0])
        if not RISON:
            theta=np.zeros([L])
    else:
         index=(x==1)
         #print(index)

         f,theta,_=sca_fmincon(libopt,h_d[:,index],G[:,:,index],f0,theta0,x,K2[index],RISON)

         h=np.zeros([N,M],dtype=complex)
         for i in range(M):
             h[:,i]=h_d[:,i]+G[:,:,i]@theta
         gain=K2/(np.abs(np.conjugate(f)@h)**2)*libopt.sigma
         #print(gain)
         #print(gain)
         #print(2/Ksum2*(sum(K[~index]))**2)
         #print(np.max(gain[index])/(sum(K[index]))**2)
         obj=np.max(gain[index])/(sum(K[index]))**2+4/Ksum2*(sum(K[~index]))**2
    return obj,x,f,theta
def Gibbs(libopt,h_d,G,x0,RISON,Joint):
    #initial
    
    N=libopt.N
    L=libopt.L
    M=libopt.M
    Jmax=libopt.Jmax
    K=libopt.K/np.mean(libopt.K) #normalize K to speed up floating computation
    K2=K**2
    Ksum2=sum(K)**2
    x=x0
    # inital the return values
    obj_new=np.zeros(Jmax+1)
    f_store=np.zeros([N,Jmax+1],dtype = complex)
    theta_store=np.zeros([L,Jmax+1],dtype = complex)
    x_store=np.zeros([Jmax+1,M],dtype=int)
    
    #the first loop
    ind=0
    [obj_new[ind],x_store[ind,:],f,theta]=find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,None,None,RISON)
    
    theta_store[:,ind]=copy.deepcopy(theta)
    f_store[:,ind]=copy.deepcopy(f)
#    beta=min(max(obj_new[ind],1)
    beta=min(1,obj_new[ind])
    # print(beta)
    alpha=0.9
    if libopt.verbose>1:
        print('The inital guess: {}, obj={:.6f}'.format(x,obj_new[ind]))
    elif libopt.verbose==1:
        print('The inital guess obj={:.6f}'.format(obj_new[ind]))
    f_loop=np.tile(f,(M+1,1))

    theta_loop=np.tile(theta,(M+1,1))
    #print(theta_loop.shape)
    #print(theta_loop[0].shape)
    for j in range(Jmax):
        if libopt.verbose>1:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f}'.format(j+1,beta));
        
        #store the possible transition solution and their objectives
        X_sample=np.zeros([M+1,M],dtype=int)
        Temp=np.zeros(M+1)
        #the first transition => no change
        X_sample[0,:]=copy.deepcopy(x)
        Temp[0]=copy.deepcopy(obj_new[ind])
        f_loop[0]=copy.deepcopy(f)
        theta_loop[0]=copy.deepcopy(theta)
        #2--M+1-th trnasition, change only 1 position
        for m in range(M):
            if libopt.verbose>1:
                print('the {}-th:'.format(m+1))
            #filp the m-th position
            x_sam=copy.deepcopy(x)
            x_sam[m]=copy.deepcopy((x_sam[m]+1)%2)
            X_sample[m+1,:]=copy.deepcopy(x_sam);
            Temp[m+1],_,f_loop[m+1],theta_loop[m+1]=find_obj_inner(libopt,
                x_sam,K,K2,Ksum2,h_d,G,f_loop[m+1],theta_loop[m+1],RISON)
            if libopt.verbose>1:
                print('          sol:{} with obj={:.6f}'.format(x_sam,Temp[m+1]))
        temp2=Temp;
        
        Lambda=np.exp(-1*temp2/beta);
        Lambda=Lambda/sum(Lambda);
        while np.isnan(Lambda).any():
            if libopt.verbose>1:
                print('There is NaN, increase beta')
            beta=beta/alpha;
            Lambda=np.exp(-1.*temp2/beta);
            Lambda=Lambda/sum(Lambda);
        
        if libopt.verbose>1:
            print('The obj distribution: {}'.format(temp2))
            print('The Lambda distribution: {}'.format(Lambda))
        kk_prime=np.random.choice(M+1,p=Lambda)
        x=copy.deepcopy(X_sample[kk_prime,:])
        f=copy.deepcopy(f_loop[kk_prime])
        theta=copy.deepcopy(theta_loop[kk_prime])
        ind=ind+1
        obj_new[ind]=copy.deepcopy(Temp[kk_prime])
        x_store[ind,:]=copy.deepcopy(x)
        theta_store[:,ind]=copy.deepcopy(theta)
        f_store[:,ind]=copy.deepcopy(f)
        
        if libopt.verbose>1:
            print('Choose the solution {}, with objective {:.6f}'.format(x,obj_new[ind]))
            
        if libopt.verbose:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}'.format(j+1,beta,obj_new[ind]))
        beta=max(alpha*beta,1e-4)
        
    return x_store,obj_new,f_store,theta_store


def gradient_f_star_u(P_A, P_T, sigma_d2, sigma_s2, h_RB, h_m, f,theta, K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    Theta = np.diag(theta)
    f_Her_H_Theta_norm2 = norm_2_2(f_.conjugate().T @ h_RB @ Theta)
    f_Her_h_m_2 = np.abs(f_.conjugate().T @ h_m_) ** 2  
    H_Theta_Theta_Her_H_Her_f = h_RB @ Theta @ Theta.conjugate().T @ h_RB.conjugate().T @ f_
    grad_f_star_u = - P_T / K2_m * (h_m_ @ h_m_.conjugate().T @ f_ * (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2)
                                    + H_Theta_Theta_Her_H_Her_f * sigma_d2 * f_Her_h_m_2)
    return grad_f_star_u

def gradient_theta_star_u(P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    h_UR_m_ = h_UR_m[:, None]
    Theta = np.diag(theta)
    f_Her_H_Theta_norm2 = norm_2_2(f_.conjugate().T @ h_RB @ Theta)
    f_Her_h_m_2 = np.abs(f_.conjugate().T @ h_m_) ** 2  
    inner_f_h_m = (f_.conjugate().T @ h_m_)[0, 0]
    H_Her_f = h_RB.conjugate().T @ f_
    grad_theta_star_u = - P_T / K2_m * (inner_f_h_m * h_UR_m_ * H_Her_f * (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2)
                                    + f_Her_h_m_2 * sigma_d2 * (Theta.conjugate().T @ h_RB.conjugate().T @ f_).conjugate() * H_Her_f)
    return grad_theta_star_u


def gradient_f_star_v(grad_f_star_u, P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    h_UR_m_ = h_UR_m[:, None]
    Theta = np.diag(theta)
    carc_M = np.sum(x)
    Theta_h_R_m_norm2 = norm_2_2(Theta @ h_UR_m_)
    h_R_m_norm2 = norm_2_2(h_UR_m_)
    grad_f_star_v = (P_A - sigma_d2 * norm_2_2(theta)) / (carc_M * P_T * (Theta_h_R_m_norm2 - h_R_m_norm2)) * grad_f_star_u
    return grad_f_star_v


def gradient_theta_star_v(grad_theta_star_u, P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    h_UR_m_ = h_UR_m[:, None]

    theta_ = theta[:, None]
    Theta = np.diag(theta)
    theta_norm_2 = norm_2_2(theta)
    card_M = np.sum(x)
    Theta_h_R_m_norm2 = norm_2_2(Theta @ h_UR_m_)
    h_R_m_norm2 = norm_2_2(h_UR_m_)
    item_1 = Theta_h_R_m_norm2 - h_R_m_norm2 
    f_Her_h_m_2 = np.abs(f_.conjugate().T @ h_m_) ** 2 
    f_Her_H_Theta_norm2 = norm_2_2(f_.conjugate().T @ h_RB @ Theta)
    d_1 = - K2_m / P_T * grad_theta_star_u * (P_A - sigma_d2 * theta_norm_2) - \
          sigma_d2 * f_Her_h_m_2 * (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2) * theta_
    d_2 = theta_ * h_UR_m_ * h_UR_m_.conjugate()
    grad_theta_star_v = (d_1 * item_1 - 
                         d_2 * (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2) * (P_A - sigma_d2 * theta_norm_2) * f_Her_h_m_2) / \
                        (item_1 ** 2 * card_M * K2_m)
    return grad_theta_star_v

def u_m(P_A, P_T, sigma_d2, sigma_s2, h_RB, h_m, f,theta, K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    Theta = np.diag(theta)
    f_Her_H_Theta_norm2 = norm_2_2(f_.conjugate().T @ h_RB @ Theta)
    f_Her_h_m_2 = np.abs(f_.conjugate().T @ h_m_) ** 2  
    u_m_ = - P_T / K2_m * (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2) * f_Her_h_m_2
    return u_m_

def v_m(P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m):
    # h_m [1, N]
    h_m_ = h_m[:, None]
    f_ = f[:, None]
    Theta = np.diag(theta)
    card_M = np.sum(x)
    h_UR_m_ = h_UR_m[:, None]
    Theta_h_R_m_norm2 = norm_2_2(Theta @ h_UR_m_)
    h_R_m_norm2 = norm_2_2(h_UR_m_)
    item_1 = Theta_h_R_m_norm2 - h_R_m_norm2 
    f_Her_H_Theta_norm2 = norm_2_2(f_.conjugate().T @ h_RB @ Theta)
    f_Her_h_m_2 = np.abs(f_.conjugate().T @ h_m_) ** 2  
    theta_norm_2 = norm_2_2(theta)   
    v_m_ = (sigma_d2 * f_Her_H_Theta_norm2 + sigma_s2) * (P_A - sigma_d2 * theta_norm_2) *  f_Her_h_m_2 / \
    (item_1 ** 2 * card_M * K2_m)
    return v_m_

def get_a_b_c_uv_m(libopt, h_m, h_UR_m, h_RB, h_d_m, f, theta, x, K2_m):
    tau_u, tau_v = libopt.tau, 10 * libopt.tau
    P_T = libopt.transmitpower
    P_A = libopt.transmitpower_A
    f_ = f[:, None]
    card_M = np.sum(x)
    theta_ = theta[:, None]
    sigma_d2 = libopt.sigma * libopt.transmitpower
    sigma_s2 = libopt.sigma * libopt.transmitpower
    Theta = np.diag(theta)
    h_m = h_d_m + h_RB @ Theta @ h_UR_m

    ## 计算梯度
    grad_f_star_u = gradient_f_star_u(P_A, P_T, sigma_d2, sigma_s2, h_RB, h_m, f,theta, K2_m)
    grad_theta_star_u = gradient_theta_star_u(P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m)
    grad_f_star_v = gradient_f_star_v(grad_f_star_u, P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m)
    grad_theta_star_v = gradient_theta_star_v(grad_theta_star_u, P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m)
    
    ### 计算函数值
    u_m_ = u_m(P_A, P_T, sigma_d2, sigma_s2, h_RB, h_m, f,theta, K2_m)
    v_m_ = v_m(P_A, P_T, sigma_d2, sigma_s2, h_UR_m, h_RB, h_d_m, h_m, f,theta,x,K2_m)

    alpha_u_m = P_T * tau_u / K2_m
    a_u_m__i = - grad_f_star_u + alpha_u_m * f_
    b_u_m__i = - grad_theta_star_u + alpha_u_m * theta_
    c_u_m__i = u_m_ - 2 * (f_.conjugate().T @ grad_f_star_u + theta_.conjugate().T @ grad_theta_star_u).real + 2 * alpha_u_m

    alpha_v_m = P_A * tau_v / (K2_m * card_M)
    a_v_m__i = - grad_f_star_v + alpha_v_m * f_
    b_v_m__i = - grad_theta_star_v + alpha_v_m * theta_
    c_v_m__i = v_m_ - 2 * (f_.conjugate().T @ grad_f_star_v + theta_.conjugate().T @ grad_theta_star_v).real + 2 * alpha_v_m

    return (alpha_u_m, a_u_m__i, b_u_m__i, c_u_m__i.real), (alpha_v_m, a_v_m__i, b_v_m__i, c_v_m__i.real)


def CVX_solve(I, alpha_u, alpha_v, a_u, a_v, b_u, b_v, c_u, c_v):
    # Define variables
    n = I # Dimension of lambda_u and lambda_v
    lambda_u = cp.Variable(n, nonneg=True)
    lambda_v = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    p = cp.Variable()
    q = cp.Variable()

    # Define parameters (replace these with your actual values)
    c_i_M = cp.sum(c_u[0,:] * lambda_u + c_v[0,:] * lambda_v)  # Replace with the actual function c_i_M(lambda_u, lambda_v)
    alpha_i_M = cp.sum(alpha_u[0, :] * lambda_u + alpha_v[0, :] * lambda_v) # Replace with the actual function alpha_i_M(lambda_u, lambda_v)
    a_i_M = a_u @ lambda_u + a_v @ lambda_v # Replace with the actual function a_i_M(lambda_u, lambda_v)
    b_i_M = b_u @ lambda_u + b_v @ lambda_v # Replace with the actual function b_i_M(lambda_u, lambda_v)
    a_i_M_norm = cp.norm(a_i_M)

    # Define the objective function
    objective = - c_i_M + 2 * p + q

    # Define the constraints
    constraints = [
        lambda_u >= 0,
        lambda_v >= 0,
        cp.sum(lambda_u + lambda_v) == 1,
        t == alpha_i_M,
        a_i_M_norm <= p,
        cp.norm(cp.hstack([2 * b_i_M, t - q]), 2) <= t + q
    ]

    # Formulate the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # Solve the problem
    problem.solve()
    # problem.solve(solver=cp.SCS)  
    # Access the optimal values
    optimal_lambda_u = lambda_u.value
    optimal_lambda_v = lambda_v.value
    optimal_t = t.value
    optimal_p = p.value
    optimal_q = q.value
    return optimal_lambda_u[:, None], optimal_lambda_v[:, None]

def calculate_dual_f_theta(ld_u, ld_v, I, alpha_u, alpha_v, a_u, a_v, b_u, b_v, c_u, c_v, sigma_d2, P_A):
    f = a_u @ ld_u + a_v @ ld_v
    f = f / np.sqrt(np.sum(np.abs(f) ** 2))
    b_M = b_u @ ld_u + b_v @ ld_v
    alpha_M = alpha_u @ ld_u + alpha_v @ ld_v
    mu = max(np.sum(b_M ** 2) * np.sqrt(sigma_d2 / P_A) - alpha_M, 0)
    theta = b_M / (alpha_M + mu)
    return f[:, 0], theta[:, 0]

def random_initialize_theta(libopt, L):
    theta = np.zeros(L, dtype=np.complex128)
    th = np.random.uniform(0, 2 * np.pi, L)
    a = np.random.uniform(1, 5, L)
    sigma_d2 = libopt.sigma * libopt.transmitpower
    P_A = libopt.transmitpower_A
    delta_A = P_A / 100
    Upper = (P_A - delta_A) / sigma_d2
    if np.sum(a ** 2) > Upper:
        a = a / np.linalg.norm(a) * np.sqrt(Upper / 2)
    return a * np.exp(1j * th)

def sca_fmincon_active(libopt,h_d,h_UR,h_RB,G,f,theta,x,K2,RISON):
    N=libopt.N
    L=libopt.L
    I=sum(x)
    tau=libopt.tau
    if theta is None:
        theta=random_initialize_theta(libopt, L)
    if not RISON:
        theta=np.zeros([L],dtype=complex)
    result=np.zeros(libopt.nit)
    h=np.zeros([N,I],dtype=complex)
    for i in range(I):
        h[:,i] = h_d[:,i]+G[:,:,i] @ theta
        
    if f is None:
        f = h[:,0] / np.linalg.norm(h[:,0])
   
    obj=min(np.abs(np.conjugate(f)@h)**2/K2)
    threshold=libopt.threshold

    
    for it in range(libopt.nit):
        obj_pre=copy.deepcopy(obj)

        # a=np.zeros([N,I],dtype=complex)
        # b=np.zeros([L,I],dtype=complex)
        # c=np.zeros([1,I],dtype=complex)
        # F_cro=np.outer(f,np.conjugate(f))
        
        alpha_u = np.zeros([1, I], dtype=np.float64)
        a_u = np.zeros([N, I],  dtype=complex)
        b_u = np.zeros([L, I],  dtype=complex)
        c_u = np.zeros([1,I],dtype=np.float64)

        alpha_v = np.zeros([1, I], dtype=np.float64)
        a_v = np.zeros([N, I],  dtype=complex)
        b_v = np.zeros([L, I],  dtype=complex)
        c_v = np.zeros([1,I],dtype=np.float64)

        for i in range(I):
            (alpha_u_m, a_u_m__i, b_u_m__i, c_u_m__i), (alpha_v_m, a_v_m__i, b_v_m__i, c_v_m__i) = get_a_b_c_uv_m(libopt, h_m=h[:, i], 
                                                                                                                  h_UR_m=h_UR[:, i], 
                                                                                                                  h_RB=h_RB, 
                                                                                                                  h_d_m=h_d[:, i], 
                                                                                                                  f=f, 
                                                                                                                  theta=theta, 
                                                                                                                  x=x, 
                                                                                                                  K2_m=K2[i])
            alpha_u[0, i] = alpha_u_m
            a_u[:, i:i+1] = a_u_m__i
            b_u[:, i:i+1] = b_u_m__i
            c_u[0, i] = c_u_m__i 
            alpha_v[0, i] = alpha_v_m
            a_v[:, i:i+1] = a_v_m__i
            b_v[:, i:i+1] = b_v_m__i
            c_v[0, i] = c_v_m__i

            # if RISON:
            #     b[:,i]=tau*K2[i]*theta+G[:,:,i].conj().T@F_cro@h[:,i]
            #     c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]*(L+1)+2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])
            # else:
            #     c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]
        
        # fun=lambda mu: np.real(2*np.linalg.norm(a@mu)+2*np.linalg.norm(b@mu,ord=1)-c@mu)
        # cons = ({'type': 'eq', 'fun': lambda mu:  K2@mu-1})
        # bnds=((0,None) for i in range(I))
        # res = minimize(fun, 1/K2,   bounds=tuple(bnds), constraints=cons)
        # if ~res.success:
        #     pass
        #     # print('Iteration: {}, solution:{} obj:{:.6f}'.format(it,res.x,res.fun[0]))
        #     # print(res.message)
        #     # return
        # fn=a@res.x
        # thetan=b@res.x
        # fn=fn/np.linalg.norm(fn)
#        thetan=thetan/np.abs(thetan)
        # if RISON:
        #     thetan=thetan/np.abs(thetan)
        #     theta=thetan
        # f=fn
        sigma_d2 = libopt.sigma * libopt.transmitpower
        P_A = libopt.transmitpower_A
        ld_u, ld_v = CVX_solve(I, alpha_u, alpha_v, a_u, a_v, b_u, b_v, c_u, c_v)
        f, theta = calculate_dual_f_theta(ld_u, ld_v, I, alpha_u, alpha_v, a_u, a_v, b_u, b_v, c_u, c_v, sigma_d2, P_A)
        for i in range(I):
            h[:,i] = h_d[:,i] + G[:,:,i] @ theta
        obj = min(np.abs(np.conjugate(f) @ h) ** 2 / K2)
        result[it] = copy.deepcopy(obj)
        if libopt.verbose > 2:
            print('  Iteration {} Obj {:.6f} Opt Obj {:.6f}'.format(it,result[it],res.fun[0]))
        if np.abs(obj - obj_pre) / min(1, abs(obj)) <= threshold:
            break
        
        #print(res)
    if libopt.verbose > 1:
        print(' SCA Take {} iterations with final obj {:.6f}'.format(it+1,result[it]))
    result=result[0:it]
    return f,theta,result



def find_obj_inner_active(libopt, x, K, K2, Ksum2, h_UR, h_RB, h_d, G, f0, theta0, RISON):
    N=libopt.N
    L=libopt.L
    M=libopt.M
    if sum(x)==0:
        obj=np.inf
        
        theta=np.ones([L],dtype=complex)
        f=h_d[:,0]/np.linalg.norm(h_d[:,0])
        if not RISON:
            theta=np.zeros([L])
    else:
        index=(x==1)
         #print(index)

        # f, theta, _ = sca_fmincon_active(libopt,h_d[:,index],G[:,:,index],f0,theta0,x,K2[index],RISON)

        f, theta, _ = sca_fmincon_active(libopt,h_d[:, index],h_UR[:, index],h_RB, G[:, :, index], f0, theta0, x, K2[index], RISON)
        h = np.zeros([N,M],dtype=complex)
        for i in range(M):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta

        P_T = libopt.transmitpower
        P_A = libopt.transmitpower_A
        power_d = libopt.sigma*libopt.transmitpower
        power_s = libopt.sigma*libopt.transmitpower

        inner=f.conj()@h[:,index]
        inner2=np.abs(inner)**2
        total_M = np.sum(index)




        Theta = np.diag(theta)
        Theta_h_Rm_norm2_m = np.sum(np.abs(Theta @ h_UR[:,index]) ** 2, axis=0)
        h_R_m = np.sum(np.abs(h_UR[:,index]) ** 2, axis = 0)
        
        max_1 = np.max(K2/(P_T*inner2))
        max_2 = np.max(total_M * K2 * (Theta_h_Rm_norm2_m-h_R_m)/(inner2*P_A-power_d*(np.sum(np.abs(theta) ** 2))))

        max_item = np.max([max_1,max_2])

        f_H_Theta = np.sum(np.abs(f.conj()@h_RB@Theta)**2)
        pre_max = (power_d*f_H_Theta+power_s)/((np.sum(K[index])) ** 2)


        obj= pre_max * max_item + 4 / Ksum2 * (sum(K[~index])) ** 2
    return obj,x,f,theta



# def Gibbs_active(libopt,h_d,h_UR,h_RB,G,x0,DS):

    
#     N=libopt.N
#     L=libopt.L
#     M=libopt.M
#     Jmax=libopt.Jmax
#     K=libopt.K/np.mean(libopt.K) #normalize K to speed up floating computation
#     K2=K**2
#     Ksum2=sum(K)**2
#     x=x0
#     # inital the return values
#     obj_new=np.zeros(Jmax+1)
#     f_store=np.zeros([N,Jmax+1],dtype = complex)
#     theta_store=np.zeros([L,Jmax+1],dtype = complex)
#     x_store=np.zeros([Jmax+1,M],dtype=int)


#     return x_store,obj_new,f_store,theta_store


def Gibbs_active(libopt,h_d,h_UR,h_RB,G,x0,DS):
    #initial
    # theta shape [L]
    # f shape [N]
    # h_d shape [N,M]
    # h_UR [L,M]
    # h_RB [N,L]
    # G [N, L, M]
    # 
    #initial
    N=libopt.N
    L=libopt.L
    M=libopt.M
    Jmax=libopt.Jmax
    K=libopt.K/np.mean(libopt.K) #normalize K to speed up floating computation
    K2=K**2
    Ksum2=sum(K)**2
    x=x0
    # inital the return values
    obj_new=np.zeros(Jmax+1)
    f_store=np.zeros([N,Jmax+1],dtype = complex)
    theta_store=np.zeros([L,Jmax+1],dtype = complex)
    x_store=np.zeros([Jmax+1,M],dtype=int)
    
    #the first loop
    ind=0
    RISON = True
    [obj_new[ind],x_store[ind,:],f,theta]=find_obj_inner_active(libopt,x,K,K2,Ksum2,h_UR, h_RB, h_d,G,None,None,RISON)
    if DS == False:
        return [x_store,obj_new,f_store,theta_store]
    theta_store[:,ind]=copy.deepcopy(theta)
    f_store[:,ind]=copy.deepcopy(f)
#    beta=min(max(obj_new[ind],1)
    beta=min(1,obj_new[ind])
    # print(beta)
    alpha=0.9
    if libopt.verbose>1:
        print('The inital guess: {}, obj={:.6f}'.format(x,obj_new[ind]))
    elif libopt.verbose==1:
        print('The inital guess obj={:.6f}'.format(obj_new[ind]))
    f_loop=np.tile(f,(M+1,1))

    theta_loop=np.tile(theta,(M+1,1))
    #print(theta_loop.shape)
    #print(theta_loop[0].shape)
    for j in range(Jmax):
        if libopt.verbose>1:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f}'.format(j+1,beta));
        
        #store the possible transition solution and their objectives
        X_sample=np.zeros([M+1,M],dtype=int)
        Temp=np.zeros(M+1)
        #the first transition => no change
        X_sample[0,:]=copy.deepcopy(x)
        Temp[0]=copy.deepcopy(obj_new[ind])
        f_loop[0]=copy.deepcopy(f)
        theta_loop[0]=copy.deepcopy(theta)
        #2--M+1-th trnasition, change only 1 position
        for m in range(M):
            if libopt.verbose>1:
                print('the {}-th:'.format(m+1))
            #filp the m-th position
            x_sam=copy.deepcopy(x)
            x_sam[m]=copy.deepcopy((x_sam[m]+1)%2)
            X_sample[m+1,:]=copy.deepcopy(x_sam);
            Temp[m+1],_,f_loop[m+1],theta_loop[m+1]=find_obj_inner(libopt,
                x_sam,K,K2,Ksum2,h_d,G,f_loop[m+1],theta_loop[m+1],RISON)
            if libopt.verbose>1:
                print('          sol:{} with obj={:.6f}'.format(x_sam,Temp[m+1]))
        temp2=Temp;
        
        Lambda=np.exp(-1*temp2/beta);
        Lambda=Lambda/sum(Lambda);
        while np.isnan(Lambda).any():
            if libopt.verbose>1:
                print('There is NaN, increase beta')
            beta=beta/alpha;
            Lambda=np.exp(-1.*temp2/beta);
            Lambda=Lambda/sum(Lambda);
        
        if libopt.verbose>1:
            print('The obj distribution: {}'.format(temp2))
            print('The Lambda distribution: {}'.format(Lambda))
        kk_prime=np.random.choice(M+1,p=Lambda)
        x=copy.deepcopy(X_sample[kk_prime,:])
        f=copy.deepcopy(f_loop[kk_prime])
        theta=copy.deepcopy(theta_loop[kk_prime])
        ind=ind+1
        obj_new[ind]=copy.deepcopy(Temp[kk_prime])
        x_store[ind,:]=copy.deepcopy(x)
        theta_store[:,ind]=copy.deepcopy(theta)
        f_store[:,ind]=copy.deepcopy(f)
        
        if libopt.verbose>1:
            print('Choose the solution {}, with objective {:.6f}'.format(x,obj_new[ind]))
            
        if libopt.verbose:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}'.format(j+1,beta,obj_new[ind]))
        beta=max(alpha*beta,1e-4)
                
    return x_store,obj_new,f_store,theta_store