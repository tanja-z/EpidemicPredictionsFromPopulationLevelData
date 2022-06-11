# Author: Tanja Zerenner
# Year: 2021
# email: t.zerenner@gmail.com
# This code contains functions for running the prediction experiments

import numpy as np
import math as math
import networkx as nx
import pickle
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from fastgillespie import *
from BD_functions import *


def GenerateObs(networkchoice, N, k, tau, gamma, i0, tauf, discretestep, obs_1, nobs, obs_n , plot=1):
    
    ## STEP 1: Simulate the epidemic on the network ##
    
    # Generate the network
    if networkchoice == 'ER':
        A = nx.fast_gnp_random_graph(N,k/float(N-1.0)) # nodes, prob of each possible edge (degreee not exactly k)
    elif networkchoice=='REG':
        A = nx.random_regular_graph(k,N) # random regular graph where k is the degree of each node (exact)
    elif networkchoice=='BA':
        A = nx.barabasi_albert_graph(N,int(k/2)) # mean degree not exactly k (first k nodes in graph generation connected to less than k nodes)
    
    # Simulate the epidemic
    model = fast_Gillespie(A, tau =tau, gamma =gamma, i0 =i0, tauf =tauf, discretestep =discretestep, restart =0)
    model.run_sim()
    
    data = np.zeros((discretestep,2))
    data[:,0] = model.time_grid
    data[:,1] = model.I


    ## STEP 2: Extract observations ##
    
    sample_from = np.argmax(data[:,1]>=obs_1)           
    sample_to = np.argmax(data[:,1]>=obs_n)
    ndata = data[sample_from:sample_to,:]
    res = ndata[::-1] 
    sample_every = math.floor((sample_to-sample_from-1)/(nobs-1))
    res_sample = res[::sample_every]
    obsdata = res_sample[::-1] 
    
    rstate = model.state[:,sample_to-1] #sum(rstate) should be equal to last entry in obsdata
    
    if plot==1:
        # Plot the number of infected nodes I(t) vs. time t
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plt.setp(ax, xlabel = '$t$', ylabel="$I(t)$", xlim=[0,tauf], ylim=[0,N])
        ax.plot(data[:,0], data[:,1], color='grey',linewidth=1)
        ax.plot(obsdata[:,0], obsdata[:,1], '.', color='darkorange')
        plt.show

    return obsdata, A, rstate





def GenerateRef(N, A, tau, gamma, rstate, tauf, discretestep, obsdata, E=10, plot=1):

    model = fast_Gillespie(A, tau =tau, gamma =gamma, i0 =rstate, tauf =tauf-obsdata[-1,0], discretestep =discretestep, restart=1)
    model.run_sim()

    simdata = np.zeros((discretestep,2*E))
    
    e=0
    ee = 0
    while e < E:
       # print(e)
        model = fast_Gillespie(A, tau =tau, gamma =gamma, i0 =rstate, tauf =tauf-obsdata[-1,0], discretestep =discretestep, restart=1)
        model.run_sim()
        simdata[:,ee] = model.time_grid
        simdata[:,ee+1] = model.I
        e = e+1
        ee = ee+2
    
    if plot==1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))       #
        plt.setp(ax, xlabel = '$t$', ylabel="$I(t)$", xlim=[0,tauf], ylim=[0,N])
        for i in range(1,E):
            plt.plot(simdata[:,0]+obsdata[-1,0],simdata[:,2*i-1],linewidth=0.5, color='gray')
        plt.plot(obsdata[:,0], obsdata[:,1], '.', color='darkorange')#,linewidth=1)
        plt.show()
        
    return simdata




def Diffeq(v,t, gam,a):
    c=[i for i in range(0,len(v))]
    vdot = np.empty_like(v)

    for i in range(0,len(v)):
        if i == 0:
            vdot[0] = gam*c[1]*v[1]
        elif i == (len(v)-1):
            vdot[i] =  a[i-1]*v[i-1] - (a[i]+gam*c[i])*v[i]
        else:
            vdot[i] =  a[i-1]*v[i-1] - (a[i]+gam*c[i])*v[i]+ gam*c[i+1]*v[i+1]
    return vdot
    



def Pushforward(v,i0,tauf,discretestep,N,gamma,i):
    
    kde,mean,chol,chol_inv = get_priors(i)
    
    t = np.linspace(0,tauf,discretestep)
    y0 = np.zeros(shape=(N+1,)) 
    y0[i0] = 1
    
    u = v_to_u(v[:,1:4], mean ,chol)

    yy = np.zeros(shape=(discretestep,N+1,len(v)))    
    
    for jj in range(0,len(v)):
        print(jj)
        ak = model_cap_N(u[jj,:], np.arange(0,N+1), N)
        yy[:,:,jj] = odeint(Diffeq,y0,t,args=(gamma,ak))

    return yy 



def Quantiles(N,q,obs_n,m_sample): # rename m_sample!

    K = np.linspace(0,N,N+1)
    avgk = np.sum(m_sample * K,1)
    cdf = np.cumsum(m_sample,1) # size: (t,N+1)
    kq = np.zeros(shape=(len(m_sample),len(q)))
        
    for t_cdf in range(1, len(m_sample)):
        for iq in range(0, len(q)):
            # find index (correponds to k) of quantile q_i
            l=abs(cdf[t_cdf,:]-q[iq])
            lmin=np.min(l)
            min_index = np.where(l == lmin)
            #kq = t, nq
            kq[t_cdf,iq]=min_index[0]
    
    kq[0,:] =  obs_n
    kq = np.vstack((q, kq))
    
    return avgk,kq



def get_priors(i): 
    
    with open('Priors.pickle_N_SI_all', 'rb') as f:
        Priors = pickle.load(f)

    print("Prior ", i)
    if (i==0):
        kde = Priors["kde_ER"]
        mean = Priors["mean_ER"]
        chol = Priors["chol_ER"]
        chol_inv = Priors["chol_inv_ER"]
    elif (i==1):
        kde = Priors["kde_Reg"]
        mean = Priors["mean_Reg"]
        chol = Priors["chol_Reg"]
        chol_inv = Priors["chol_inv_Reg"]
    else:
        kde = Priors["kde_BA"]
        mean = Priors["mean_BA"]
        chol = Priors["chol_BA"]
        chol_inv = Priors["chol_inv_BA"]
            
    return kde,mean,chol,chol_inv




def get_case_params(case):
             
        if case == 1:
            # CASE 1 (E-R large epidemic)
            c = 0 # E-R
            N = 1000
            k = 8
            tau = 1.251
            gamma = 0.969
            tauf = 2.5
            
        elif case == 2:
            # CASE 2 (E-R medium epidemic)
            c = 0 # E-R
            N = 1000
            k = 16
            tau = 0.859
            gamma = 6.338
            tauf = 2.5
        
        elif case == 3:
            # CASE 3 (E-R small epidemic)
            c = 0 # E-R
            N = 1000
            k = 12
            tau = 1.143
            gamma = 9.579
            tauf = 4
        
        elif case == 4:
            # CASE 4 (Reg large epidemic)
            c = 1 
            N = 1000
            k = 5
            tau = 4.251
            gamma = 2.969
            tauf = 1.5
            
        elif case == 5:
            # CASE 5 (Reg medium epidemic)
            c = 1 
            N = 1000
            k = 10
            tau = 1.265
            gamma = 5.773
            tauf = 3
            
        elif case == 6:
            # CASE 6 (Reg small epidemic)
            c = 1 
            N = 1000
            k = 7
            tau = 0.762
            gamma = 3.356
            tauf = 16
        
        elif case == 7:
            # CASE 7 (B-A large epidemic)
            c = 2 
            N = 1000
            k = 15
            tau = 3.123
            gamma = 6.969
            tauf = 0.5
      
        elif case == 8:
            # CASE 8 (B-A medium epidemic)
            c = 2 
            N = 1000
            k = 11
            tau = 2.190
            gamma = 8.948
            tauf = 1.0
    
        elif case == 9:    
            # CASE 9 (B-A small epidemic)
            c = 2
            N = 1000
            k = 8
            tau = 0.612
            gamma = 3.803
            tauf = 5
            
        return N, k, c, tau, gamma, tauf



