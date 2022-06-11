#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tanja Zerenner
# Year: 2021
# email: t.zerenner@gmail.com

from scipy.integrate import odeint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math as math
import os
from fastgillespie import *
from Inference import *
from MCMC import *
from BD_functions import *
from PRE_functions import *

'''
******************************************************************************
***
** SET PARAMETERS
***
******************************************************************************
'''

networkclasses = ['ER', 'REG', 'BA']

'''
Select a case (network & epidemic parameters) from the paper; see https://doi.org/10.1016/j.mbs.2022.108854
'''
case = 2
[N, k, c, tau, gamma, tauf] = get_case_params(case)
networkchoice = networkclasses[c]

'''
OR: Choose network and epidemic parameters by hand 
'''
'''
networkchoice = 'ER' # select from networkclasses
N = 1000 # population size
k = 16  # approx. avaerage node degree
tau = 0.859 # per-link infection rate
gamma = 6.338 # recovery rate
tauf = 2.5 # simulation time
'''


'''
Initialization, observation & sample size parameters
'''
i0 = 5 # number of infected nodes at time 0

nobs = 10 # number of observations for inference, will be approx equidistant in time between obs_1 and obs_n
obs_1 = 50 # number of infected individuals at first observation 
obs_n = 320 # number of infected individuals at last observation

E = 10 # number of Gillespie simulations initialised at last observation (serves as reference to evaluate the predictions)

N_MC_0 = 500 # sample size to be drawn from network class specific posteriors ~500 is a reasonable value

thin_sample = True # thin sample before applying pushforward (to reduce computational cost)
ke = 50 # keep every ke's draw when thinning

'''
Select which steps to run
'''
# if any of these is set to False, data from the previous step have to be available in 'myfolder' (see below)
generate_data = False # generate observations using Gillespie algorithm
generate_reference = False# restart Gillespie simulations at last observation to generate a reference to assess predictive performance
inference = False # infer posteriors and draw samples (computationally expensive when sample is large)
generate_predictions = True # apply Pushforward to samples (computationally expensive when sample is large)
process_predictions_and_plot = True #load/generate predictions from Pushforward of sample and plot


'''
Specify folder to save output
'''
myfolder = '../example/'



myrandseed = 29058 # seed



'''
******************************************************************************
***
** RUN
***
******************************************************************************
'''

    
'''
STEP 1: Generate the observational data
'''

if generate_data == True:
    discretestep = 5000 # has to be large enough to extract discrete observations as specified above
    print('1. Generating observational data')
    [obsdata, A, rstate] = GenerateObs(networkchoice, N, k, tau, gamma, int(i0), tauf, discretestep, obs_1, nobs, obs_n , plot=1)
    np.savetxt(myfolder + "observations.csv", obsdata, delimiter=",")
    nx.write_adjlist(A, myfolder + "network.adjlist.gz")
    np.savetxt(myfolder + "restartstate.csv", rstate, delimiter=",")
else:
    obsdata=np.loadtxt(myfolder + "observations.csv", delimiter=",")
    rstate=np.loadtxt(myfolder + "reference.csv", delimiter=",")
    A = nx.read_adjlist(myfolder + "network.adjlist.gz")
    
print('Network is connected:', nx.is_connected(A))


'''
STEP 2: Generate Reference
'''
if generate_reference == True:
    discretestep = 100
    print('2. Generating reference (restarting Gillespie at last observation)')
    simdata = GenerateRef(N, A, tau, gamma, rstate, tauf, discretestep, obsdata, E, plot=1)
    np.savetxt(myfolder + "reference.csv", simdata, delimiter=",")
else:
    simdata=np.loadtxt(myfolder + "reference.csv", delimiter=",")


'''
STEP 3: Inference
'''

if inference == True:
    
    print('3. Infering network class and drawing samples.')
    
    Classify_and_Sample(obsdata, N, gamma, N_MC_0, myfolder, seed = myrandseed)


'''
STEP 4: Predictions
'''

if generate_predictions == True:
    
    classprobs = np.loadtxt(myfolder + "PosteriorClass.csv", delimiter=",")
    iMAP = np.argmax(classprobs, axis=0) # maximum index of netqwork class          

    #Type 1: Apply pushforward to MAP (C,a,p) of most likely network class
    print('4a. Applying Pushfoward to MAP')   
    v = np.loadtxt(myfolder + networkclasses[iMAP] + "_Posterior_v_MAP.csv", delimiter=",") 
    v = np.expand_dims(v, axis=0)
    yy = Pushforward(v,int(obsdata[-1,1] ),tauf,len(simdata),N,gamma,iMAP)
    m_map = yy[:,:,0]
    
    np.savetxt(myfolder + networkclasses[iMAP] + "_PredMAP.csv", m_map, fmt='%.8e', delimiter=",")
    
     
    #Type2: Apply pushforward to samples of (C,a,p)
    print('4b. Applying Pushfoward to samples')
    for i in range(0,3): 
        sample = np.loadtxt(myfolder + networkclasses[i] + "_PosteriorSample.csv", delimiter=",")
        
        if thin_sample == True:
            cms,tsample = post_results(sample, 0, ke)
            #https://github.com/Gabriel-p/multiESS
            #sampleESS = multiESS(sample2, b='less', Noffsets=10, Nb=None)  
            #print('ESS=',sampleESS)
        else:
            tsample = sample
        
        yy = Pushforward(tsample,int(obsdata[-1,1]),tauf,len(simdata),N,gamma,i)
        yy2d = np.reshape(yy,(len(simdata),(N+1)*len(tsample)))
        
        np.savetxt(myfolder + networkclasses[i] + "_PredSample.csv", yy2d, fmt='%.8e', delimiter=",")
       

'''
STEP 5: Process predictions and
'''
    
if process_predictions_and_plot == True:
    
    print('5. Processing predictions...')   
    
    classprobs = np.loadtxt(myfolder + "PosteriorClass.csv", delimiter=",")
    iMAP = np.argmax(classprobs, axis=0) # maximum index of netqwork class         
    
    # Load MAP-base prediction
    m_map = np.loadtxt(myfolder + networkclasses[iMAP] + "_PredMAP.csv", delimiter=",") 
      
    # Generate CM-base predictions from Pushforward of sample
    m_sample = np.zeros(shape=(len(simdata),N+1))         
    for i in range(0,3):
         yy = np.loadtxt(myfolder + networkclasses[i] + "_PredSample.csv", delimiter=",") 
         yy_sample =  np.reshape(yy, (len(simdata), N+1, int(len(yy[0,:])/(N+1)) )) # dimension: time * state * no of draws
         del yy   
         m_sample = m_sample + classprobs[i] * np.mean(yy_sample,axis=2)

    # Get expectations (avg) and quantiles (kq) from predictions
    q = [0.05,0.15,0.85,0.95]
                  
    [avgk_CM,kq_CM] = Quantiles(N,q,obs_n,m_sample)
    [avgk_MAP,kq_MAP] = Quantiles(N,q,obs_n,m_map)
    
    
    
    print('...and plotting.')    
   
    mycolors = ['#377eb8','#e41a1c','#4daf4a']
    mycolor =  mycolors[c]
    
    # Plot predictions of expected number of infected nodes
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))       
    plt.setp(ax, xlabel = '$t$', ylabel="$I$", xlim=[0,tauf], ylim=[0,N])
    plt.plot(simdata[:,0]+obsdata[-1,0],simdata[:,1],linewidth=0.6, color=mycolor,label='reference')
    for i in range(2,E):
        plt.plot(simdata[:,0]+obsdata[-1,0],simdata[:,2*i-1],linewidth=0.6, color=mycolor)
    plt.plot(obsdata[:,0], obsdata[:,1], '.', color=mycolor, label='observations')
    plt.plot(simdata[:,0]+obsdata[-1,0], avgk_CM,linewidth=1, color='black', label='CM')
    plt.plot(simdata[:,0]+obsdata[-1,0], avgk_MAP,linewidth=1.2, color='black', linestyle='dashed', label='MAP')
    ax.legend(frameon=False,ncol=2)
    
    plt.tight_layout()
    plt.savefig(myfolder+'predictions_expectations.png', dpi=400)
    
    
    # Plot predictions as quantiles
    fig, ax = plt.subplots(1, 2,  figsize=(7.5,3), sharex=True, sharey=True)
    plt.setp(ax, xlim = [0,tauf], ylim = [0,N], xlabel="$t$", ylabel='$I$')
    
    FanCol = "grey"
    mylwd = 0.6
    alphav = [0.2,0.4]
    
    reference = simdata
    
    t = np.linspace(obsdata[-1,0], tauf+obsdata[-1,0], len(kq_CM)-1) # tauf
    
    # MAP-based predictions
    for i in range(0,int(len(q)/2)):
        alpha=alphav[i]
        ax[0].fill_between(t[1:len(t)], kq_MAP[1:len(t),i], kq_MAP[1:len(t),len(q)-1-i], color=FanCol, alpha=alpha, linewidth=0)
    
    ax[0].plot(obsdata[:,0],obsdata[:,1],'.', color=mycolor, markersize = 3)    
       
    ii = 1
    for i in range(1,E):
        ax[0].plot(reference[:,0]+obsdata[-1,0], reference[:,ii], color=mycolor, linewidth = 0.5, alpha = 1)
        ii = ii+2
    ax[0].text(0+tauf/20, 1050,"MAP-based prediction", fontsize = 12)   
    
    
    # CM-based predictions
    for i in range(0,int(len(q)/2)):
        alpha=alphav[i]
        ax[1].fill_between(t[1:len(t)], kq_CM[1:len(t),i], kq_CM[1:len(t),len(q)-1-i], color=FanCol, alpha=alpha, linewidth=0)
    
    ax[1].plot(obsdata[:,0],obsdata[:,1],'.', color=mycolor, markersize = 3)   
       
    ii = 1
    for i in range(1,E):
        ax[1].plot(reference[:,0]+obsdata[-1,0], reference[:,ii], color=mycolor, linewidth = 0.5, alpha = 1)
        ii = ii+2
    ax[1].text(0+tauf/20, 1050,"CM-based prediction", fontsize = 12)   
    
    
    plt.tight_layout()
    
    plt.savefig(myfolder+'predictions_quantiles.png', dpi=400)
