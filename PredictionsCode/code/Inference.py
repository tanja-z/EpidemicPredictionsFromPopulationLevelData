# Author: Tanja Zerenner
# Year: 2021
# email: t.zerenner@gmail.com
# This code infers the posterior(s)
# It is based on the function 3_Classify_Graphs.py by
# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk
# https://github.com/BayIAnet/NetworkInferenceFromPopulationLevelData

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numdifftools as nd
from BD_functions import *
from pyswarm import pso
from MCMC import *
import pickle
import sys
import math

def Classify_and_Sample(y, N, gamma, N_MC_0, savepath, seed):

    np.random.seed(seed)
    networkclasses = [ 'ER', 'REG', 'BA']
    
    N_MC_1 = 3000 # MCMC steps
    # Read the kde estimators
    with open('Priors.pickle_N_SI_all', 'rb') as f:
        Priors = pickle.load(f)

    MCT = np.zeros((3, 3000))
    res = np.zeros((3, 3000))
    
    lb = [-5, -5, -5]
    ub = [5, 5, 5]

    MAP_temp = np.array([])

    for i in range(3):
        # For each prior
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

        # Negative log-integrand in the u-space
        def pen_log_like_u(u):
            u = u[np.newaxis, :]
            res = -log_like(u[0, :], y, gamma, N)
            res -= kde.score(u_to_v(u, mean, chol_inv))
            return  res
        
        # Negative log-integrand in the v-space
        def pen_log_like_v(v):
            v = v[np.newaxis, :]
            res = -log_like(v_to_u(v, mean, chol)[0, :], y, gamma, N)
            res -= kde.score(v) 
            return  res # prob
        
        # MAP point in the v-coordinates and its Hessian
        min_v, min_fv = pso(pen_log_like_v, lb=lb, ub=ub, maxiter=5, swarmsize=150)
        v_max = minimize(pen_log_like_v, min_v, method="Nelder-Mead")
        v_MAP = v_max.x
        u_MAP = v_to_u(v_max.x[np.newaxis, :], mean, chol)[0, :]
        MAP_temp = np.hstack((MAP_temp, u_MAP)) #(C*,a*,p*)

        # Computing the Hessian at MAP location
        Hess = nd.Hessian(pen_log_like_v, step=1e-3, method="central")
        neg_Hess = Hess(v_MAP)

        if (np.linalg.det(neg_Hess)<=0):
            Hess = nd.Hessian(pen_log_like_v, step=1e-2, method="central")
            neg_Hess = Hess(v_MAP)

        Cov_max = np.linalg.inv(neg_Hess)

        # MH sampling of posterior distribution
        MCMC_RW = MC(negll = pen_log_like_v, Cov = Cov_max)
        proba, results = MCMC_RW.MCMC(v_MAP, h=0.6, n=N_MC_0) 

        v_MAP_P = np.insert(v_MAP, 0, v_max.fun)
        
        np.savetxt(savepath + networkclasses[i] + "_Posterior_v_MAP.csv", v_MAP_P, delimiter=",")
        np.savetxt(savepath + networkclasses[i] + "_PosteriorSample.csv", results, delimiter=",")
              
     
        def dist1(v, threshold):
            # Compute maximum distance
            IndA = np.zeros(v.shape[0])
            vc = v-v_MAP
            for (i, vi) in enumerate(vc):
                IndA[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi))<=threshold
            return IndA

        def dist2(v):
            # Compute Ind_A
            dist = np.zeros(v.shape[0])
            vc = v-v_MAP
            for (i, vi) in enumerate(vc):
                dist[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi)) #3. in paper
            return dist

        # Region A with P(A|y)=1
        idx = np.where(np.abs(results[:, 0]-results[0, 0])<=3)[0] #Why 3? close to true gamma
        temp = dist2(results[idx, 1:4])
        threshold = temp.max() #r

        # Assessing P(A)
        sample1 = multivariate_normal.rvs(mean=v_MAP, cov=Cov_max, size=10000)
        idx1 = dist1(sample1, threshold)
        temp = kde.score_samples(sample1)-multivariate_normal.logpdf(sample1, mean=v_MAP, cov=Cov_max) #kde.score - mvar_normal.logpdf (???) log of prob(C,a,p) sample; vector!
        PA = np.mean(idx1*np.exp(temp))
        print("PA = ", PA) # prior(A)

        # CAME estimator with importance sampling
        sample = multivariate_normal.rvs(mean=v_MAP, cov=5*Cov_max, size=5*N_MC_1) #Why 5? (???)
        IndA = dist1(sample, threshold)
        idx = np.where(IndA==1)[0]
        
        def newMCT(v):
            res = -pen_log_like_v(v)
            res -= multivariate_normal.logpdf(v, mean=v_MAP, cov=5*Cov_max)
            return res
        
        for j in range(np.minimum(3000, idx.size)):
            MCT[i, j] = newMCT(sample[idx[j], :])

        res[i, :] = np.log(PA*np.cumsum(np.exp(MCT[i, :]-MCT[i, :].max())) / np.arange(1, 3000+1))+MCT[i, :].max() #log of lhs of (7) in paper

    np.savetxt(savepath + "MCMC_0.csv", res, delimiter=",")
    np.savetxt(savepath + "MAPs_0.csv", MAP_temp)

    classprobs = 1/np.sum(np.exp(res[:,-1])) * np.exp(res[:,-1])   
    np.savetxt(savepath + "PosteriorClass.csv", classprobs, delimiter=",")
    
