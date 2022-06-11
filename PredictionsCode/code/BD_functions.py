# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk
# This code provides functions that are useful for the classification
# See https://github.com/BayIAnet/NetworkInferenceFromPopulationLevelData

import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy.optimize import minimize, root
from scipy.special import binom
from scipy.linalg import pascal
from pyswarm import pso
from mpmath import nsum, inf
from matplotlib import pyplot as plt

threshold = 1e-12

def u_to_v(u, mean, chol_inv):
    v = u.copy()
    v[:, 0] = np.log(v[:, 0])
    v = np.matmul(v - mean, chol_inv.transpose())
    return v


def v_to_u(v, mean, chol):
    u = v.copy()
    u = np.matmul(u, chol.transpose()) + mean
    u[:, 0] = np.exp(u[:, 0])
    return u

def model_cap_N(u, k, N):
    # Return ak curve given u
    return u[0] * k**u[2] * (N-k)**u[2] * (u[1]*(k-N/2)+N)

def log_like_exp(u, Data, gamma, N):
    # Compute discrete log_likelihood
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap_N(u, k, N)
    ck = gamma * np.arange(N+1)
    Q = Q_assembly(ak, ck)
    log_like = 0
    for i in range(1, Data.shape[0]):
        proba = p_k0k_exp(Data[i, 0]-Data[i-1,0], Data[i, 1], Data[i-1, 1], Q)
        log_like += np.log(proba)
    return log_like


def log_like_levin(u, Data, gamma, N):
    # Compute discrete log_likelihood
    # Using Crawford method for Laplace Inversion
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap_N(u, k, N)
    ck = gamma * np.arange(N+1)
    log_like = 0
    for i in range(1, Data.shape[0]):
        proba = p_k0k_vec_levin(Data[i, 0]-Data[i-1,0], Data[i, 1], Data[i-1, 1], ak, ck)
        log_like += np.log(proba)
    return log_like


def log_like(u, Data, gamma, N):
    # Compute discrete log_likelihood
    # Using Crawford method for Laplace Inversion
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap_N(u, k, N)
    ck = gamma * np.arange(N+1)
    log_like = 0
    for i in range(1, Data.shape[0]):
        proba = p_k0k_vec(Data[i, 0]-Data[i-1,0], Data[i, 1], Data[i-1, 1], ak, ck)
        log_like += np.log(proba)
    return log_like


def log_like_old(u, Data, gamma, N):
    # Compute discrete log_likelihood
    # Using Crawford method for Laplace Inversion
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap_N(u, k, N)
    ck = gamma * np.arange(N+1)
    log_like = 0
    for i in range(1, Data.shape[0]):
        proba = p_k0k_vec_old(Data[i, 0]-Data[i-1,0], Data[i, 1], Data[i-1, 1], ak, ck)
        log_like += np.log(proba)
    return log_like


def p_k0k_exp(t, k, k0, Q):
    # Compute P(I(t)=k|I(0)=k0,theta) by matrix exponential
    return np.maximum(expm(Q*t)[int(k0), int(k)], threshold)


def cont_frac_vec(a, b, eps=1e-30):
    # Compute the continued fraction (Modified Lentz's method)
    # a0/(b0+)a1/(b1+)a2/(b2+)
    D = np.zeros(b.shape[0], dtype=complex)
    frac = eps * np.ones(b.shape[0], dtype=complex)
    C = frac
    for i in range(a.size):
        D = 1 / (b[:, i]+a[i]*D)
        C = b[:, i] + a[i]/C
        frac *= C * D
    return frac


def f_vec(s, k, k0, ak, ck):
    # Compute the Laplace transform of P(I(t)=k|I(0)=k0,theta).
    # s is an array of complex values

    log_cte = 0
    if (k<k0):
        log_cte = np.sum(np.log(ck[(k+1):(k0+1)]))
    elif (k>k0):
        log_cte = np.sum(np.log(ak[k0:k]))

    idx1 = np.minimum(k, k0)
    idx2 = np.maximum(k, k0)

    # Compute B_idx1(s), B_{idx2}(s), B_{idx2+1}(s)
    a = np.ones(ak.size) #a_1,  a_2, ..., a_N
    b = np.ones((s.size, ak.size), dtype=complex)
    a[1:] = -ak[0:-1] * ck[1:]
    b = b * s[:, np.newaxis]
    b = b + ak + ck
    cf = cont_frac_vec(a[(idx2+1):], b[:, (idx2+1):])

    D = np.zeros((s.size, idx2+2), dtype=complex) #D_0, D_1,...,D_{max(k0, k)+1}
    for i in range(idx2+1):
        D[:, i+1] = 1 / (b[:, i]+a[i]*D[:, i])

    # Compute f_{k0, k}(s)
    log_res = np.sum(np.log(D[:, (idx1+1):(idx2+2)]), axis=1)
    res = np.exp(log_res+log_cte) / (1+D[:, idx2+1]*cf)
    return res


def p_k0k_vec(t, k, k0, ak, ck, gamma=4, N=100):
    # Compute P(I(t)=k|I(0)=k0,theta) by Crawford's method
    # Inverse Laplace transform of continued fraction representation
    # Euler's transform
    
    A = gamma * np.log(10)

    # Summing few terms first
    idx = np.arange(0, 20)
    s = (A+2*idx*np.pi*1j) / 2 / t
    val = np.real(f_vec(s, int(k), int(k0), ak, ck))
    val[0] /= 2
    val = val * (-1)**idx
    Sum = np.sum(val)
    
    # Euler acceleration
    idx = np.arange(20, N)
    s = (A+2*idx*np.pi*1j) / 2 / t
    val = np.real(f_vec(s, int(k), int(k0), ak, ck))
    val = val * (-1)**idx
    #plt.plot(val)
    val = np.hstack((-1, np.cumsum(val)))
    for i in range(val.size-1):
        val = (val[:-1]+val[1:]) / 2
    res = np.exp(A/2) / t * (Sum+val[0])
    return np.maximum(res, threshold)


def p_k0k_vec_old(t, k, k0, ak, ck, gamma=4, N=30):
    # Compute P(I(t)=k|I(0)=k0,theta) by Crawford method
    # Inverse Laplace transform of continued fraction representation
    # Euler's transform
    A = gamma*np.log(10)
    idx = np.arange(0, N)
    s = (A+2*idx*np.pi*1j)/2/t
    val = np.real(f_vec(s, int(k), int(k0), ak, ck))
    col = np.ones((N, N), dtype="int")*np.arange(N)
    lig = np.ones((N, N), dtype="int")*np.arange(N)[:, np.newaxis]
    temp = np.triu(np.choose(col-lig, val, mode="clip"))*(-1)**(col+lig)
    temp *= pascal(N, kind='upper')/2**np.arange(1, N+1)
    res = np.exp(A/2) / t * (np.sum(temp)-val[0]/2)
    return np.maximum(res, threshold)


def p_k0k_vec_levin(t, k, k0, ak, ck, gamma=4, maxN=100, beta=1):
    # Compute P(I(t)=k|I(0)=k0,theta) by Crawford's method
    # Inverse Laplace transform of continued fraction representation
    # Levin's u transform
    
    # Terms needed for the estimation
    A = gamma * np.log(10)
    idx = np.arange(0, maxN+1)
    s = (A+2*idx*np.pi*1j) / 2 / t
    func = np.real(f_vec(s, int(k), int(k0), ak, ck))
    func[0] /= 2
    func = func * (-1)**np.arange(func.size)
    pre_sum = np.cumsum(func[1:])
    
    # Numerator/Denominator of maximum size
    num = np.zeros(maxN)
    denom = np.zeros(maxN)
    
    lastval = 0
    ncv = 0
    val = 0
    
    # Do the Levin update here
    for j in range(maxN):
        omega = func[j+1] * (beta+j) #w_n = a_n * (beta + n)
        term = 1 / (beta+j)
        denom[j] = term / omega
        num[j] = pre_sum[j] * denom[j]
        
        if (j>0):
            ratio = (beta+j-1) * term
            for i in range(1, j+1):
                fact = (j-i+beta) * term
                num[j-i] = num[j-i+1] - fact*num[j-i]
                denom[j-i] = denom[j-i+1] - fact*denom[j-i]
                term *= ratio
        
        if (np.abs(denom[0])<1e-300):
            val = lastval
        else:
            val = num[0] / denom[0]
        
        laststep = np.abs(val-lastval)
        if (laststep <= 1e-50):
            ncv += 1
        if (ncv>=2):
            break
        lastval = val
        
    res = np.exp(A/2) / t * (func[0] + val)
    print(j, "steps", func[0], val)
    return np.maximum(res, threshold)


def Q_assembly(ak, ck):
    # Assemble transition rates matrix
    N = ak.size
    Q = np.diag(-(ak+ck))
    idxu = np.nonzero(np.tri(N, k=1)-np.tri(N, k=0))
    Q[idxu] = ak[0:-1]
    idxl = np.nonzero(np.tri(N, k=-1)-np.tri(N, k=-2))
    Q[idxl] = ck[1:]
    return Q
