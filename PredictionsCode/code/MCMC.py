# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk
# This code contains MonteCarlo functions used in the paper
# See https://github.com/BayIAnet/NetworkInferenceFromPopulationLevelData
#
# Modified by Tanja Zerenner t.zerenner@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import timeit

class MC():
    # Monte-Carlo algorithms for inverse problems

    def __init__(self, negll, Cov):

        # Negative log-likelihood and its derivatives
        self.negll = negll
        self.Cov = Cov

    # Definition of algorithms (MCMC MH, SMC)

    def MCMC(self, u0, h, n):
        # Run the MCMC algorithm with Metropolis-Hastings

        print('Running MCMC MH sampling')
        print('# of total samples: ', n)

        # Initialization
        proba = 0.0
        results = np.zeros((n+1, u0.size+1))
        self.ui = u0
        self.nll = self.negll(self.ui)
        results[0, :] = np.hstack((self.nll, self.ui))

        print('Initial nll', np.round(results[0, 0], decimals=2))
        tic = timeit.default_timer()

        for j in range(n):
            acpt = self.RW_proposal(h)
            proba += acpt / n
            results[j+1, :] = np.hstack((self.nll, self.ui))

        toc = timeit.default_timer()
        print('Run time is', toc-tic)
        print('Probability', proba)

        return proba, results

    def RW_proposal(self, h):
        # Standard Random-Walk
        up = np.random.multivariate_normal(self.ui, h**2*self.Cov)

        # Likelihood ratio
        nll = self.negll(up)

        # Phi(u)-Phi(v)
        log_ratio = self.nll - nll

        acpt = (np.random.rand() <= np.minimum(1, np.exp(log_ratio)))
        if (acpt == 1):
            self.nll = nll
            self.ui = up
        return acpt


def post_results(results, nburn, nskip):
    # Post treatment of MCMC results
    ntot = results.shape[0]-1
    idx = (nburn + np.arange(0, ntot+1-nburn, nskip)).astype(int)
    keep = results[idx, :].copy()
    CM = np.mean(keep[:, 1:], axis=0)
    return CM, keep


def plot_results(results):
    # Plot evolution of NLL and u

    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].plot(-results[:, 0])
    axarr[0].set_ylabel(r'$l$')
    axarr[1].plot(results[:, 1])
    axarr[1].set_ylabel(r'$C$')
    axarr[2].plot(results[:, 2])
    axarr[2].set_ylabel(r'$p$')
    axarr[3].plot(results[:, 3])
    axarr[3].set_ylabel(r'$q$')
    plt.xlim(0, results[:, 0].size)
    plt.xlabel("Steps")
    plt.show()


def keep_MAP(PDE, keep):
    omf = PDE.negll(keep[0, 1:]) + np.dot(keep[0, 1:], keep[0, 1:])/2
    MAP = keep[0, :]
    print("Omf", omf)
    for i in range(1, keep.shape[0]):
        omfn = PDE.negll(keep[i, 1:]) + np.dot(keep[i, 1:], keep[i, 1:])/2
        if (omfn <= omf):
            omf = omfn.copy()
            MAP = keep[i, :]
            print("Omf", omf)
    return MAP[1:]


def plot_acf(results, burn, lags=200):
    # Plot acf for NLL, 1st, 10th and 50th coordinates
    NLL = acorr(results[burn:, 0])
    u0 = acorr(results[burn:, 1])
    u1 = acorr(results[burn:, 2])
    u2 = acorr(results[burn:, 3])

    plt.figure()
    plt.plot(np.arange(lags), NLL[0:lags], label=r'$NLL$')
    plt.plot(np.arange(lags), u0[0:lags], label=r"$C$")
    plt.plot(np.arange(lags), u1[0:lags], label=r"$p$")
    plt.plot(np.arange(lags), u2[0:lags], label=r"$q$")

    plt.title('Acf (excluding burnin)')
    plt.xlim(0, lags)
    plt.ylim(-0.1, 1)
    plt.legend(loc='best')
    plt.show()


def acorr(x):
    # http://stackoverflow.com/q/14297012/190597
    # http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
