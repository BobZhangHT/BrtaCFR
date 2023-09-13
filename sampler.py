import pymc as pm
import numpy as np
import scipy.stats as sps
import pytensor.tensor as pyt

# define the logarithmic proability for the fused LASSO prior
def logp(value, lam, N):
    # value: beta within the probability
    # lam: penalty parameter
    # N: sample size
    
    Dmat = pyt.zeros((N-1,N))
    Dmat += pyt.eye(N-1,N)
    Dmat -= pyt.eye(N-1,N,1)
    return -lam*pyt.sum(pyt.abs(pyt.dot(Dmat,value)))+(N-1)*pyt.log(lam/2)

# the sampler for BrtaCFR
def BrtaCFR_EST(ct, dt, 
                F_mean, 
                F_shape,
                random_state=2023):
    # ct, dt: number of cases and deaths
    # F_mean, F_shape: mean and shape parameters of Gamma distribution
    
    N = ct.shape[0]
    # F(s)
    Fs = sps.gamma.cdf(np.arange(N+1), 
                       a=F_shape, 
                       scale=F_mean/F_shape)
    # f(s) = F(s) - F(s-1)
    fs = np.diff(Fs)
    # f(s) matrix
    f_mat = np.zeros((N,N))
    for i in range(N):
        f_mat += np.diag(np.ones(N-i)*fs[i],-i)
    # c(t) matrix
    c_mat = np.diag(ct)
    fc_mat = np.dot(f_mat, c_mat)
    
    with pm.Model() as poisson_model:
        
        lam = pm.HalfCauchy('lam', 
                            beta=1)        
        beta = pm.CustomDist('beta',
                              lam,
                              N,
                              logp=logp,
                              size=N)
        p = pm.Deterministic('p', pm.math.sigmoid(beta))
        # define linear model and exp link function
        theta = pm.math.dot(fc_mat, p)
        # define Poisson likelihood
        y = pm.Poisson("dt", 
                       mu=theta, 
                       observed=dt)
        
    with poisson_model:
         advi_approx = pm.fit(100000, 
                              method=pm.ADVI(random_seed=random_state))

    return advi_approx