import numpy as np
import numpy.random as rgt


def mad(x):
    '''
    Compute Median Absolute Deviation
    '''
    return np.median(abs(x-np.median(x))) * 1.4826


def cov_generate(std, corr=0.5):
    '''
    Generate a Covariance Matrix with 
    Exponentially Decay Off-diagonal Entries
    '''
    p = len(std)
    R = np.zeros(shape=[p,p])
    for j in range(p-1):
        R[j, j+1:] = np.array(range(1, len(R[j,j+1:])+1))
    R += R.T
    return np.outer(std, std) * (corr*np.ones(shape=[p,p]))** R