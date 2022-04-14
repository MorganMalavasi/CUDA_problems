import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg
from numba import jit


def computing_weights(dataset):
    weights = euclidean_distances(dataset, dataset)
    return weights / linalg.norm(weights)

@jit
def C_S(matrixOfWeights, theta):
    sin_t = np.sin(theta)
    S = np.dot(matrixOfWeights, sin_t)

    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)

    return S, C

@jit(nopython=True) 
def loop(matrixOfWeights, theta, S, C, eps):
    
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]

    while ok == True:
        ok = False
        rounds += 1
        
        ''' loop on the theta '''
        for k in range(thetaSize):
            old = theta[k]  
            
            ''' find a theta that improves the equation '''
            theta[k] = np.arctan(S[k]/C[k])
            if C[k] >= 0:
                theta[k] += np.pi
            elif S[k] > 0:
                theta[k] += 2*np.pi
                
            jit_elementwise_multiplication(matrixOfWeights[k,:], C, S, theta, k, old)

            ''' exit condition '''
            if min(abs(old - theta[k]), abs(2*np.pi - old + theta[k])) > eps:
                ok = True
    
    return theta

@jit(nopython=True, parallel=True)
def jit_elementwise_multiplication(line_weights, C, S, theta, k, old):
    # elementwise multiplication
    C += np.multiply(line_weights, np.repeat(np.cos(theta[k]) - np.cos(old), theta.shape[0]))
    S += np.multiply(line_weights, np.repeat(np.sin(theta[k]) - np.sin(old), theta.shape[0]))