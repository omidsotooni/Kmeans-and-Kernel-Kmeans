from math import exp 
import numpy as np
def avgji(data1,data2,sigma):
    sum=0
    for i in range(0, data2.shape[0]):
        sum += kernel(data1, data2[i,:], sigma)
    avg= sum / data2.shape[0]
    return avg


def sqnormi(nData,sigma):
    sum = 0
    nData=np.array(nData)
    nCData = len(nData)
    #Gram = [[0] * nData for i in range(nData)] # nData x nData matrix
    # TODO    
    # symmetric matrix
    for i in range(nCData):
        for j in range(nCData):        
            sum += kernel(nData[i,:],nData[j,:],sigma)  
    # Compute squared norm of cluster means     
    sqnorm=(1/(nCData**2))*sum
    return sqnorm

def kernel(data1,data2, sigma):
    """
    RBF kernel-k-means
    :param data: data points: list of list [[a,b],[c,d]....]
    :param sigma: Gaussian radial basis function
    :return:
     RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
    """
    square_dist = squaredDistance(data1,data2)
    base = 2.0 * sigma**2
    result = exp(-square_dist/base)    
    return result 

def squaredDistance(vec1, vec2):
    sum = 0 
    dim = len(vec1) 
    vec2=(np.asarray(vec2)).flatten()
    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) 
    
    return sum
