import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from LoadData import * 
from evaluation import * 
from kernel_k_means import * 


if __name__ == "__main__":
    data0 = loadPoints('self_test.data') 
    groundtruth = loadClusters('self_test.ground') 

    K = 2  # Suppose there are 2 clusters
    t=0
    sigma = 5.0

    nData = len(data0)             
    clusterID = [0] * nData     
    centroid = [0] * K   
    # centroid = [0] * K   
    for i in range(nData):
        # Randomly partition points into k clusters 
        #clusterID[i] = random.randint(0, K-1)            
        clusterID[i] = i%K

    clusterID=  np.array(clusterID)
    data0   =   np.array(data0)       
    DclusterID = [[] for i in range(K)] 
    DclusterID = [data0[clusterID == i] for i in range(clusterID.max() + 1)]    
        
    cluster=data0
    while(True):  

        
            

        kernelDAllCluster= np.ndarray(shape=(data0.shape[0], 0))        
        for k in range(K):    

            # calculate centroid, only for visualization   
            DclusterID[k]=np.array(DclusterID[k])
            centroid[k] = DclusterID[k].mean(axis=0)

            # Compute squared norm of cluster means                                
            sqnorm=sqnormi(DclusterID[k], sigma)  
            sqnormmatrix = np.repeat(sqnorm, data0.shape[0], axis=0)
            sqnormmatrix = np.asmatrix(sqnormmatrix)

             # Average kernel value for xj and Ci
            avgmatrix = np.ndarray(shape=(0,1))
            for j in range(0, data0.shape[0]):                
                avg = avgji(data0[j,:], np.asmatrix(DclusterID[k]),sigma)
                avgmatrix = np.concatenate((avgmatrix, np.asmatrix(avg)), axis=0)
            avgmatrix = np.asmatrix(avgmatrix)
            
            #Find closest cluster for each point
            kernelD = np.add(-2*avgmatrix, sqnormmatrix)
            kernelDAllCluster = np.concatenate((kernelDAllCluster, kernelD), axis=1)            
        jclusterMatrix = np.ravel(np.argmin(np.matrix(kernelDAllCluster), axis=1))
        jclusterMatrix=np.where(jclusterMatrix>K-1, K-1, jclusterMatrix)
        #Cluster reassignment   
        listClusterMember = [[] for l in range(K)]
        for i in range(0, data0.shape[0]):
            listClusterMember[np.asscalar(jclusterMatrix[i])].append(data0[i,:])
        for i in range(0, K):
            print("Cluster member numbers-", i, ": ", listClusterMember[0].__len__())

        #break when converged
        boolAcc = True
        for m in range(0, K):
            prev = np.asmatrix(DclusterID[m])
            current = np.asmatrix(listClusterMember[m])
            if (prev.shape[0] != current.shape[0]):
                boolAcc = False
                break
            if (prev.shape[0] == current.shape[0]):
                boolPerCluster = (prev == current).all()
            boolAcc = boolAcc and boolPerCluster
            if(boolAcc==False):
                break
        if(boolAcc==True):
            break
        t += 1
        #update new cluster member
        DclusterID = np.array(listClusterMember) 
        print("iteration: ", t)


    results=jclusterMatrix
    print ('K=',K)

    
    res_Purity = purity(results, groundtruth)
    res_NMI = NMI(results, groundtruth) 
    
    print ("Purity =", res_Purity)
    print ("NMI = ", res_NMI)

    points = []
    
    #for i in range(len(results)):
    #    if (jclusterMatrix[i]!=groundtruth[i]):
    #        points.append(data0[i])
            
    fig = plt.figure()    
    plt.scatter(*zip(*data0),c=results,marker='s',facecolor="w" ) 
    # بر روی داده های اشتباه  ضربدر زده می شود
    #plt.scatter(*zip(*points),marker="x",edgecolor="r",s=70);    
    plt.show()

    

    
