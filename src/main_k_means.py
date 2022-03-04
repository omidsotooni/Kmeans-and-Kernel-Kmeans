import sys
from LoadData import * 
from k_means import * 
from evaluation import * 
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("[usage] <data-file> <ground-truth-file>")
        exit(1) 
    
    dataFilename = sys.argv[1]
    groundtruthFilename = sys.argv[2]
    
    data = loadPoints(dataFilename) 
    groundtruth = loadClusters(groundtruthFilename) 
    
    nDim = len(data[0]) 
   
    K = 3  # Suppose there are 2 clusters
    print ('K=',K)

    # انتخاب دو عدد اول به عنوان مرکز خوشه
    # use the first two data points as initial cluster centers
    centers = []
    for i in range(K):
        centers.append(data[i])


    # get clusterID, list, updated centers
    results = kmeans(data, centers) 

    res_Purity = purity(groundtruth, results) 
    res_NMI = NMI(groundtruth, results) 
    
    print ("Purity =", res_Purity)
    print ("NMI = ", res_NMI)
    

    # برای اینکه راحت تر ایندکس بشه نقاط دوباره تعیین میشوند
    points = np.empty((0,len(data[0])), float)
    # محاسبه ی فاصله که برای پیدا کردن داده های پرت استفاده میشود
    distances = np.empty((0,len(data[0])), float)
    
    d={}
    for x,y in zip(data,results):
        d.setdefault(y, []).append(x)
    
    for i, center_elem in enumerate(centers):
        # cdist از این کتابخانه برای حساب کردن فاصله هر داده تا مرکز استفاده میشود
        distances = np.append(distances , cdist([center_elem],d[i], 'euclidean')) 
        points = np.append(points, data, axis=0)
    print(points)
    percentile = 98
    # داده های پرتی که فاصله آنها از مرکز بیشتر از درصد تعیین شده را پیدا میکند
    outliers = points[np.where(distances > np.percentile(distances, percentile))]
    fig = plt.figure()    
    plt.scatter(*zip(*data),c=results,marker='s',facecolor="g" ) 
    # یک ضربدر روی داده های پرت میزند
    plt.scatter(*zip(*outliers),marker="x",edgecolor="b",s=70);
    # مرکز هر خوشه یک دایره با کادر قرمز نمایش داده میشود
    plt.scatter(*zip(*centers),marker="o",edgecolor="r",s=20);   
    plt.savefig() 
    plt.show()

