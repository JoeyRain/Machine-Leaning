import numpy as np
import pandas as pd
from pandas import Series,DataFrame

class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return np.multiply(x ,self.slope.transpose()) + np.multiply(np.ones(x.shape),self.intercept.transpose())
    def reverse(self, y):
        return (y - self.intercept) / self.slope

def mapminmax(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=0)
    xmin = x.min(axis=0)
    if (xmax == xmin).any():
        raise ValueError("some rows have no variation")
    slope = ((ymax - ymin) / (xmax - xmin))[:, np.newaxis]
    intercept = (-xmin * (ymax - ymin) / (xmax - xmin))[:, np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps

def creatCent(dataSet,k):
    n=dataSet.shape[1]
    minimal=dataSet.min(axis=0)
    scope=dataSet.max(axis=0)-minimal
    centroids=np.multiply(np.ones([k,n]),minimal)+np.random.random([k,n])*scope
    return centroids

def distEuclid(vecA, vecB):
    return np.sqrt(np.sum(np.power((np.asanyarray(vecA) - np.asanyarray(vecB)), 2)))

def kmeans(dataSet,k):  #k代表聚类中心个数
    m=dataSet.shape[0]
    clusterLabel=np.zeros([m,1])
    centroids=creatCent(dataSet,k)
    clusterChanged=True
    tempCentroids=np.zeros([k,dataSet.shape[1]])
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist=np.inf
            minIndex=-1
            for j in range(k):
                tempDist=distEuclid(centroids[j,:],dataSet[i,:])
                if tempDist<minDist:
                    minDist=tempDist
                    minIndex=j
                if clusterLabel[i]!=minIndex:
                    clusterChanged=True #如果有一个点的归类发生变化，就重新进行聚类
                clusterLabel[i]=minIndex
            print(centroids)
        for L in range(k):
            allInCluster=dataSet[np.nonzero(clusterLabel==L)[0],:]
            if(len(allInCluster)!=0):
                centroids[L,:]=np.mean(allInCluster,axis=0)
        if(tempCentroids.all()==centroids.all()):
            break
        else:
            tempCentroids=centroids
    return centroids,clusterLabel

def gauss(dataSet,centroids,k):
    N=dataSet.shape[0]
    #确定径向基函数的宽度因子
    delta=np.ones(k)*100
    for i in range(k):
        for j in range(k):
            if(j!=i):
                tempDelta=distEuclid(centroids[j,:],centroids[i,:])
                if(tempDelta<delta[i]):
                    delta[i]=tempDelta
    H=np.zeros([N,k])
    for m in range(N):
        for n in range(k):
            squareDistEuclid=np.sum(np.power((dataSet[m,:] - centroids[n,:]),2))
            H[m,n]=np.exp(-squareDistEuclid/delta[n]**2)
    return H

def softmax(rawY):
    rawY[rawY == 'bending1'] = -3
    rawY[rawY == 'bending2'] = -2
    rawY[rawY == 'walking'] = -1
    rawY[rawY == 'standing'] = 0
    rawY[rawY == 'sitting'] = 1
    rawY[rawY == 'lying'] = 2
    rawY[rawY == 'cycling'] = 3
    print(type(rawY))
    denSoftmax =sum(np.exp(range(-3,4)))
    trainY=np.exp(rawY.astype(float))/denSoftmax
    return trainY

def main():
    k=10    #聚类个数
    filepath='../arem_train.csv'
    trainData=pd.read_csv(filepath)
    rawX=trainData.iloc[:,range(0,6)].values
    rawY=trainData.iloc[:,6].values
    trainX,ps1=mapminmax(rawX)
    trainY=softmax(rawY)
    centroids,clusterLabel=kmeans(trainX,k)
    #print(centroids,clusterLabel)
    H=gauss(trainX,centroids,k)
    H=np.mat(H)
    T=np.mat(trainY).T
    W=(H.T*H).I*H.T*T
    #print(W)




if __name__=='__main__':
    main()