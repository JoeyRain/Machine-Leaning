import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import random

class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y - self.intercept) / self.slope

def mapminmax(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=-1)
    xmin = x.min(axis=-1)
    if (xmax == xmin).any():
        raise ValueError("some rows have no variation")
    slope = ((ymax - ymin) / (xmax - xmin))[:, np.newaxis]
    intercept = (-xmin * (ymax - ymin) / (xmax - xmin))[:, np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps

def activate(x,activateFunction='sigmoid'):
    if activateFunction=='tansig':
        return (1-np.exp(-x))/(1+np.exp(-x))
    elif activateFunction=='pureline':
        return x
    else:
        return 1/(1+np.exp(-x.astype(float)))

def deltaActivateFunction(x,activateFunction='sigmoid'):
    if activateFunction=='tansig':
        return 2*np.exp(-x)/np.square(1+np.exp(-x))
    elif activateFunction=='pureline':
        return 1
    else:
        return np.exp(-x)/np.square(1+np.exp(-x))

class bpNet:
    learningRate=0.25
    momentum=0.9
    batch=1
    epoches=10000
    accuracy=0.1    #normalized RMSE is adopted
    numInputNode:int=0
    numHiddenNode: int=0
    numOutputNode:int=0
    activateFunction1='sigmoid' #隐含层激励函数
    activateFunction2='pureline'    #输出层激励函数
    "各层网络的相关数值"
    V=[]   #输入层到隐含层的权重mx(n+1)
    iH=[]   #隐含层的局部诱导域，即激活之前的值
    H=[]    #隐含层激活之后的值mxb
    W=[]   #隐含层到输出层的权重Lx(m+1)
    iO=[]   #输出层的局部诱导域
    O=[]    #输出层激活之后的值

    def __init__(self,numInputNode,numOutputNode):
        self.numInputNode=numInputNode
        self.numOutputNode=numOutputNode
        self.numHiddenNode=int(np.sqrt(numInputNode+numOutputNode)+5)  #默认采用典型的三层结构
        self.V=-1+2*np.random.random([self.numHiddenNode,self.numInputNode+1]) #mx(n+1)
        self.W=-1+2*np.random.random([self.numOutputNode,self.numHiddenNode+1])    #Lx(m+1)

    def train(self,x,y,batch=1):
        '''训练神经网络'''
        numbatches=int(x.shape[0]/batch)
        W1=np.zeros([self.numOutputNode,self.numHiddenNode+1])
        W2=np.zeros([self.numOutputNode,self.numHiddenNode+1])
        V1=np.zeros([self.numHiddenNode,self.numInputNode+1])
        V2=np.zeros([self.numHiddenNode,self.numInputNode+1])
        for i in range(0,self.epoches):
            state=np.random.get_state()
            np.random.shuffle(x)  # 采用随机梯度
            np.random.set_state(state)
            np.random.shuffle(y)
            #前向计算过程
            for j in range(0,numbatches):
                batchX=x[j*batch:(j+1)*batch,:].T    #nxb
                batchY=y[j*batch:(j+1)*batch]   #Lxb
                barX=np.concatenate((batchX,np.ones([1,batch])),axis=0) #加一行偏置(n+1)xb
                self.iH=np.dot(self.V,barX)    #mxb
                self.H=activate(self.iH,self.activateFunction1)
                barH=np.concatenate((self.H,np.ones([1,batch])),axis=0)  #(m+1)xb
                self.iO=self.W@barH    #Lxb
                self.O=activate(self.iO,self.activateFunction2)
                #误差反向传播过程
                error=batchY-self.O #Lxb
                #delta2=error*(-deltaActivateFunction(self.iO,self.activateFunction2))@np.transpose(barH)   #Lx(m+1)
                delta2 = error * (-1) @ np.transpose(barH)
                self.W=self.W-self.learningRate*delta2+self.momentum*(W1-W2)
                W2=W1
                W1=self.W
                delta1=np.transpose(np.transpose(error*(-1))*self.W[:,range(0,self.numHiddenNode)])*self.H*(1-self.H)*np.transpose(barX)
                self.V=self.V-self.learningRate*delta1+self.momentum*(V1-V2)
                V2=V1
                V1=self.V
            batchMSE=np.mean(np.square(error),axis=0)
            print(batchMSE)
            if batchMSE.max(axis=0)<0.001:
                break

    def simulate(self,x):
        barX = np.concatenate((x, np.ones([1,x.shape[0]])), axis=1)
        self.iH=np.dot(self.V,np.transpose(barX))
        self.H=activate(self.iH)
        barH = np.concatenate((self.H, np.ones([1,self.H.shape[1]])), axis=0)
        self.iO=np.dot(self.W,barH)
        self.O=self.iO
        return self.O

def valize(op):
    op[op == 'INLAND'] = 1
    op[op == '<1H OCEAN'] = 10
    op[op == 'NEAR OCEAN'] = 100
    op[op == 'NEAR BAY'] = 1000
    op[op == 'ISLAND'] = 10000
    return op

def main():
    #训练数据和测试数据的预处理
    filepath='../ml-homework2/housing_train.csv'
    trainData=pd.read_csv(filepath)
    op=trainData['ocean_proximity']
    op=op.values
    op=valize(op)
    rawX=pd.concat([trainData.iloc[:,range(0,8)],pd.DataFrame(op)],axis=1)
    rawY=trainData.iloc[:,9]
    normalX,ps1=mapminmax(np.transpose(rawX.values))
    trainX=np.transpose(normalX)
    Y = rawY.values
    ymin = Y.min(axis=0)
    ymax = Y.max(axis=0)
    trainY = (Y-ymin) / (ymax - ymin)

    filepath = '../ml-homework2/housing_test.csv'
    testData = pd.read_csv(filepath)
    op = testData['ocean_proximity'].values
    op=valize(op)
    testX = pd.concat([testData.iloc[:, range(0,8)], pd.DataFrame(op)], axis=1)
    testY = testData.iloc[:, 9]
    testX=np.transpose(ps1(np.transpose(testX.values)))

    net=bpNet(trainX.shape[1],1)
    bpNet.train(net,trainX,trainY)
    simuY=bpNet.simulate(net,testX)
    simuY=simuY*(ymax-ymin)+ymin
    rmse=np.sqrt(np.sum(np.square(testY-simuY)/testY)/len(testY))   #normalized RMSE
    print(rmse)

if __name__=='__main__':
    main()