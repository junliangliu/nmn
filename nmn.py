
# coding: utf-8

# In[115]:

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random as rd


# In[116]:

#样本数
Xn = 10000


# # 生成数据

# In[117]:

def add_m(x, y):
    return (x+y)

def minus_m(x, y):
    return (x-y)

def multiply_m(x, y):
    return (x*y)

print(add_m(1, 2), minus_m(1, 2), multiply_m(1, 2))

int2func = dict()
int2func[1] = add_m
int2func[2] = minus_m
int2func[3] = multiply_m

f1 = int2func[1]
f2 = int2func[2]
f3 = int2func[3]
print(f1(1, 2), f2(1, 2), f3(1, 2))


#len是模块的个数，也是数据的个数
def get_data(len):
    fs = []
    ds = []
    for i in range(len):
        rdf = rd.randint(1,3)
        rdd = rd.random()
        fs.append(rdf)
        ds.append(rdd)
    return fs,ds
print(get_data(5))

def get_result(fs, ds):
    assert len(fs) == len(ds)
    d0 = 0
    for i in range(len(ds)):
        func = int2func[fs[i]]
        d1 = ds[i]
        d0 = func(d0, d1)
    return d0

print(get_result([1,2,3,1], [1, 3,2,4]))


# In[118]:

funcs = [] #计算链
datas = [] #待计算数据
ys = []    #计算结果
for i in range(0, Xn):
    length = rd.randint(2,3) #限定计算链长度，太长数据可能远远超过1
    assert length>=2 and length<=3
    fs,ds = get_data(length)
    funcs.append(fs)
    datas.append(ds)
    ys.append(get_result(fs, ds))


# # 定义模型

# In[121]:

#每个模块是一个三层的神经网络，relu激活函数
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#最终的模块网络，网络最终有哪些模块由forward函数动态定义
class ModuleNet(nn.Module):
    def __init__(self):
        super(ModuleNet, self).__init__()
        self.add_model = Model()
        self.minus_model = Model()
        self.multiply_model = Model()
        
        self.int2module = dict()
        self.int2module[1] = self.add_model
        self.int2module[2] = self.minus_model
        self.int2module[3] = self.multiply_model        

    # d0 --- module -- d0 --- module --- d0 
    #          |                |
    #         d1               d1
    def forward(self, fs, ds):
        assert len(fs)==len(ds)
        d0 = Variable(torch.Tensor([0]))
        ds = Variable(torch.Tensor(ds))
        for i in range(len(fs)):
            d1 = ds[i]
            invar = torch.cat((d0, d1))
            module = self.int2module[fs[i]]
            d0 = module(invar)
        return d0


# In[122]:

#实例化模块网络，定义损失函数优化器
modulenet = ModuleNet()
lose_fn = torch.nn.MSELoss()
optimizer = optim.SGD(modulenet.parameters(), lr=0.001, momentum=0.9)


# In[123]:

ys = Variable(torch.Tensor(ys))
t = Variable(torch.Tensor([0.2, 0.3]))

for epo in range(100):
    #逐个样本前向计算、反向传播
    for i in range(0, Xn):
        optimizer.zero_grad()
        pred = modulenet(funcs[i], datas[i])
        loss = lose_fn(pred, ys[i])
        loss.backward() 
        optimizer.step()
        
        #打印
        if(epo % 2 == 0 and i<10):
            if i==0:
                print("add: ", modulenet.add_model(t))
                print("minus: ", modulenet.minus_model(t))
                print("multiply: ", modulenet.multiply_model(t))
            print(epo, i, loss.data[0])


# In[128]:

#测试单个模块是否学习结果是否符合预期
t = Variable(torch.Tensor([0.01, 0.03]))
print("add: ", modulenet.add_model(t))
print("minus: ", modulenet.minus_model(t))
print("multiply: ", modulenet.multiply_model(t))


# In[ ]:



