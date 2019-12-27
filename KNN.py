
# coding: utf-8

# In[44]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator
import math
import re


# In[5]:


#Working on data, preprocessing


# In[63]:


split_cost=0
file_name = input ("write the file address")


# In[64]:


data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[65]:


print(data)
type(data)


# In[66]:


def xysplit(label_index, data):
    x_data = n.delete(data , label_index , axis = 1)
    y_data = data[:, label_index]
    labels = n.unique(y_data, return_counts=True)
    feature_values = {}
    x_data_T=n.transpose(x_data)
    for i in range(0,n.shape(x_data_T)[0]):
        index= str(i)
        feature_values[index] = n.unique(x_data_T[i,:])
    
    #returning the attribute values, labels, x_data, y_data
    return x_data, y_data, labels, feature_values


# In[67]:


x_data, y_data, labels, feature_values = xysplit(split_cost, data)
if file_name == "ecoli":
        x_data = x_data.astype(n.float)
print(x_data)

if file_name == "breast_cancer_wisconsin_missing":
        x_data = x_data.astype(n.int)
print(x_data)

if file_name == "letter-recognition":
        x_data = x_data.astype(n.int)
print(x_data)


# In[68]:


#test
print (y_data)
print (x_data)


# In[69]:


#test
print (labels)
print (n.shape(labels))
print(labels[0])


# In[70]:


#test
print(feature_values)
# test_data = x_data
# print(test_data)
# n.shape(test_data)


# In[71]:


def Euclidean_Distance(test_record,x_data,k):

    indicies = {}
    
    for row, data in enumerate(x_data):
        dist = 0
        for col, field in enumerate(data):
            if type(field) == "int" or type(field) == "float":
                dist += math.pow(test_record[col]-field)
            elif field != test_record[col]:
                dist += 1
        if len(indicies)< k:
            indicies[row]=dist
        else:
            index = max(indicies.items(), key=operator.itemgetter(1))[0]
            if indicies[index]> dist: 
                del indicies[index]
                indicies[row]=dist
    return indicies


def Manhattan_Distance(test_record,x_data,k):
    indicies = {}
    
    for row, data in enumerate(x_data):
        dist = 0
        for col, field in enumerate(data):
            if type(field) == "int" or type(field) == "float" :
                dist += math.abs(test_record[col]-field)
            elif field != test_record[col]:
                dist += 1
        if len(indicies)< k:
            indicies[row]=dist
        else:
            index = max(indicies.items(), key=operator.itemgetter(1))[0]
            if indicies[index]> dist: 
                del indicies[index]
                indicies[row]=dist
    return indicies

def vote(k_dist,y_data):
    labels = n.take(y_data,k_dist)
    labels = n.unique(labels,return_counts=True)
    return labels[0][n.argmax(labels[1])]    
    
def KNN(x_data_test,y_data_test,x_data, y_data, labels,feature_values,k):
    test=0
    for index, record_data in enumerate(x_data_test):
#          k_dist = Euclidean_Distance(record_data,x_data,k)
        k_dist = Manhattan_Distance(record_data,x_data,k)
#         print(k_dist)
        keys=list(k_dist.keys())
        if y_data_test[index] == vote(keys,y_data):
            test +=1
    return 100*test/len(y_data_test)      


# In[72]:


def xysplit(split, data):
    x_data = n.delete(data , split , axis = 1)
    y_data = data[:, split]
    return x_data, y_data

def five_fold_cross_validation(data, split,fold):
    n.random.shuffle(data)
    subdata = n.array_split(data,indices_or_sections=fold,axis=0)
#     print(subdata)
    accuracy=0
    for i in n.arange(fold):
        x_data_test, y_data_test = xysplit(split, subdata[i])
        subdata_copy = n.delete(subdata, i, axis=0)
        subdata_copy = n.concatenate(subdata_copy, axis=0)
        x_data_train, y_data_train = xysplit(split, subdata_copy) 
        accuracy += KNN(x_data_test,y_data_test,x_data_train, y_data_train, labels,feature_values,3)
     
    print(accuracy/fold)
    return(accuracy/fold)
    
accuracy=0
for i in n.arange(10):
    accuracy += five_fold_cross_validation(data, split=split_cost,fold=5)
print(accuracy/10)

