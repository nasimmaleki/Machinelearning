
# coding: utf-8

# In[69]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator
import math
import csv


# In[70]:


#Working on data, preprocessing


# In[71]:


file_name = input ("write the file address")


# In[1]:


data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[73]:


print(data)
type(data)


# In[74]:


#replaces values of ? with the most frequent feature value of that column that has the same label
def bining(data,label_col,num):
    bins = int(len(data)/num)
    print(bins,num)
    for index, col in  enumerate(n.arange(label_col)):
        data = sorted(data,key=lambda x:x[index])#sorting dataset based on the value of column number #index
        for i, d in enumerate(data): 
            data[i][index]=int(i/bins)
    print(data)
    return data


# In[75]:


if file_name == 'ecoli':
    data = n.delete(data,0,axis=1)#Remove ID column of Ecoli
    data = bining(data,label_col=7,num=10)#Bining For Ecoli all columns except 7th col which is the label
    myFile = open('ecoli_bining.csv', 'w') #Saving a file
    with myFile:  
        writer= csv.writer(myFile)
        writer.writerows(data)

