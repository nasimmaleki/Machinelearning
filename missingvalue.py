
# coding: utf-8

# In[29]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator
import math
import csv


# In[30]:


#Working on data, preprocessing


# In[31]:


file_name = input ("write the file address")


# In[32]:


data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[33]:


print(data)
type(data)


# In[34]:


#replaces values of ? with the most frequent value of the same label it has

#When missing value is categorical we use most frequent feature value of that column
def missing_value_category(data,missed_col,label_col):#
    for index, i in enumerate(data[:,missed_col]):
        if i == '?':
            label = data[index,label_col]
            indicies = n.where(data[:,label_col] == label)
            coldata_samelabel = n.take(data[:,missed_col],indicies)
            indicies = n.where(coldata_samelabel != '?')
            coldata_samelabel = n.take(coldata_samelabel ,indicies)
            coldata_samelabel = n.unique(coldata_samelabel,return_counts=True)
            max_index = n.argmax(coldata_samelabel[1])
            data[index,missed_col] = coldata_samelabel[0][max_index]
    return data

#when we have continues values for missing values that we use average
def missing_value_continues(data,missed_col,label_col):
  
    for index, i in enumerate(x_data[:,missed_col]):
        if i == '?':
            label = data[index,label_col]
            indicies = n.where(data[:,label_col] == label)
            coldata_samelabel = n.take(data[:,missed_col],indicies)
            indicies = n.where(coldata_samelabel != '?')
            coldata_samelabel = n.take(coldata_samelabel ,indicies)
            coldata_samelabel = n.mean(coldata_samelabel)
            data[index,missed_col] = coldata_samelabel
    return data


# In[59]:


if file_name == 'mushroom':
    data = missing_value_category(data,11, 0)
    myFile = open('mushroom_missing.csv', 'w') #calling above function to deal with missing values 
    with myFile:  
        writer= csv.writer(myFile)
        writer.writerows(data)
        
if file_name == 'breast-cancer-wisconsin':
    data = missing_value_category(data, 6, 10)#calling above function to deal with missing values
    print(data)
    data = n.delete(data,0,axis=1)
    print(data)
    myFile = open('breast_cancer_wisconsin_missing.csv', 'w')  
    with myFile:  
        writer= csv.writer(myFile)
        writer.writerows(data)

