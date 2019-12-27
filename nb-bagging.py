
# coding: utf-8

# In[1]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator
import math


# In[2]:


#Working on data, preprocessing


# In[3]:


split_const=0
file_name = input ("write the file address")


# In[4]:


data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[5]:


print(data)
type(data)


# In[6]:


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


# In[7]:


x_data, y_data, labels, feature_values = xysplit(split_const, data)


# In[8]:


#test
print (y_data)
print(len(y_data))
print (x_data[:,3])


# In[9]:


#test
print (labels)
print (n.shape(labels))
print(labels[0])


# In[10]:


#test
print(feature_values)


# In[11]:


def nb_training(x_data, y_data, labels,feature_values):
    p_label = {} 
    for index, label in enumerate(labels[0]):
        p_label[label] = labels[1][index]/len(y_data)
#     print(p_label)

    p_features = {}
    for feature in feature_values.keys():
        if feature not in p_features:
            p_features[feature]={}     
        for feature_value in feature_values[feature]:
            if feature_value not in p_features[feature]:
                p_features[feature][feature_value] = {}
            for index, label in enumerate(labels[0]):
                filterx_indicies = n.where(x_data[:,int(feature)]==feature_value)[0]
                count = len(n.where(n.take(y_data,filterx_indicies)==label)[0])
#                 print(feature,feature_value, count, label)
                p_features[feature][feature_value][label] = count/labels[1][index]
#     print(p_features)  
    
    p_features_only = {}
    for feature in feature_values.keys():
        if feature not in p_features_only:
            p_features_only[feature]={}     
        for feature_value in feature_values[feature]:
            if feature_value not in p_features_only[feature]:
                count = len(n.where(x_data[:,int(feature)] == feature_value)[0])
                p_features_only[feature][feature_value] = count/len(x_data)
#     print(p_features_only)
    return p_label, p_features, p_features_only


# In[12]:


def test_one_record(bags, y_record, x_record,labels):
    accuracy = 0
#     p_label, p_features, p_features_only 
    votes={}
    for bag in bags:
        probs={}
        for label in labels[0]:
            probs[label] = bag['p_label'][label]
            for index, value in enumerate(x_record):
                probs[label] *= bag['p_features'][str(index)][value][label] 
#                 probs[label] /= p_features_only[str(index)][value]
        maximum=0
        finallabel=''
        for label, prob in probs.items():
            if prob > maximum:
                maximum = prob
                finallabel = label
        if finallabel in votes:
            votes[finallabel] += 1
        else:
            votes[finallabel] = 1
    l = max(votes.items(), key=operator.itemgetter(1))[0] 
#     print(l)
    if l == y_record:
        return 1
    else: 
        return 0  


# In[13]:


def samples(x_data, y_data, subset_size):#Get the subset of a dataset(training) Randomly
    length = len(y_data)
    indecies_new = n.random.choice(n.arange(length), size=subset_size, replace=True)
    y_data_new = n.take(y_data, indecies_new)
    x_data_new = n.take(x_data, indecies_new, axis=0)
    return y_data_new, x_data_new


# In[14]:


#A Function for calling Naive bayes for each bag(Random Sample)
def bagging(y_data, x_data,labels,feature_values, num_of_bags, subset_size):
    bags=[]
    
    for num in n.arange(num_of_bags):
        y_data_new, x_data_new = samples(x_data, y_data, subset_size)
        p_label, p_features, p_features_only = nb_training(x_data_new,y_data_new,labels,feature_values)  
        bags.append({'p_label':p_label,'p_features':p_features,'p_features_only':p_features_only})
        
    return bags


# In[15]:


def test(bags, y_data, x_data,labels):
    accuracy = 0
    
    for index, y_rec in enumerate(y_data):
            accuracy += test_one_record(bags ,y_rec, x_data[index],labels)

    return accuracy*100/len(x_data)  


# In[16]:


def xysplit(split, data):
    x_data = n.delete(data , split , axis = 1)
    y_data = data[:, split]
    return x_data, y_data

def five_fold_cross_validation(data, split,fold):
    n.random.shuffle(data)
    subdata = n.array_split(data,indices_or_sections=10,axis=0)
#     print(subdata)
    accuracy=0
    for i in n.arange(fold):
        x_data_test, y_data_test = xysplit(split, subdata[i])
        subdata_copy = n.delete(subdata, i, axis=0)
        subdata_copy = n.concatenate(subdata_copy, axis=0)
        x_data_train, y_data_train = xysplit(split, subdata_copy)
        bags = bagging(y_data_train, x_data_train,labels,feature_values, num_of_bags=20, subset_size=1000)  
        accuracy += test(bags, y_data_test, x_data_test, labels) 
    print(accuracy/fold)
    return (accuracy/fold)    
      


# In[17]:


accuracy=0
for i in n.arange(10):
    accuracy += five_fold_cross_validation(data, split=split_const,fold=5)
print(accuracy/10)

