
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


file_name = input ("write the file address")


# In[4]:


split_const=0
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


# In[10]:


#test
print(feature_values)


# In[11]:


#Entropy of the Root (source or labels)
def source_entropy(y_data,labels):
    #probabilities
    probs = n.empty(len(labels[1]))
    for label in labels[1]:
        probs = n.append(probs, float(label/len(y_data)))
    #Entropy
    source_entropy = stats.entropy(probs,base=2)
    return source_entropy


# In[12]:


#test
print (source_entropy(y_data,labels))


# In[13]:


def attribute_value_entropyCal(y_data_given_attr,labels):
    #probabilities
    probs_init = dict(zip(labels[0], n.zeros(n.shape(labels[1]))))
    y_data_given_attr_occurances = n.unique(y_data_given_attr, return_counts=True)
#     print(y_data_given_attr_occurances, probs_init )
    probs_given_attr = dict(zip(y_data_given_attr_occurances[0], y_data_given_attr_occurances[1]))
#     print(probs_given_attr, probs_init)
    probs = n.empty(len(labels[1]))
    for key, value in probs_given_attr.items():
        probs_init[key] += value
        probs = n.append(probs, probs_init[key]/len(y_data_given_attr))
    #Entropy
    attribute_value_entropy = stats.entropy(probs)
    return attribute_value_entropy


# In[14]:


#Feature_Entropy
def attribute_value_Entropy(att,attr_value, x_data, y_data):
#     print(n.shape(x_data))
    indecies= n.where(x_data[:,att] == attr_value)# for instance all records have sunny value for the weather feature
    y_data_given_attr = n.take(y_data,indecies)
#     test
#     print(y_data_given_attr[0])
    attribute_value_entropy = attribute_value_entropyCal(y_data_given_attr[0],labels)
    
    return attribute_value_entropy

# print (attribute_value_Entropy(0,'high', x_data, y_data))


# In[15]:


#computing information Gain for each attribute    
def attribute_informationGain(x_data, y_data, feature_values, labels):
    
    sentropy = source_entropy(y_data,labels)
    infogain = {}
    for att, attr_values in feature_values.items():#'0': array(['high', 'low', 'med', 'vhigh'],'1' :array(['high', 'low', 'med', 'vhigh']
        sum = 0.0
        for attr_value in attr_values:#['high', 'low', 'med', 'vhigh']
            sum += attribute_value_Entropy(int(att),attr_value, x_data, y_data)
        infogain[att] = sentropy - sum
    attribute = max(infogain.items(), key=operator.itemgetter(1))[0]
    return attribute


# In[16]:


# print (attribute_informationGain(x_data,y_data,feature_values,labels), type(attribute_informationGain(x_data,y_data,feature_values,labels)))


# In[17]:


def pureness(y_data):
    
    labels = n.unique(y_data)
    if len(labels) == 1:
         return labels[0]
    else:
         return 0
        
def most_frequent_label(y_data):
    if len(y_data)==0 :
        return None
    counts = n.unique(y_data, return_counts=True)
#     print(counts)
    index = n.argmax(counts[1])
    return counts[0][index]

# print(most_frequent_label(y_data))
    


# In[18]:


def training_tree(y_data,x_data,feature_values,labels,d3, parent,feature_value):
       
       #Final Conditions 
       #First, Check there is any data
       global counter
       if len(x_data)==0:
           return
       #Second, Check it is pure 
       #check to see It is Pure or Not!!!!!!PURENESSSSS
       if pureness(y_data)!=0:#It is Pure
           d3.create_node(str(feature_value),str(counter), parent, data={'label': most_frequent_label(y_data), 'featurevalue':feature_value}) 
           counter += 1
           return
       #Third, Check there is any feature
       if len(feature_values) > 1:
           feature = attribute_informationGain(x_data, y_data, feature_values, labels)
          
       elif len(feature_values) == 1:
           feature = list(feature_values.keys())[0]
           
       #When Third Condition is met. There is no more feature   
       elif len(feature_values) == 0:
           d3.create_node(str(feature_value),str(counter), parent, data={'label': most_frequent_label(y_data),'featurevalue':feature_value})
           counter +=1
           return 
       
       #preparing feature_values_new
       feature_values_new = feature_values.copy()
       feature_values_new.pop(feature)
       
       if len(feature_values) >= 1:
#           test
#           print(feature_values,feature)
           if parent == None:
               d3.create_node(str(feature),str(counter),data={'feature':feature,'featurevalue':feature_value,'label': most_frequent_label(y_data)})
           else:
               d3.create_node(str(feature),str(counter),parent,data={'feature':feature ,'featurevalue':feature_value, 'label': most_frequent_label(y_data)})  
           new_parent = str(counter)
              
           for fv in feature_values[feature]:
               counter +=1  
               #preparing x_data, getting x_data with column of attribute=attribute value
               x_data_new_indecies = n.where(x_data[:,int(feature)] == fv)[0]
               x_data_new = n.take(x_data, indices=x_data_new_indecies, axis=0)
               #print(x_data_new)
               #preparing y_data
               y_data_new = n.take(y_data, x_data_new_indecies)
               d3.create_node(str(feature),str(counter),new_parent,data={'feature':feature ,'featurevalue':fv, 'label': most_frequent_label(y_data_new)})   
       return 


# In[19]:


# training_tree(y_data,x_data,feature_values,labels,D3,parent=None, feature_value="All")
# D3.show()


# In[20]:


def weight_to_record(y_data):
    weights = n.ones(n.shape(y_data)) / len(y_data)
    return weights

def samples(x_data, y_data, weights, subset_size):
    length = len(y_data)
    indecies_new = n.random.choice(n.arange(length), size=subset_size, replace=True, p=weights)
    y_data_new = n.take(y_data, indecies_new)
    x_data_new = n.take(x_data, indecies_new, axis=0)
    return y_data_new, x_data_new


# In[21]:


def test_record(d3, y_record, x_record, node):
    if node.is_leaf()== True: # it is leaf
        if node.data['label'] == y_record:
            return 1 #detect correctly
        else:
            return 0 #detect wrongly
        
    # It is not leaf and has to keep going
    feature_value = x_record[int(node.tag)] # get the feature value of the expected column
    for n in d3.children(node.identifier):
        if n.data['featurevalue'] == feature_value:
            #print(n)
            return test_record(d3, y_record, x_record, n)


# In[22]:


def error(d3, y_data, x_data, root, weights):
    err=0
    correctlabels = []
    for index in n.arange(n.shape(x_data)[0]):
        if test_record(d3, y_data[index], x_data[index], root)!=1:
            err += weights[index]
            correctlabels.append(0)
        else:
            correctlabels.append(1)
    return err, correctlabels                    


# In[23]:


def new_weights(correctlabels, weights, error):
    beta = error/(1-error)
    
    for  i in range(len(correctlabels)):
        if correctlabels[i]==1:
            weights[i]= weights[i]*beta
    weights = weights/n.sum(weights)        
    return weights        


# In[24]:


# weights = weight_to_record(y_data) # give 1/N probability to each sample of dataset
# trees = [] 
# counter = 0
def adaboost(y_data ,x_data, feature_values, labels, weights, trees,size):
    y_data_new, x_data_new = samples(x_data, y_data, weights, 500)
    D3 = Tree()
    global counter
    counter = 0 
    training_tree(y_data_new ,x_data_new, feature_values, labels, D3, parent=None, feature_value="All")
#     print(D3)
    error_, correctlabels = error(D3, y_data, x_data, D3.get_node(D3.root), weights)
    if len(trees) < size:
        trees.append({'tree':D3,'beta':error_/(1-error_)})
#         trees.append(D3)
        weights = new_weights(correctlabels, weights, error_)
        adaboost(y_data ,x_data, feature_values, labels, weights, trees,size)   
    else:
        return    


# In[25]:


#print(test_record(D3, y_data[10], x_data[10], D3.get_node(D3.root)))
# adaboost(y_data ,x_data, feature_values, labels, weights, trees,size=60)
# for tree in trees:
#     tree.show()


# In[26]:


def label_in_tree(d3, x_record, node):
    if node.is_leaf()== True: # it is leaf
        return node.data['label'] 
        
    # It is not leaf and has to keep going
    feature_value = x_record[int(node.tag)] # get the feature value of the expected column
    flag=0
    for node_ in d3.children(node.identifier):
#         print(node_)
        if node_.data['featurevalue'] == feature_value:
            flag=1
            return label_in_tree(d3, x_record, node_)
            
    if flag!=1:
        return node.data['label'] 


# In[27]:


def voting(trees, x_record):
    votes={}
#     math.log10(t['beta'])
    for t in trees:
            label =  label_in_tree(t['tree'], x_record,  t['tree'].get_node(t['tree'].root))
            if label in votes:
                votes[label] += t['beta']
            else:
                votes[label] = t['beta']
#     print(votes, max(votes.items(), key=operator.itemgetter(1))[0])            
    return max(votes.items(), key=operator.itemgetter(1))[0] #this is the final record      


# In[28]:


def test(trees, y_data, x_data):
    accuracy = 0
    
    for index in n.arange(n.shape(x_data)[0]):
        if voting(trees, x_data[index]) == y_data[index]:
            accuracy +=1 

    return accuracy*100/len(x_data)


# In[29]:


# test(trees, y_data, x_data)


# In[ ]:


def xysplit(split, data):
    x_data = n.delete(data , split , axis = 1)
    y_data = data[:, split]
    return x_data, y_data


def five_fold_cross_validation(data, split,fold):
    global counter
    global trees
    n.random.shuffle(data)
    subdata = n.array_split(data,indices_or_sections=fold,axis=0)
#     print(subdata)
    accuracy=0
    for i in n.arange(fold):
        x_data_test, y_data_test = xysplit(split, subdata[i])
        subdata_copy = n.delete(subdata, i, axis=0)
        subdata_copy = n.concatenate(subdata_copy, axis=0)
        x_data_train, y_data_train = xysplit(split, subdata_copy) 
        trees = []
        counter = 0   
        weights = weight_to_record(y_data_train)
        adaboost(y_data_train ,x_data_train, feature_values, labels, weights, trees,size=200)
#         D3.show()
        accuracy += test(trees, y_data_test, x_data_test)
#         print(accuracy)
    print(accuracy/fold)
    return (accuracy/fold)  
    

#Ten times 5-fold Cross Validation
accuracy=0
for i in n.arange(10):
    accuracy += five_fold_cross_validation(data, split=split_const,fold=5)
print(accuracy/10)

