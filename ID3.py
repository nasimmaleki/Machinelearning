
# coding: utf-8

# In[1]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator


# In[2]:


#Working on data, preprocessing


# In[3]:


file_name = input ("write the file address")


# In[20]:


split_const=0
data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[21]:


# data = n.delete(data,0,axis=1)
print(data)
type(data)


# In[22]:


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


# In[23]:


x_data, y_data, labels, feature_values = xysplit(split_const, data)


# In[24]:


#test
# print (y_data)
print(len(y_data))
# print (x_data[:,3])


# In[25]:


#test
print (labels)
print (n.shape(labels))


# In[26]:


#test
print(feature_values)


# In[27]:


#Entropy of the Root or any other nodes based on the labels(source or labels) 
def source_entropy(y_data,labels):
    #probabilities
    probs = n.empty(len(labels[1]))
    for label in labels[1]:
        probs = n.append(probs, float(label/len(y_data)))
    #Entropy
    source_entropy = stats.entropy(probs,base=2)
    return source_entropy


# In[28]:


#test
print (source_entropy(y_data,labels))


# In[29]:


#computing Entropy of each attribute values for all existing labels.
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


# In[30]:


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


# In[31]:


#computing information Gain for each attribute (use source entropy and feature entropy to compute this)    
def attribute_informationGain(x_data, y_data, feature_values, labels):
    
    sentropy = source_entropy(y_data,labels)
    infogain = {}
    for att, attr_values in feature_values.items():
        #'0': array(['high', 'low', 'med', 'vhigh'],'1' :array(['high', 'low', 'med', 'vhigh']
        sum = 0.0
        for attr_value in attr_values:#['high', 'low', 'med', 'vhigh']
            sum += attribute_value_Entropy(int(att),attr_value, x_data, y_data)
        infogain[att] = sentropy - sum
    attribute = max(infogain.items(), key=operator.itemgetter(1))[0]
    return attribute


# In[32]:


# print (attribute_informationGain(x_data,y_data,feature_values,labels), type(attribute_informationGain(x_data,y_data,feature_values,labels)))


# In[33]:


#Check if the node is pure or not(all data has to have the same label)
def pureness(y_data):
    
    labels = n.unique(y_data)
    if len(labels) == 1:
         return labels[0]
    else:
         return 0


# In[34]:


#Returns the most frequent label among all labels
def most_frequent_label(y_data):
    counts = n.unique(y_data, return_counts=True)
#     print(counts)
    index = n.argmax(counts[1])
    return counts[0][index]

# print(most_frequent_label(y_data))


# In[35]:


counter = 0  
D3 = Tree() 
def training_tree(y_data,x_data,feature_values,labels,d3, parent,feature_value):
        
        #Final Conditions 
        #First, Check there is any data
        global counter
        if len(x_data)==0:
            return d3
        #Second, Check it is pure 
        #check to see It is Pure or Not!!!!!!PURENESSSSS
        if pureness(y_data)!=0:#It is Pure
            d3.create_node(str(most_frequent_label(y_data))+"("+str(feature_value)+")",str(counter), parent, data={'label': most_frequent_label(y_data), 'featurevalue':feature_value}) 
            counter += 1
            return
        #Third, Check there is any feature
        if len(feature_values) > 1:
            feature = attribute_informationGain(x_data, y_data, feature_values, labels)
           
        elif len(feature_values) == 1:
            feature = list(feature_values.keys())[0]
            
        #When Third Condition is met. There is no more feature   
        elif len(feature_values) == 0:
            d3.create_node(str(most_frequent_label(y_data))+"("+str(feature_value)+")",str(counter), parent, data={'label': most_frequent_label(y_data),'featurevalue':feature_value})
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
                d3.create_node(str(feature),str(counter),parent,data={'feature':feature ,'featurevalue':feature_value,'label': most_frequent_label(y_data)})  
            new_parent = str(counter)
            counter +=1     
            for fv in feature_values[feature]:
                #preparing x_data, getting x_data with column of attribute=attribute value
                x_data_new_indecies = n.where(x_data[:,int(feature)] == fv)[0]
                x_data_new = n.take(x_data, indices=x_data_new_indecies, axis=0)
                #print(x_data_new)
                #preparing y_data
                y_data_new = n.take(y_data, x_data_new_indecies)
                training_tree(y_data_new, x_data_new, feature_values_new, labels, d3, parent=new_parent,feature_value=fv)       


# In[36]:


# training_tree(y_data,x_data,feature_values,labels,D3,parent=None, feature_value="All")


# In[37]:


# D3.show()


# In[38]:


def test_record(d3, y_record, x_record, node):
    if node.is_leaf()== True: # it is leaf
        if node.data['label'] == y_record:
            return 1 #detect correctly
            
        else:
            return 0 #detect wrongly
        
    # It is not leaf and has to keep going
    feature_value = x_record[int(node.tag)] # get the feature value of the expected column
    flag=0
    for n in d3.children(node.identifier):
        if n.data['featurevalue'] == feature_value:
            #print(n)
            flag=1
            return test_record(d3, y_record, x_record, n)
    if flag == 0:
            if node.data['label'] == y_record:
                return 1 #detect correctly
            else:
                return 0


# In[39]:


#print(test_record(D3, y_data[10], x_data[10], D3.get_node(D3.root)))


# In[40]:


def test(d3, y_data, x_data,root):
    accuracy = 0
    
    for index in n.arange(n.shape(x_data)[0]):
        accuracy += test_record(d3, y_data[index], x_data[index], root)
    
        
    return accuracy*100/len(x_data)


# In[41]:


def xysplit(split, data):
    x_data = n.delete(data , split , axis = 1)
    y_data = data[:, split]
    return x_data, y_data

#Five Fold Cross Validation
def five_fold_cross_validation(data, split,fold):
    global counter
    global D3
    n.random.shuffle(data)#shuffle dataset
    subdata = n.array_split(data,indices_or_sections=fold,axis=0)#split data to 5 folds
#     print(subdata)
    accuracy=0
    for i in n.arange(fold):
#         print(subdata[i])
        x_data_test, y_data_test = xysplit(split, subdata[i])#Test data , 1/5 of all data(Each fold every time)
        subdata_copy = n.delete(subdata, i, axis=0)
        subdata_copy = n.concatenate(subdata_copy, axis=0)
        x_data_train, y_data_train = xysplit(split, subdata_copy) #Train data , 4/5 of all data
        D3 = Tree()
        counter = 0  
        #train ID3
        training_tree(y_data_train,x_data_train,feature_values,labels,D3,parent=None, feature_value="All")
        #Test ID3
        accuracy += test(D3, y_data_test, x_data_test, D3.get_node(D3.root))

    return accuracy/fold  

#Ten time 5-fold Cross Validation
accuracy=0
for i in n.arange(10):
    temp = five_fold_cross_validation(data, split=split_const,fold=5)
    accuracy += temp
    print(temp)
print(accuracy/10)
    

