
# coding: utf-8

# In[1]:


import numpy as n
from numpy import genfromtxt
import collections
from scipy import stats
from treelib import Node, Tree
import operator
from random import seed
from random import sample


# In[2]:


#Working on data, preprocessing


# In[6]:


split_const=0
file_name = input ("write the file address")


# In[22]:


data = genfromtxt((file_name+'.csv'),dtype="str",delimiter=',')


# In[23]:


print(data)
type(data)


# In[24]:


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



# In[25]:


x_data, y_data, labels, feature_values = xysplit(split_const, data)


# In[26]:


#test
print (y_data)
print(len(y_data))
print (x_data[:,3])


# In[27]:


#test
print (labels)
print (n.shape(labels))


# In[28]:


#test
print(feature_values)


# In[29]:


#Entropy of the Root (source or labels)
def source_entropy(y_data,labels):
    #probabilities
    probs = n.empty(len(labels[1]))
    for label in labels[1]:
        probs = n.append(probs, float(label/len(y_data)))
    #Entropy
    source_entropy = stats.entropy(probs,base=2)
    return source_entropy


# In[30]:


#test
print (source_entropy(y_data,labels))


# In[31]:


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


# In[32]:


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


# In[33]:


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


# In[34]:


# print (attribute_informationGain(x_data,y_data,feature_values,labels), type(attribute_informationGain(x_data,y_data,feature_values,labels)))


# In[35]:


def pureness(y_data):
    
    labels = n.unique(y_data)
    if len(labels) == 1:
         return labels[0]
    else:
         return 0
        
def most_frequent_label(y_data):
    counts = n.unique(y_data, return_counts=True, axis=0)
#     print(counts)
    index = n.argmax(counts[1])
    return counts[0][index]

# print(most_frequent_label(y_data))
    


# In[36]:


#We need to take m attributes randomly to compute their information gain 
#and choose the one with the most info gain
def generate_mrand_numbers(feature_values,size):
    # seed random number generator
    seed(1)
    # prepare a sequence
    list_= []
    for key, value in feature_values.items():
        list_.append(int(key))
    # select a subset without replacement
    subset = sample(list_, size)
    items = [str(i) for i in subset]
    return items


# In[37]:


#test
print(generate_mrand_numbers(feature_values,4))


# In[ ]:





# In[38]:


def training_tree(y_data,x_data,feature_values,labels,d3, parent,feature_value,size):
       
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
       if len(feature_values) > size:#Random forest!!!!!!!!!!!!!!!!!NOTICE HERE
           random_features = generate_mrand_numbers(feature_values,size) #select m numbers from M features!!!!!!!!!
           feature_values_random = feature_values.copy()
           for rf in random_features:
               feature_values_random.pop(rf)
           feature = attribute_informationGain(x_data, y_data, feature_values_random, labels)
           
       if len(feature_values) <= size:
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
               d3.create_node(str(feature),str(counter),data={'feature':feature,'featurevalue':feature_value, 'label': most_frequent_label(y_data)})
           else:
               d3.create_node(str(feature),str(counter),parent,data={'feature':feature ,'featurevalue':feature_value, 'label': most_frequent_label(y_data)})  
           new_parent = str(counter)
           counter +=1     
           for fv in feature_values[feature]:
               #preparing x_data, getting x_data with column of attribute=attribute value
               x_data_new_indecies = n.where(x_data[:,int(feature)] == fv)[0]
               x_data_new = n.take(x_data, indices=x_data_new_indecies, axis=0)
               #print(x_data_new)
               #preparing y_data
               y_data_new = n.take(y_data, x_data_new_indecies)
               training_tree(y_data_new, x_data_new, feature_values_new, labels, d3, parent=new_parent,feature_value=fv,size=size)       


# In[39]:


#training_tree(y_data,x_data,feature_values,labels,D3,parent=None, feature_value="All",size=3)


# In[40]:


#D3.show()
counter =0


# In[41]:


#Bootstraping for random forest

def bootstrap(B,subset_size,y_data,x_data):
    Trees = []
    seed(123)
    global counter
    for i in n.arange(B):
        indecies_new = n.random.choice(n.arange(len(y_data)), size=subset_size, replace=True)
        y_data_new = n.take(y_data, indecies_new)
        x_data_new = n.take(x_data, indecies_new, axis=0)
#         print(y_data_new, x_data_new)
        D3 = Tree()
        counter = 0  
        training_tree(y_data_new,x_data_new,feature_values,labels,D3, parent=None,feature_value="All",size=2)
        Trees.append(D3)
#         D3.show(
    return Trees    
       

#test

        


# In[42]:


def test_record_one_tree(d3, x_record, node):
    
    
    if node.is_leaf() == True: # it is leaf
#         print(node.data['label'])
        return node.data['label'] 
        
    # It is not the leaf and has to keep going
    feature_value = x_record[int(node.tag)] # get the feature value of the expected column
    flag=0
    for node_ in d3.children(node.identifier):
#         print(node_)
        if node_.data['featurevalue'] == feature_value:
            flag=1
            return test_record_one_tree(d3, x_record, node_)
            
    if flag!=1:
        return node.data['label'] 


# In[43]:


def test_record(trees,  x_record, y_record):
    labels = []
    for tree in trees:
        labels.append(test_record_one_tree(tree, x_record, tree.get_node(tree.root)))   
    final_label = most_frequent_label(labels)
#     print(final_label)
    if final_label == y_record:
        return 1
    else:
        return 0


# In[44]:


def test(trees,y_data, x_data):
    accuracy = 0
    
    for index in n.arange(n.shape(x_data)[0]):
        accuracy += test_record(trees, x_data[index], y_data[index])
        
    return accuracy*100/len(x_data)  


# In[45]:


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
        trees = bootstrap(10, 500, y_data_train,x_data_train)
        accuracy += test(trees, y_data_test, x_data_test)
    print(accuracy/fold)
    return accuracy/fold
    
#Ten time 5-fold Cross Validation
accuracy=0
for i in n.arange(10):
    accuracy += five_fold_cross_validation(data, split=split_const,fold=5)
print(accuracy/10)

