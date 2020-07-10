#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
from os.path import isfile, join
import os
import math


# In[2]:


def sigmoid(z):

    s = 1/(1+np.exp(-z))

    return s


# In[3]:


def relu(x):
    return max(0.0, x)


# In[4]:


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)           

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters


# In[5]:


def linear_forward(A, W, b):

    Z = np.dot(W,A)+b
    cache = (A, W, b)
    
    return Z, cache


# In[6]:


def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
       
    elif activation == "relu":
       
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)
    #print(cache[1])
    return A, cache


# In[7]:


def forward_propagation(X, parameters):
    caches=[]
    A_prev=X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
    AL,cache=linear_activation_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    return AL, caches


# In[8]:


def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)      
    
    return cost


# In[9]:


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db


# In[10]:


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db =linear_backward(dZ,linear_cache) 
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db


# In[11]:


def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =linear_activation_backward(dAL, current_cache, 'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


# In[12]:


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads['db'+str(l+1)]
    return parameters


# In[13]:


def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for l in range(L):

        parameters["W" + str(l+1)] = parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] = parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]
        
    return parameters


# In[14]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    np.random.seed(seed)          
    m = X.shape[1]                 
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:,math.floor(m/mini_batch_size)*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,math.floor(m/mini_batch_size)*mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[15]:


def initialize_velocity(parameters):
    
    L = len(parameters) // 2 
    v = {}
    
    for l in range(L):
        
        v["dW" + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return v


# In[16]:


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    
    L = len(parameters) // 2 
    for l in range(L):

        v["dW" + str(l+1)] = beta*v['dW'+str(l+1)]+(1-beta)*grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta*v['db'+str(l+1)]+(1-beta)*grads['db'+str(l+1)]
       
        parameters["W" + str(l+1)] = parameters['W'+str(l+1)]-learning_rate*v['dW'+str(l+1)]
        parameters["b" + str(l+1)] = parameters['b'+str(l+1)]-learning_rate*v['db'+str(l+1)]
        
    return parameters, v


# In[17]:


def initialize_adam(parameters) :
    
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
    
        v["dW" + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)

    return v, s


# In[18]:


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         

    for l in range(L):
        
        v["dW" + str(l+1)] = beta1*v['dW'+str(l+1)]+(1-beta1)*grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta1*v['db'+str(l+1)]+(1-beta1)*grads['db'+str(l+1)]
        
        v_corrected["dW" + str(l+1)] = v['dW'+str(l+1)]/(1-beta1**(l+1))
        v_corrected["db" + str(l+1)] = v['db'+str(l+1)]/(1-beta1**(l+1))
        
        s["dW" + str(l+1)] = beta2*s['dW'+str(l+1)]+(1-beta2)*(grads['dW'+str(l+1)]**2)
        s["db" + str(l+1)] = beta2*s['dW'+str(l+1)]+(1-beta2)*(grads['db'+str(l+1)]**2)
       
        s_corrected["dW" + str(l+1)] = s['dW'+str(l+1)]/(1-beta2**(l+1))
        s_corrected["db" + str(l+1)] = s['db'+str(l+1)]/(1-beta2**(l+1))
        
        parameters["W" + str(l+1)] = parameters['W'+str(l+1)]-learning_rate*(v_corrected['dW'+str(l+1)]/(np.sqrt(s_corrected['dW'+str(l+1)])+epsilon))
        parameters["b" + str(l+1)] = parameters['b'+str(l+1)]-learning_rate*(v_corrected['db'+str(l+1)]/(np.sqrt(s_corrected['db'+str(l+1)])+epsilon))
        

    return parameters, v, s


# In[23]:


def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 32, beta = 0.9,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

    L = len(layers_dims)             
    costs = []                       
    t = 0                            
    seed = 10                        
    m = X.shape[1]                  
    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass 
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    for i in range(num_epochs):
        
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)

            cost_total += compute_cost(a3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / m
        
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


# In[20]:


plastic=os.path.abspath('plastic')
onlyfiles = [ f for f in listdir(plastic) if isfile(join(plastic,f)) ]
non_plastic=os.path.abspath('non_plastic')
onlyfiles1 = [ f for f in listdir(non_plastic) if isfile(join(non_plastic,f)) ]
train_images = np.zeros((len(onlyfiles)+len(onlyfiles1),384,512,3), dtype=np.uint8)
train_labels = []
for n in range(0, len(onlyfiles)):
    train_images[n] = cv2.imread( join(plastic,onlyfiles[n]),1 )
    train_labels.append(1)

for n in range(0, len(onlyfiles1)):
    train_images[len(onlyfiles)+n] = cv2.imread( join(non_plastic,onlyfiles1[n]),1 )
    train_labels.append(0)

train_labels=[train_labels]
train_labels_array=np.asarray(train_labels)
print(train_labels_array.shape)


# In[21]:


train_x_flatten = train_images.reshape(train_images.shape[0], -1).T
train_x = train_x_flatten/255
print(train_x.shape)


# In[24]:


# train 3-layer model
layers_dims = [train_x.shape[0], 5, 2, 1]
parameters = model(train_x, train_labels_array, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# In[ ]:


'''
end
'''

