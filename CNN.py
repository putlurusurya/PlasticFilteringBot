import numpy as np
import matplotlib.pyplot as plt

class utils:
    
    def relu(Z):
        A = np.maximum(0,Z) 
        return A
        
    def compute_cost(AL, Y):
    
        m = Y.shape[1]
        cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
        cost = np.squeeze(cost)      
        
        return cost
    def flatten(X):
        x_flatten = X.reshape(X.shape[0],-1).T
        return x_flatten

class convolve:
    
    def __init__(self,num_filters,filter_size,pad,stride,Input):
        self.num_filters=num_filters
        self.filter_size=filter_size
        self.Input=Input
        self.conv_filter=np.random.randn(filter_size,filter_size,Input.shape[-1],num_filters)/(filter_size*filter_size)
        self.bias=np.zeros((1,1,1,num_filters))
        self.pad=pad
        self.stride=stride
        
    def zero_pad(X, pad):

        X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0,))
        return X_pad
        
    def conv_single_step(self,a_slice_prev, W,b):

        s = a_slice_prev*W
        Z = s.sum()
        Z = Z+b
        return Z
    
    def conv_forward(self,Input1):
        
        print(Input1.shape)
        W=self.conv_filter
        b=self.bias
        
        (m, n_H_prev, n_W_prev, n_C_prev) = Input1.shape
        
        (f, f, n_C_prev, n_C) = W.shape
        
        stride =self.stride
        pad = self.pad
        
        n_H = int((n_H_prev-f+2*pad)/stride)+1
        n_W = int((n_W_prev-f+2*pad)/stride)+1
        
        Z = np.zeros((m,n_H,n_W,n_C))
        
        A_prev_pad = np.pad(Input1,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0,))
        
        for i in range(m):              
            a_prev_pad = A_prev_pad[i]              
            for h in range(n_H):           
                for w in range(n_W):           
                    for c in range(n_C):   
                        a_slice_prev = a_prev_pad[h*stride:h*stride+f, w*stride:w*stride+f]
                        weights = W[:,:,:,c]
                        biases = b[:,:,:,c]
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, weights,biases)
        
        return Z
    
    def conv_backward(self,dZ,learning_rate):
        
        W = self.conv_filter
        A_prev=self.Input
    
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        (f, f, n_C_prev, n_C) = W.shape
        
        stride = self.stride
        pad = self.pad
        
        (m, n_H, n_W, n_C) = dZ.shape
        
        dA_prev = np.zeros(A_prev.shape)                           
        dW = np.zeros(W.shape)
        db = np.zeros((1,1,1,n_C))
    
        A_prev_pad = self.zero_pad(A_prev,pad)
        dA_prev_pad = self.zero_pad(dA_prev,pad)
    
        for i in range(m):                       
            
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   
                for w in range(n_W):               
                    for c in range(n_C):           
                        
                        a_slice = a_prev_pad[h*stride:h*stride+f,w*stride+f:w*stride+f,:]
                        da_prev_pad[h*stride:h*stride+f,w*stride+f:w*stride+f,:] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            
            self.conv_filter = self.conv_filter - learning_rate*dW
            self.bias = self.bias -learning_rate*db
        return dA_prev
    
class pooling:
    
    def __init__(self,filter_size,stride,Input):
        self.filter_size=filter_size
        self.stride=stride
        self.Input=Input
        
    def pool_forward(self, mode = "max"):
        
        A_prev = self.Input
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        f = self.filter_size
        stride = self.stride
        
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        A = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):
            for h in range(n_H):                    
                for w in range(n_W):                     
                    for c in range (n_C):           
                        a_prev_slice = A_prev[i][h*stride:h*stride+f, w*stride: w*stride+f,c]
                        
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        return A
    
    def create_mask_from_window(x):
    
        mask = (np.max(x)==x)
        return mask
    
    def distribute_value(dz, shape):
    
        (n_H, n_W) = shape
        average = dz/(n_H+n_W)
        a = np.ones((n_H,n_W))*average
        return a
    def pool_backward(self,dA, mode = "max"):
    
        A_prev = self.Input
        
        stride = self.stride
        f = self.filter_size
        
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       
            
            a_prev = A_prev[i]
            
            for h in range(n_H):                  
                for w in range(n_W):               
                    for c in range(n_C):           
                        
                        if mode == "max":
                            
                            a_prev_slice = a_prev[h*stride:h*stride+f,w*stride:w*stride+f,c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i,h*stride:h*stride+f,w*stride:w*stride+f, c] += np.multiply(mask, dA[i, h, w, c])
                            
                        elif mode == "average":
                            
                            da = dA[i,h,w,c]
                            shape = (f,f)
                            dA_prev[i, h*stride:h*stride+f,w*stride:w*stride+f, c] += self.distribute_value(da,shape)
        
        return dA_prev

'''
class final_layer:
    
    def __init__(self,input_node):
        self.weight = np.random.randn(input_node,1)/input_node
        self.bias = np.zeros((1))
        
    
    def forward(self, input):
      
      self.last_input_shape = input.shape
  
      input = input.flatten()
      self.last_input = input
  
      input_len, nodes = self.weights.shape
  
      totals = np.dot(input, self.weights) + self.biases
      self.last_totals = totals
  
      exp = np.exp(totals)
      return exp / np.sum(exp, axis=0)
   
    def forward_propagation(self,Input):
        
        m = Input.shape[0]
        out = np.zeros((1,m))
        for i in range(m):
            X = Input[i]
            out[:,i]= self.forward(X)
        
        return out
        
        
        
    def backprop(self, d_L_d_out, learn_rate):
        
        for i, gradient in enumerate(d_L_d_out):
          if gradient == 0:
            continue
    
          # e^totals
          t_exp = np.exp(self.last_totals)
    
          # Sum of all e^totals
          S = np.sum(t_exp)
    
          # Gradients of out[i] against totals
          d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
          d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
    
          # Gradients of totals against weights/biases/input
          d_t_d_w = self.last_input
          d_t_d_b = 1
          d_t_d_inputs = self.weights
    
          # Gradients of loss against totals
          d_L_d_t = gradient * d_out_d_t
    
          # Gradients of loss against weights/biases/input
          d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
          d_L_d_b = d_L_d_t * d_t_d_b
          d_L_d_inputs = d_t_d_inputs @ d_L_d_t
    
          # Update weights / biases
          self.weights -= learn_rate * d_L_d_w
          self.biases -= learn_rate * d_L_d_b
     
          return d_L_d_inputs.reshape(self.last_input_shape)
    
    def back_propagation():
   '''     
     
class final_layer:
    
    def __init__(self,input_node):
        
        self.weight = np.random.randn(1,input_node)/input_node
        self.bias = np.zeros((1))
    
    def sigmoid(Z):
    
        A = 1/(1+np.exp(-Z))
        return A
    
    def forward_propagation(self,Input):
        
        self.last_input=Input
        Z = np.dot(self.weight,Input)+self.bias
        self.Z = Z
        AL=self.sigmoid(Z)
        
        return AL
    
    def sigmoid_backward(self,dA):
   
        Z = self.Z
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ
    
    def backward_propagation(self,AL,Y,learning_rate):
        
        m = self.last_input.shape[0]
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ = self.sigmoid_backward(dAL)
        dW = np.dot(dZ,self.last_input.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev = np.dot(self.weight.T,dZ)
        self.weight = self.weight-learning_rate*dW
        self.bias = self.bias - learning_rate*db
        return dA_prev
    
'''
 W1 : [4, 4, 3, 8]
 W2 : [2, 2, 8, 16]
'''

def model(train_x,train_y,learning_rate,epochs):
    
    costs=[]
    
    for i in range(epochs):
        #num_filters,filter_size,pad,stride,Input
        conv1=convolve(8,4,2,1,train_x)
        Z1=conv1.conv_forward(train_x)
        A1=utils.relu(Z1)
        
        pool1=pooling(8,8,A1)
        AP1=pool1.pool_forward()
        
        conv2=convolve(16,2,2,2,AP1)
        Z2=conv2.conv_forward(AP1)
        A2=utils.relu(Z2)
        
        pool2=pooling(4,4,A2)
        AP2=pool2.pool_forward()
        
        F1=utils.flatten(AP2)
        Final=final_layer(F1.shape[1])
        AL=Final.forward_propagation(F1)
        
        cost=utils.compute_cost(AL, train_y)
        print('cost after'+str(i)+'epochs is :'+str(cost))
        
        prev1=Final.backward_propagation(AL, train_y, learning_rate)
        prev2=pool2.pool_backward(prev1)
        prev3=conv2.conv_backward(prev2,learning_rate)
        prev4=pool1.pool_backward(prev3)
        prev5=conv1.conv_backward(prev4,learning_rate)
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return conv1,conv2,Final

def predict(train_x,train_y,conv1,conv2,Final):
    
    Z1=conv1.conv_forward(train_x)
    A1=utils.relu(Z1)
    
    pool1=pooling(8,8,A1)
    AP1=pool1.pool_forward()
    
    Z2=conv2.conv_forward(AP1)
    A2=utils.relu(Z2)
    
    pool2=pooling(4,4,A2)
    AP2=pool2.pool_forward()
    
    F1=utils.flatten(AP2)
    AL=Final.forward_propagation(F1)
    
    p=np.zeros(train_y.shape)
    
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.mean((p[0,:] == train_y[0,:]))))
    
    return p


            
            