
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')


# In[12]:


class Neural_Network(object):
    
    def __init__(self, dimensionality, hidden_size=1, output_size = 1, learning_rate=0.5, dropout_rate=0.2, 
                 do_dropout=False, error_function='sum_of_squared',
                 do_regularize=False, regularization_rate=0.5, add_bias=True, use_nesterov_momentum=False, momentum_rate=0.9, do_random_seed=True, random_seed=1):
        self.dimensionality = dimensionality
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.do_dropout = do_dropout
        self.error_function = error_function
        self.do_regularize=do_regularize
        self.regularization_rate = regularization_rate
        self.add_bias = add_bias
        self.use_nesterov_momentum = use_nesterov_momentum
        self.momentum_rate = momentum_rate
        if do_random_seed == True:
            np.random.seed(random_seed)
        self.W1 = np.random.randn(dimensionality,hidden_size)*np.sqrt(2.0/dimensionality)
        self.W2 = np.random.randn(hidden_size,output_size)*np.sqrt(2.0/dimensionality)
        if add_bias == True:
            self.b1 = np.zeros((1,hidden_size))
            self.b2 = np.zeros((1,output_size))
        self.last_W1 = np.zeros(self.W1.shape)
        self.last_W2 = np.zeros(self.W2.shape)
        
        
    def sigmoid(self, x, clip=True):
        if clip==True:
            return np.clip(1/(1+np.exp(-x)),0.0001,0.9999)
        else:
            return 1/(1+np.exp(-x))
    
    def sigmoidDeriv(self, x, clip):
        return self.sigmoid(x, clip)*(1-self.sigmoid(x,clip))
    
    def forward(self, x, clip=True):
        self.z2 = np.dot(x, self.W1)
        if self.add_bias==True:
            self.z2 += self.b1
        self.a2 = self.sigmoid(self.z2, clip)
        if self.do_dropout==True:
            drop = (np.random.rand(*self.a2.shape) < self.dropout_rate) / self.dropout_rate 
            self.a2 *= drop
        self.z3 = np.dot(self.a2,self.W2)
        if self.add_bias==True:
            self.z3 += self.b2
        self.a3 = self.sigmoid(self.z3, clip)
        if self.do_dropout==True:
            drop = (np.random.rand(*self.a3.shape) < self.dropout_rate) / self.dropout_rate 
            self.a3 *= drop
        return self.a3
    
    @staticmethod
    def loss(yHat, y, error_function='sum_of_squared',axis=None, do_regularize=False, regularization_rate=0, W1=0, W2=0):
        err = 0
        if error_function == 'sum_of_squared':
            err = 0.5/yHat.shape[0]*np.sum(((y-yHat)**2),axis=axis)
        else:
            err = -1/yHat.shape[0]*np.sum((y*np.log(yHat)+(1-y)*np.log(1-yHat)),axis=axis)
        if do_regularize==True:
            return err + regularization_rate/(2*y.shape[0])*(sum(np.linalg.norm(w)**2 for w in W1)+sum(np.linalg.norm(w)**2 for w in W2))
        return err
    
        
    def delta_error(self, x, y):
        #dJda3 = 0
        #if self.error_function=='sum_of_squared':
        #    dJda3=(self.a3-y)
        #else:
        #    dJda3=(self.a3-y)/(self.a3*(1.0-self.a3))
        # da3dz3 = self.a3*(1.0-self.a3)
        # dJdz3 = dJda3*da3dz3
        dJdz3 = 0
        if self.error_function=='sum_of_squared':
            dJdz3=(self.a3-y)*self.a3*(1.0-self.a3)
        else:
            dJdz3=(self.a3-y)
        delta3 = dJdz3
        dJdz2 = delta3
        dz2da2 = self.W2.T
        da2dz2 = self.a2*(1.0-self.a2)
        delta2 = np.dot(dJdz2,dz2da2)*da2dz2
        return delta2, delta3
    
    def lossDeriv(self, x, y):
        delta2, delta3 = self.delta_error(x, y)
        dJdb1 = 0
        dJdb2 = 0
        if self.add_bias == True:
            dJdb1 = np.sum(delta2,axis=0)
            dJdb2 = np.sum(delta3,axis=0)
        dJdW2 = np.dot(self.a2.T,delta3)
        dJdW1 = np.dot(x.T,delta2)
        if self.do_regularize==True:
                dJdW2 += self.regularization_rate/x.shape[0]*self.W2
                dJdW1 += self.regularization_rate/x.shape[0]*self.W1
        
        return dJdW1, dJdW2, dJdb1, dJdb2
                
            
    def backprop(self,x,y):
        return self.lossDeriv(x,y)
    
    def learn_using_gradient_descent(self, x, y, current_iteration, print_loss_every, clip=True):
        if self.use_nesterov_momentum==True:
                delta_W1 = self.W1 - self.last_W1
                delta_W2 = self.W2 - self.last_W2
                self.W1 += self.momentum_rate*delta_W1
                self.W2 += self.momentum_rate*delta_W2
                self.last_W1 = self.W1
                self.last_W2 = self.W2 
        self.forward(x, clip)
        dJdW1, dJdW2, dJdb1, dJdb2 = self.backprop(x,y)
        self.W1 = self.W1 - self.learning_rate*dJdW1
        self.W2 = self.W2 - self.learning_rate*dJdW2
        if self.add_bias == True:
            self.b1 = self.b1 - self.learning_rate*dJdb1
            self.b2 = self.b2 - self.learning_rate*dJdb2
        if current_iteration % print_loss_every == 0:
            print("Iteration ", current_iteration, 'loss ', self.loss(self.a3, y, self.error_function,None, self.do_regularize, self.regularization_rate,self.W1, self.W2))
            
    def learn_using_stochastic_gradient_descent(self, x, y, mini_batch_size, current_epoch,clip=True,print_loss=True):
        combined_train_array = np.append(x,y,axis=1)
        random.shuffle(combined_train_array)
        x_train = combined_train_array[:,:-y.shape[1]]
        y_train = combined_train_array[:,-y.shape[1]:]
        for k in range(0, x_train.shape[0], mini_batch_size):
            if self.use_nesterov_momentum==True:
                delta_W1 = self.W1 - self.last_W1
                delta_W2 = self.W2 - self.last_W2
                self.W1 += self.momentum_rate*delta_W1
                self.W2 += self.momentum_rate*delta_W2
                self.last_W1 = self.W1
                self.last_W2 = self.W2   
            self.forward(x_train[k:k+mini_batch_size],clip)
            dJdW1, dJdW2, dJdb1, dJdb2 = self.backprop(x_train[k:k+mini_batch_size],y_train[k:k+mini_batch_size])    
            self.W1 = self.W1 - self.learning_rate*dJdW1
            self.W2 = self.W2 - self.learning_rate*dJdW2                    
            if self.add_bias == True:
                self.b1 = self.b1 - self.learning_rate*dJdb1
                self.b2 = self.b2 - self.learning_rate*dJdb2
        if print_loss==True:
            print("Epoch ", current_epoch, 'loss ', self.loss(self.forward(x,clip), y, self.error_function,None, self.do_regularize, self.regularization_rate,self.W1, self.W2))
    
    def predict(self, x_test, threshold):
        self.do_dropout=False
        arr_test=np.zeros((x_test.shape[0],1))
        yHat_test=self.forward(x_test,False)
        for i in range(0,x_test.shape[0]-1):
            if yHat_test[i] > threshold:
                arr_test[i]=1
        return arr_test
        
    def error(self, y_test, yHat_test):
        print(np.sum(np.subtract(y_test,yHat_test)==0)/y_test.shape[0])
        
    def accuracy(self, x_test, y_test):
        self.do_dropout = False
        correctly_classified=0
        yHat_test=self.forward(x_test,False)
        for i in range(y_test.shape[0]):
            if np.argmax(yHat_test[i])==np.argmax(y_test[i]):
                correctly_classified+=1
        return correctly_classified/y_test.shape[0]


# In[ ]:




