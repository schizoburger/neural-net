import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random


class Neural_Network(object):

    def __init__(self, dimensionality, hidden_size=1, output_size = 1,\
                    learning_rate=0.5, decay_learning_rate = False,\
                    learning_rate_decay_rate = 0.5,\
                    learning_rate_decay_step = 5,\
                    dropout_hidden_rate=0.5, do_dropout=False,\
                    error_function='sum_of_squared', do_regularize=False,\
                    regularization_rate=0.5, add_bias=True,\
                    use_nesterov_momentum=False, momentum_rate=0.9,\
                    do_random_seed=True, random_seed=1):

        self.dimensionality = dimensionality
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.decay_learning_rate = decay_learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_step = learning_rate_decay_step
        self.dropout_hidden_rate = 1 - dropout_hidden_rate
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


    @staticmethod
    def sigmoid(x):
        stable_sigmoid = np.vectorize(Neural_Network.__stable_sigmoid_function)
        return stable_sigmoid(x)

    @staticmethod
    def __stable_sigmoid_function(x):
        "Numerically-stable sigmoid function."
        if x >= 0:
            z = np.exp(-x)
            return 1/(1+z)
        else:
            z = np.exp(x)
            return z/(1+z)

    def sigmoidDeriv(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def forward(self, x, test_time=False):
        W1 = self.W1
        W2 = self.W2
        self.z2 = np.dot(x, W1)
        if self.add_bias==True:
            self.z2 += self.b1
        self.a2 = self.sigmoid(self.z2)
        if self.do_dropout==True:
            if test_time==False:
                self.dropout_hidden_mask = (np.random.rand(self.hidden_size)<self.dropout_hidden_rate)
                self.a2 *= self.dropout_hidden_mask
            else:
                W2 *= self.dropout_hidden_rate
        self.z3 = np.dot(self.a2, W2)
        if self.add_bias==True:
            self.z3 += self.b2
        self.a3 = self.sigmoid(self.z3)
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
        dz3da2 = self.W2.T
        da2dz2 = self.a2*(1.0-self.a2)
        if self.do_dropout==True:
            da2dz2*=self.dropout_hidden_mask
        delta2 = np.dot(dJdz3,dz3da2)*da2dz2
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

    def learn_using_gradient_descent(self, x, y, current_iteration, print_loss_every):
        if self.use_nesterov_momentum==True:
                delta_W1 = self.W1 - self.last_W1
                delta_W2 = self.W2 - self.last_W2
                self.W1 += self.momentum_rate*delta_W1
                self.W2 += self.momentum_rate*delta_W2
                self.last_W1 = self.W1
                self.last_W2 = self.W2
        self.forward(x)
        dJdW1, dJdW2, dJdb1, dJdb2 = self.backprop(x,y)
        self.W1 = self.W1 - self.learning_rate*dJdW1
        self.W2 = self.W2 - self.learning_rate*dJdW2
        if self.add_bias == True:
            self.b1 = self.b1 - self.learning_rate*dJdb1
            self.b2 = self.b2 - self.learning_rate*dJdb2
        if current_iteration % print_loss_every == 0:
            print("Iteration ", current_iteration, 'loss ', self.loss(self.a3, y, self.error_function,None, self.do_regularize, self.regularization_rate,self.W1, self.W2))
        if self.decay_learning_rate and current_iteration % self.learning_rate_decay_step == 0:
            self.learning_rate*=self.learning_rate_decay_rate

    def learn_using_stochastic_gradient_descent(self, x, y, mini_batch_size, current_epoch,print_loss=True):
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
            self.forward(x_train[k:k+mini_batch_size])
            dJdW1, dJdW2, dJdb1, dJdb2 = self.backprop(x_train[k:k+mini_batch_size],y_train[k:k+mini_batch_size])
            self.W1 = self.W1 - self.learning_rate*dJdW1
            self.W2 = self.W2 - self.learning_rate*dJdW2
            if self.add_bias == True:
                self.b1 = self.b1 - self.learning_rate*dJdb1
                self.b2 = self.b2 - self.learning_rate*dJdb2
        if print_loss==True:
            print("Epoch ", current_epoch, 'loss ', self.loss(self.forward(x), y, self.error_function,None, self.do_regularize, self.regularization_rate,self.W1, self.W2))
        if self.decay_learning_rate and current_epoch % self.learning_rate_decay_step == 0:
            self.learning_rate*=self.learning_rate_decay_rate

    def accuracy(self, x_test, y_test, threshold=None):
        multiple_outputs = y_test.shape[1] > 1
        correctly_classified=0
        yHat_test=self.forward(x_test,True)
        for i in range(y_test.shape[0]):
            if multiple_outputs == True:
                if np.argmax(yHat_test[i])==np.argmax(y_test[i]):
                    correctly_classified+=1
            else:
                prediction = yHat_test[i] > threshold
                if int(prediction)==int(y_test[i]):
                    correctly_classified += 1
        return correctly_classified/y_test.shape[0]
