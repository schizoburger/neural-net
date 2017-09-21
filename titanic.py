import neural_network as nn
import time
import titanic_data as td
import numpy as np

combined_train = td.combined.head(891)
sample = np.random.rand(len(combined_train)) < 0.5

x_train = combined_train[sample].values
y_train = np.array([td.targets[sample].values]).T

x_cv = combined_train[~sample].values
y_cv = np.array([td.targets[~sample].values]).T

x_test = td.combined[891:].values

print('combined: ', td.combined.shape, ', x_train: ', x_train.shape, ', x_cv: ', x_cv.shape, ', x_test: ', x_test.shape)
print('\n\n')

NN = nn.Neural_Network(dimensionality=x_train.shape[1],hidden_size=30,\
        output_size=1,learning_rate=0.001,\
        dropout_hidden_rate=0.5,do_dropout=False,\
        error_function='cross_entropy',do_regularize=False,\
        regularization_rate=1,add_bias=True,use_nesterov_momentum=False,\
        momentum_rate=0.9,do_random_seed=True,random_seed=1)

start_time = time.time()

for i in range(1000):
    NN.learn_using_gradient_descent(x=x_train,y=y_train,current_iteration=i,\
    print_loss_every=100,clip=True)

print('\n\n')
print("--- %s seconds ---" % (time.time() - start_time))
print('\n\n')
print('Accuracy on CV with gradient descent: ', NN.accuracy(x_cv,y_cv, 0.5))
print('\n\n')

NN = nn.Neural_Network(dimensionality=x_train.shape[1],hidden_size=30,\
        output_size=1,learning_rate=0.01,dropout_input_rate=0.2,\
        dropout_hidden_rate=0.5,do_dropout=False,\
        error_function='cross_entropy',do_regularize=False,\
        regularization_rate=1,add_bias=True,use_nesterov_momentum=False,\
        momentum_rate=0.9,do_random_seed=True,random_seed=1)

start_time = time.time()

for i in range(30):
    NN.learn_using_stochastic_gradient_descent(x=x_train,y=y_train,\
    mini_batch_size=10,current_epoch=i, clip=True,print_loss=True)

print('\n\n')
print("--- %s seconds ---" % (time.time() - start_time))
print('\n\n')
print('Accuracy on CV with stochastic gradient descent: ', NN.accuracy(x_cv,y_cv, 0.5))
