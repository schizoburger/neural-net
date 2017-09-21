print('Getting MNIST data from TensorFlow.\nThis may take a few seconds.')
from tensorflow.examples.tutorials.mnist import input_data
import neural_network as nn
import time
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_cv = mnist.validation.images
y_cv = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels


#NN = nn.Neural_Network(dimensionality=x_train.shape[1],hidden_size=30,\
#        output_size=10,learning_rate=0.1,dropout_input_rate=0.2,\
#        dropout_hidden_rate=0.5,do_dropout=False,\
#        error_function='cross_entropy',do_regularize=False,\
#        regularization_rate=1,add_bias=True,use_nesterov_momentum=False,\
#        momentum_rate=0.9,do_random_seed=True,random_seed=1)


#start_time = time.time()

#for i in range(50):
#    NN.learn_using_gradient_descent(x=x_train,y=y_train,current_iteration=i,\
#    print_loss_every=10,clip=True)

#print("\n\n--- %s seconds ---" % (time.time() - start_time))

#print('Accuracy on CV with gradient descent: ', NN.accuracy(x_cv,y_cv))

NN = nn.Neural_Network(dimensionality=x_train.shape[1],hidden_size=100,\
        output_size=10,learning_rate=0.1,\
        dropout_hidden_rate=0.5,do_dropout=True,\
        error_function='cross_entropy',do_regularize=False,\
        regularization_rate=1,add_bias=True,use_nesterov_momentum=False,\
        momentum_rate=0.9,do_random_seed=True,random_seed=1)

start_time = time.time()

for i in range(30):
    NN.learn_using_stochastic_gradient_descent(x=x_train,y=y_train,\
    mini_batch_size=10,current_epoch=i, clip=True,print_loss=True)

print("\n\n--- %s seconds ---" % (time.time() - start_time))
print('Accuracy on CV with stochastic gradient descent: ', NN.accuracy(x_cv,y_cv))
