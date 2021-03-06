# Neural Net

Implemented a neural network with one hidden layer and tested it on the Titanic from Kaggle and MNIST datasets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install beforehand:

Minimum requirements:
```
Python 3.5
NumPy
Pandas
```
For testing my implementation of a neural network on MNIST and
comparing it with Tensorflow:
```
Tensorflow
```

## Description of files
* neural_network.py contains my implementation of a neural network with one hidden layer. Apart from the vanilla settings, there are also cross entropy loss function, stochastic gradient descent, L2 regularization, dropout, learning rate decay and Nesterov momentum.
* titanic_data.py processes the Titanic dataset from Kaggle and engineers new features.
* titanic.py tests neural_networks.py on the processed data from titanic_data.py.
* mnist.py tests neural_networks.py on the MNIST dataset downloaded with Tensorflow module.
* mnist_tf_and_my_nn_comparison.ipynb compares the performance of neural_network.py with a neural network built with Tensorflow on the MNIST dataset.
* gradient_descent.tex contains the code for my article on Gradient Descent algorithm.


## Acknowledgments

* Titanic dataset from Kaggle processed with the following tutorial: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
* Inspiration for my implementation of a neural network:
http://neuralnetworksanddeeplearning.com/chap1.html
http://cs231n.github.io/
