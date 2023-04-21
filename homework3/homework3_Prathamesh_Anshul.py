from cmath import inf
import numpy as np
from numpy import random


def gradient_w_b(X, w, b, y, alpha):
    eZ = np.exp(np.dot(X.T, w) + b)  # find eZ
    eZ_mean = np.reshape(np.sum(eZ, axis=1), (-1, 1))  # find mean of eZ
    yhat = eZ / eZ_mean
    error = yhat - y  # find y^ - y
    grad_w = (np.dot(X, error)) / X.shape[1] + alpha * w
    grad_b = np.array([(-(np.sum((y - yhat), 0)) / X.shape[1])])
    return [grad_w, grad_b]

def Fce(X, y, w, b, alpha):  # cross entropy function
    eZ = np.exp(np.dot(X.T, w) + b)
    eZ_mean = np.reshape(np.sum(eZ, axis=1), (-1, 1))
    yhat = eZ / eZ_mean
    Fce = -np.sum(y * np.log(yhat)) / X.shape[1]
    return [Fce, yhat]

def sgd(epochs, learning_rate, alpha, mini_batch_size, X_train, Y_train, w, b):
    for i in range(epochs):
        batch = int((len(X_train.T) / mini_batch_size))
        start = 0
        end = mini_batch_size
        for j in range(batch):
            mini_batch = X_train[:, start:end]

            y_mini_batch = Y_train[start:end, :]

            grad_w, grad_b = gradient_w_b(mini_batch, w, b, y_mini_batch, alpha)

            w_values = w - (np.dot(learning_rate, grad_w))
            b_values = b - (np.dot(learning_rate, grad_b))

            start = end
            end = end + mini_batch_size
            w = w_values
            b = b_values
        # print("W shape in sgd: ", w.shape)
        fce_each_epoch, _ = Fce(X_train, Y_train, w, b, alpha)
        fce_each_epoch += (alpha / 2) * (np.sum(np.dot(w.T, w)))
        print("fCE for current {", i, "} epoch : ", fce_each_epoch)
    print("")
    return [fce_each_epoch, w, b]

def softmax_regression():
    # Load data
    X_tr = np.reshape(np.load("D:/CS 541 DL/HW3/fashion_mnist_train_images.npy"), (-1, 28 * 28))
    ytr = np.load("D:/CS 541 DL/HW3/fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("D:/CS 541 DL/HW3/fashion_mnist_test_images.npy"), (-1, 28 * 28))
    yte = np.load("D:/CS 541 DL/HW3/fashion_mnist_test_labels.npy")

    X = X_tr.T

    # Division of Dataset 80% Training Dataset and 20% validation dataset
    index_values = np.random.permutation(X.shape[1])

    X_train = X[:, index_values[:int(X.shape[1] * 0.8)]]
    X_valid = X[:, index_values[int(X.shape[1] * 0.8):]]

    Y_train = ytr[index_values[:int(X.shape[1] * 0.8)]]
    Y_valid = ytr[index_values[int(X.shape[1] * 0.8):]]

    Ytrain = np.zeros((Y_train.size, Y_train.max() + 1))
    Yvalid = np.zeros((Y_valid.size, Y_valid.max() + 1))
    Ytrain[np.arange(Y_train.size), Y_train] = 1
    Yvalid[np.arange(Y_valid.size), Y_valid] = 1

    # Structure Test Data
    Xtest = X_te.T
    Ytest = np.zeros((yte.size, yte.max() + 1))
    Ytest[np.arange(yte.size), yte] = 1

    # Random values of w and zeros for b
    w = np.random.randint(0, 1, int(X.shape[0]))
    w = np.atleast_2d(w).T
    b = np.zeros(10)

    epochs = [400, 600, 800, 1000]
    learning_rate = [3e-6, 4e-6, 5e-6, 6e-6]
    alpha = [2, 3, 4, 5]
    mini_batch_size = [100, 200, 400, 600]

    Fce_min = inf

    for m in epochs:
        print("Epochs: ", m)
        for n in learning_rate:
            for o in alpha:
                for p in mini_batch_size:
                    # Fce=Stochastic_gradient_descent(epochs = m,learning_rate = n,alpha = o,mini_batch_size = p,
                    # X_train,Y_train,w,b)
                    Fce_, w, b = sgd(m, n, o, p, X_train, Ytrain, w, b)
                    Fce_valid, _ = Fce(X_valid, Yvalid, w, b, o)
                    if Fce_valid < Fce_min:
                        print("Lowest f(Cross Entropy) for epochs =", m, "; Learning Rate = ", n, "; Alpha = ", o,
                              "; Mini Batch Size = ", p)
                        Min_FCE = Fce_valid
                        print("Min_FCE", Min_FCE)
                        hyper_parameters = [m, n, o, p]
                        Fce_min = Fce_valid

    b_epoch = hyper_parameters[0]
    b_learning_rate = hyper_parameters[1]
    b_alpha = hyper_parameters[2]
    b_minibatch = hyper_parameters[3]
    # Random values of
    w = np.random.randint(0, 1, int(X.shape[0]))
    w = np.atleast_2d(w).T
    b = np.zeros(10)
    Fce_train, weights, bias = sgd(b_epoch, b_learning_rate, b_alpha, b_minibatch, X_train,
                                   Ytrain, w, b)
    Fce_test, Yhat = Fce(Xtest, Ytest, weights, bias, b_alpha)
    Yhat = np.argmax(Yhat, 1)
    accuracy = 100 * np.sum(yte == Yhat) / X_te.shape[0]
    print("Min. value for fCE on Validation Set : ", Fce_min)
    print("Cross Entropy on the test set : ", Fce_test)
    print("Best hyper parameters: epochs (m)", m, " Learning rate", n, "alpha", o, " mini_batch_size", p)
    print("% accuracy : ", accuracy)
    return 0

softmax_regression()
