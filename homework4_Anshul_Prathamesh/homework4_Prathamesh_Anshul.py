from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

bestw = []
besth = []
losses = []
accuracies = []
minloss = 100
maxacc = -1

def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z), axis=0, keepdims=True))  # calculating softmax by formula


def relu(input):
    relu = input.copy()
    relu[relu < 0] = 0  # returning the relu output
    return relu


def relu_diff(input):
    # below 0 it is 0
    relu_diff = input.copy()
    relu_diff[relu_diff <= 0] = 0
    # Relu_diff[Relu_diff < 0] = 0
    relu_diff[relu_diff > 0] = 1
    # Relu_diff[Relu_diff >= 0] = 1
    return relu_diff  # return relu_diff for back propagation

def unpack(weightsAndBiases, hidden_num, H_Layers):
    # Unpack arguments
    Ws = []

    # Weight matrices    
    start = 0
    end = NUM_INPUT * hidden_num
    W = weightsAndBiases[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(H_Layers - 1):
        start = end
        end = end + hidden_num * hidden_num
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + hidden_num * NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = np.array(Ws[0]).reshape(hidden_num, NUM_INPUT)
    for i in range(1, H_Layers):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(hidden_num, hidden_num)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, hidden_num)

    # bias terms
    bias = []
    start = end
    end = end + hidden_num
    b = weightsAndBiases[start:end]
    bias.append(b)

    for i in range(H_Layers - 1):
        start = end
        end = end + hidden_num
        b = weightsAndBiases[start:end]
        bias.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bias.append(b)

    return Ws, bias


# defining forward propagation
def forward_prop(X, Y, weightsAndBiases, hidden_num, H_Layers):
    H_H = []
    H_Z = []

    Ws, bs = unpack(weightsAndBiases, hidden_num, H_Layers)
    h = X  # first layer

    for i in range(H_Layers):
        b = bs[i].reshape(-1, 1)
        Z = np.dot(Ws[i], h) + b
        H_Z.append(Z)
        h = relu(Z)
        H_H.append(h)

    H_Z.append(np.dot(Ws[-1], H_H[-1]) + bs[-1].reshape(-1, 1))

    yhat = softmax(np.dot(Ws[-1], H_H[-1]) + bs[-1].reshape(-1, 1))

    loss = np.sum(np.log(yhat) * Y)

    loss = (-1 / Y.shape[1]) * loss

    # Return loss, pre-activations, post-activations, and predictions
    return loss, H_Z, H_H, yhat


def back_prop(X, Y, weightsAndBiases, hidden_num, H_Layers):
    _, h_z, h_h, yhat = forward_prop(X, Y, weightsAndBiases, hidden_num, H_Layers)
    dJ_dWs = []  # Gradients w.r.t. weights
    dJ_dbs: list[Optional[Any]] = []  # Gradients w.r.t. biases

    Ws, bs = unpack(weightsAndBiases, hidden_num, H_Layers)
    G = yhat - Y

    for i in range(H_Layers, -1, -1):

        # Finding grads of b

        if i != H_Layers:  # at last layer G =Yhat-Y
            dh_dzs = relu_diff(h_z[i])
            G = dh_dzs * G

        dj_db_term = np.sum(G, axis=1) / Y.shape[1]
        dJ_dbs.append(dj_db_term)
        # Finding grads of w
        if i == 0:
            fst_layer = np.dot(G, X.T) / Y.shape[1]  # at first layer we multiply with input values X
            dJ_dWs.append(fst_layer)

        else:
            dJ_dWs.append(np.dot(G, h_h[i - 1].T) / Y.shape[1])  # G term multipled with privious term h thats why i-1

        G = np.dot(Ws[i].T, G)  # updated G for next layer

    dJ_dbs.reverse()  # make a list from last layer to first layer
    dJ_dWs.reverse()  # reverse to makes it first to end layer
    # Concatenate gradients
    return np.hstack([dJ_dW.flatten() for dJ_dW in dJ_dWs] + [dJ_db.flatten() for dJ_db in dJ_dbs])


def accuracy(yhat, y):
    yhat = yhat.T
    y = y.T
    # print("yhat.shape,y.shape",yhat.shape,y.shape)
    Yhat = np.argmax(yhat, 1)
    Y = np.argmax(y, 1)
    accuracy = 100 * np.sum(Y == Yhat) / y.shape[0]
    return accuracy


def Update_W_B(W, B, gradW, gradB, epsilon, alpha, trainY):
    for i in range(len(W)):
        W[i] = W[i] - (epsilon * gradW[i]) + (alpha * W[i] / trainY.shape[1])
        B[i] = B[i] - (epsilon * gradB[i])
    return W, B


def train(trainX, trainY, weightsAndBiases, hidden_num, H_Layers, epsilon, alpha):

    bp = back_prop(trainX, trainY, weightsAndBiases, hidden_num, H_Layers)
    gradW, gradB = unpack(bp, hidden_num, H_Layers)
    W, B = unpack(weightsAndBiases, hidden_num, H_Layers)
    W, B = Update_W_B(W, B, gradW, gradB, epsilon, alpha, trainY)

    weightsAndBiases = np.hstack([w.flatten() for w in W] + [b.flatten() for b in B])

    return weightsAndBiases


def sgd(train_X, train_Y, epochs, batch_size, weightsAndBiases, hidden_num, H_Layers, learning_rate, alpha,
        valid_X, valid_Y):
    
    global losses
    global accuracies
    
    print("epochs", epochs)
    global maxacc, minloss, besth, bestw
    for epoch in range(epochs):
        print("epoch", epoch)

        N_batches = int((len(train_X.T) / batch_size))

        init = 0
        end = batch_size
        for i in range(N_batches):
            mini_batch = train_X[:, init:end]

            y_mini_batch = train_Y[:, init:end]

            # mini_batch.shape, y_mini_batch.shape (784, 16) (10, 16)
            # batch_list[0][i].shape, batch_list[1][i].shape (784, 16) (10, 16)

            weightsAndBiases = train(mini_batch, y_mini_batch, weightsAndBiases, hidden_num, H_Layers,
                                                 learning_rate, alpha)
            init = end
            end = end + batch_size
            # if i % 10 == 0:  # sampled every 50 batches to get how the weights evolve
                # TRAJECT.extend(trajectory)  # stored all trej on mini batched to TRAJECT

        loss, _, _, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, hidden_num, H_Layers)
        acc = accuracy(yhat, valid_Y)
        
        if acc>maxacc:
            maxacc=acc
            minloss=loss
            besth = [epochs, batch_size, hidden_num, H_Layers, learning_rate, alpha]
            bestw=copy.deepcopy(weightsAndBiases)

        losses.append(loss)
        accuracies.append(acc)
        print("Loss on epoch: ", loss, "Accuracy: ", acc)

    return weightsAndBiases


def findBestHyperparameters(trainX, trainY, testX, testY):

    hidden_layers_list = [3]
    hidden_numbers_list = [81]
    mini_batch_size_list = [16]
    epsilon_list = [0.003]
    epochs_list = [70]
    alpha_list = [0.000001]

    change_order_index = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_index]
    trainY = trainY[:, change_order_index]

    index_values = np.random.permutation(trainX.shape[1])
    train_X = trainX[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_X = trainX[:, index_values[int(trainX.shape[1] * 0.8):]]
    train_Y = trainY[:, index_values[:int(trainX.shape[1] * 0.8)]]
    valid_Y = trainY[:, index_values[int(trainX.shape[1] * 0.8):]]

    for h_layers in hidden_layers_list:
        for hidden_num in hidden_numbers_list:
            for epochs in epochs_list:
                for batch_size in mini_batch_size_list:
                    for learning_rate in epsilon_list:
                        for alpha in alpha_list:

                            print("Hidden Layers=", h_layers, "Number of neurons=", hidden_num, "Batch_size=",
                                  batch_size)
                            print("Learning Rate=", learning_rate, "Epochs=", epochs, "Alpha=", alpha)

                            weightsAndBiases = initWeightsAndBiases(hidden_num, h_layers)

                            weightsAndBiases = sgd(train_X, train_Y, epochs, batch_size,
                                                                    weightsAndBiases, hidden_num, h_layers,
                                                                    learning_rate, alpha, valid_X, valid_Y)

                            _, _, _, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, hidden_num,
                                                                h_layers)

    b_h_layers = besth[3]
    b_hidden_num = besth[2]
    b_epochs = besth[0]
    b_batch_size = besth[1]
    b_learningrate = besth[4]
    b_alpha = besth[5]

    weightsAndBiases = initWeightsAndBiases(b_hidden_num, b_h_layers)
    weightsAndBiases=copy.deepcopy(bestw)                                           
    _, _, _, yhat = forward_prop(testX, testY, weightsAndBiases, b_hidden_num, b_h_layers)

    print("\nThe Best Hyper Parameters:  \nBest_Hidden_Layers:", b_h_layers, "\nBest_hidden_num: ", b_hidden_num,
          "\nBest_epochs: ", b_epochs, "\nBest_batch_size: ", b_batch_size)
    print("Best_learning_rate: ", b_learningrate, "\nBest_alpha: ", b_alpha)
    print("Best_accuracy on validation data :", maxacc)

    print("\nloss value on best hyperparameters: ", minloss)
    acc = accuracy(yhat, testY)
    print("\nAccuracy on Test data: ", acc)
    print("\n")
    saveval(unpack(bestw, b_hidden_num, b_h_layers)[0],'weights.txt')
    saveval(unpack(bestw, b_hidden_num, b_h_layers)[1],'biases.txt')

    return weightsAndBiases, h_layers, hidden_num


def initWeightsAndBiases(hidden_num, H_Layers):
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(hidden_num, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(hidden_num)
    bs.append(b)

    for i in range(H_Layers - 1):
        W = 2 * (np.random.random(size=(hidden_num, hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(hidden_num)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

def show_W1 (W):
    Ws,bs = unpack(W, besth[2], NUM_HIDDEN_LAYERS)
    W = Ws[1]
    n = int(besth[2] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 9, 9 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def saveval(nested_list, name):
    with open(name, 'w') as file:
        for inner_list in nested_list:
            line = ','.join(map(str, inner_list)) + '\n'
            file.write(line)

if __name__ == "__main__":
    # Load training data and normalizing
    X_tr = np.reshape(np.load("D:/CS 541 DL/HW4/fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    ytr = np.load("D:/CS 541 DL/HW4/fashion_mnist_train_labels.npy")
    
    # Load testing data and normalizing
    X_te = np.reshape(np.load("D:/CS 541 DL/HW4/fashion_mnist_test_images.npy"), (-1, 28 * 28)) / 255
    yte = np.load("D:/CS 541 DL/HW4/fashion_mnist_test_labels.npy")
    
    trainX, train_Y, testX, test_Y = X_tr.T, ytr, X_te.T, yte

    # onehot encoding
    trainY = np.zeros((train_Y.size, train_Y.max() + 1))
    testY = np.zeros((test_Y.size, test_Y.max() + 1))
    trainY[np.arange(train_Y.size), train_Y] = 1
    testY[np.arange(test_Y.size), test_Y] = 1
    trainY = trainY.T
    testY = testY.T

    weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN, NUM_HIDDEN_LAYERS)

    print("===========scipy.optimize.check_grad==========")
    print(scipy.optimize.check_grad(
        lambda wab:
        forward_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN, NUM_HIDDEN_LAYERS)[
            0],
        lambda wab: back_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN,
                              NUM_HIDDEN_LAYERS),
        weightsAndBiases))
    print("==============================================\n")
    wandb, H_Layers, hidden_num = findBestHyperparameters(trainX, trainY, testX, testY)
    change_order_idx = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_idx]
    trainY = trainY[:, change_order_idx]
    
    show_W1(wandb)
    
    x_axis = [i+1 for i in range(len(losses))]    
    plt.plot(x_axis, losses, '-b', label = 'Loss')
    plt.plot(x_axis, accuracies, '-g', label = 'Accuracy')
    plt.legend(loc="upper left")
    plt.show()