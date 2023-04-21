import numpy as np
import math   

miniBatchSizes = [16, 32, 64, 128]
learningRates = [0.001, 0.0001, 0.00001]
epochs = [10, 20, 30, 40]
alphas = [0.1, 0.01, 0.001]
        
def main():
    X_tr = np.reshape(np.load("D:/CS 541 DL/HW2/age_regression_Xtr.npy"),(-1, 48*48)) 
    Xtr = np.append(np.transpose(X_tr), [np.ones(np.transpose(X_tr).shape[1])], axis = 0)
    X_tr = Xtr[:, :4000]
    y_tr = np.load("D:/CS 541 DL/HW2/age_regression_ytr.npy")
    ytr = y_tr[:4000]
    X_val = Xtr[:, 4000:]
    y_val = y_tr[4000:]
    X_te = np.reshape(np.load("D:/CS 541 DL/HW2/age_regression_Xte.npy"),(-1, 48*48))
    X_te = np.append(np.transpose(X_te), [np.ones(np.transpose(X_te).shape[1])], axis = 0)
    yte = np.load("D:/CS 541 DL/HW2/age_regression_yte.npy")
    cost = []
    for miniBatchSize in miniBatchSizes:  
        for learning_rate in learningRates: 
            for epoch in epochs: 
                for alpha in alphas: 
                    wts = np.random.randn(X_tr.shape[0])               
                    for epoch in range(epoch):
                        m = X_tr.shape[1]
                        permutation =list(np.random.permutation(m))
                        X_tr = X_tr[:, permutation]
                        ytr = ytr[permutation]
                        inc = miniBatchSize
                        minibatches = []
                        mini_batch_complete = math.floor(m/miniBatchSize)
                        for i in range(0, mini_batch_complete):
                            mini_batch_X_tr = X_tr[:, i*inc:(i+1)*inc]
                            mini_batch_ytr = ytr[i*inc:(i+1)*inc]
                            mini_batch = [mini_batch_X_tr, mini_batch_ytr]
                            minibatches.append(mini_batch)
                            
                        if m % miniBatchSize != 0:
                            mini_batch_X_tr = X_tr[:, (inc*math.floor(m/inc)):]
                            mini_batch_ytr = ytr[(inc*math.floor(m/inc)):]
                            mini_batch = (mini_batch_X_tr, mini_batch_ytr)
                            minibatches.append(mini_batch)
                            
                        for minibatch in minibatches:
                            mini_batch_X_tr, mini_batch_ytr = minibatch
                            temp = np.append(wts[:-1], [0], axis = 0)
                            wts -= 2*learning_rate*((mini_batch_X_tr.dot(np.dot(np.transpose(mini_batch_X_tr), wts) - mini_batch_ytr)/miniBatchSize) + alpha*temp)
                            fmse = (np.mean(np.square(np.dot(np.transpose(X_val), wts) - y_val)))/2
                            cost.append(fmse)
        print("f_MSE = ", min(cost))
                            
if __name__ == "__main__":
    main()
                            