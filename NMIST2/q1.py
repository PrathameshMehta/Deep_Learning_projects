import numpy as np

def trainModel():
    return 1

def testModel():
    return 1

"""
def doCrossValidation (D, k, h):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    for fold in range(k):
        # Get all indexes for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indexes
        trainIdxs = idxs[list(set(allIdxs) - set(testIdxs)),:].flatten()
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    
    return np.mean(accuracies)
"""

def doDoubleCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k outer folds
    outer_idxs = np.random.permutation(allIdxs)
    outer_idxs = outer_idxs.reshape(k, -1)
    outer_accuracies = []
    for outer_fold in range(k):
        # Get all indexes for this outer fold
        outer_testIdxs = outer_idxs[outer_fold,:]
        # Get all the other indexes
        outer_trainIdxs = outer_idxs[list(set(allIdxs) - set(outer_testIdxs)),:].flatten()
        # Split the outer training data into k inner folds
        inner_idxs = np.random.permutation(outer_trainIdxs)
        inner_idxs = inner_idxs.reshape(k, -1)
        inner_accuracies = []
        for inner_Fold in range(k):
            # Get all indexes for this inner fold
            inner_testIdxs = inner_idxs[inner_Fold,:]
            # Get all the other indexes
            inner_trainIdxs = inner_idxs[list(set(outer_trainIdxs) - set(inner_testIdxs)),:].flatten()
            # Find best hyperparameter configuration from list of hyperparameter configurations H
            # best_accuracy is not set to zero to always allow atleast one accuracy to be appended to innerAccuracies
            best_accuracy = -1 
            best_config = None
            for hyperparameter_config in H:
                # Train the model on the inner training data
                model = trainModel(D[inner_trainIdxs], hyperparameter_config)
                # Test the model on the inner testing data
                config_accuracy = testModel(model, D[inner_testIdxs])
                if config_accuracy < best_accuracy:
                    continue
                best_accuracy = config_accuracy
                best_config = hyperparameter_config
            inner_accuracies.append(best_accuracy)
        # Use the best configuration for this outer fold to test on the outer testing data
        model = trainModel(D[outer_trainIdxs], best_config)
        outer_accuracies.append(testModel(model, D[outer_testIdxs]))
        
    return np.mean(outer_accuracies)
