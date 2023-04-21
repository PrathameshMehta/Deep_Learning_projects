import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A, B) - C

def problem_1c (A, B, C):
    return A * B + C.T

def problem_1d (x, y):
    return np.dot(x.T, y)

def problem_1e (A, x):
    return np.linalg.solve(A, x)

def problem_1f (A, i):
    return np.sum(A[i, ::2])

def problem_1g (A, c, d):
    indices = np.nonzero((A >= c) & (A <= d))
    S = A[indices]
    return np.mean(S)

def problem_1h (A, k):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    indices_of_eigenvalues = np.argsort(np.abs(eigenvalues))
    indices_of_largest_eigenvalues = indices_of_eigenvalues[-k:]
    return eigenvectors[:, indices_of_largest_eigenvalues]

def problem_1i (x, k, m, s):
    z = np.ones(x.shape)
    I = np.eye(x.shape[0])
    mean = x + m*z
    covariance = s*I
    return np.random.multivariate_normal(mean, covariance, k).T

def problem_1j (A):
    n = A.shape[1]
    return A[:, np.random.permutation(n)]

def problem_1k (x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    return [(x[i] - x_mean) / x_std for i in range(len(x))]

def problem_1l (x, k):
    i= np.reshape(x,(x.shape[0],1))
    return np.repeat(i,k,1)

def problem_1m (X, Y):
    a=np.repeat(np.atleast_3d(X), Y.shape[0], 2)
    return np.sum(np.square(a - Y.T),1)

def problem_1n (matrices):
    count = 0
    for i in range(len(matrices)-1):
        rows_of_first_matrix = matrices[i].shape[0]
        columns_of_first_matrix = matrices[i].shape[1]
        columns_of_second_matrix = matrices[i+1].shape[1]
        count += rows_of_first_matrix * columns_of_first_matrix * columns_of_second_matrix
    return count*columns_of_first_matrix

def linear_regression (X_tr, y_tr):
    X_tr = np.append(np.transpose(X_tr), [np.ones(np.transpose(X_tr).shape[1])], axis = 0)
    w_hat = np.linalg.inv(X_tr.dot(np.transpose(X_tr))).dot(X_tr.dot(y_tr))
    w = w_hat[:-1]
    b = w_hat[-1]
    return w,b
  

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w, b = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
    y_hat = np.dot(X_tr,w)+b*np.ones((1,len(ytr)))
    c_tr = 1/(2*len(ytr))*(np.sum(np.square(y_hat-ytr)))
    print(c_tr)
    y_hat = np.dot(X_te,w)+b*np.ones((1,len(yte)))
    c_te = 1/(2*len(yte))*(np.sum(np.square(y_hat-yte)))
    print(c_te)