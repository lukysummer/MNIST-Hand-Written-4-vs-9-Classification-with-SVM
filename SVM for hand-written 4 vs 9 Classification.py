# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:40:06 2017
@author: Luky
"""

import numpy as np 
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from datetime import datetime

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.
    You shouldn't need to touch this.
    '''    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size
        self.data = data
        self.targets = targets
        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.
        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.
        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  



class GD_Optimizer(object):
    # "object" : has lr=a, beta=b as input
    '''     A stochastic gradient descent optimizer with momentum     '''

    def __init__(self, lr, beta):
        ''' beta is the momentum '''
        self.lr = lr
        self.beta = beta
        self.vel = 0.0
        self.cnt = 1

    # Update parameters using GD with momentum and return the updated parameters
    def update_params(self, w, grad):
        '''compute gradient, compute new params with GD, set svm weights/bias to new params.'''
        # first iteration
        if(self.cnt == 1):
            # if w is scalar
            if(isinstance(w, float)):
                length = 1
            # if w is vecotr
            else:
                length= len(w)
            self.vel = [self.vel]*length
            self.cnt = 2
        
        self.vel = np.subtract(np.dot(self.beta,self.vel), np.dot(grad, self.lr))
        new_w = np.add(w, self.vel)
        
        return new_w
    

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    
    for _ in range(steps):
        new_grad = func_grad(w)
        w = optimizer.update_params(w, new_grad)
        w_history.append(w)
        
    return w_history


class SVM(object):
    '''   
    A Support Vector Machine   
    '''
    def __init__(self, c, w):
        self.c = c
        self.w = np.array(w)
 
       
    def hinge_loss(self, X, y):
        '''
        --Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).
        --Returns a length-n vector containing the hinge-loss per data point.
        '''
        # adding bias as column of 1 to x
        bias = np.ones([len(X),1])
        X = np.concatenate((X,bias), axis=1)
           
        hinge_losses = []
        cnt = 0
        for i in range(len(y)):
            dist = 1 - (y[i] * (np.dot([X[i]], np.transpose(self.w))[0]))

            if(dist > 0):
                hinge_losses.append(dist)
                if(dist <1):
                    cnt += 1
            else:
                hinge_losses.append(0)
                
        return hinge_losses

    
    def grad(self, X, y):
        '''
        --Compute the gradient of the SVM objective for input data X (shape (n, m)) + target y (shape (n,))
        --Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        hinge_losses = SVM.hinge_loss(self, X, y)
        
        # adding bias as column of 1 to x
        bias = np.ones([len(X),1])
        X = np.concatenate((X,bias), axis=1)
        
        # excluding bias weights
        # w_no_bias = self.w[0:np.shape(self.w)[0]-1]       
        
        #grad = self.w # regularization term
        grad = self.w[0:np.shape(self.w)[0]-1] # regularization term
        # (above: not including bias weight (last element) to regularization term)
        grad = np.insert(grad,len(grad),0)  # not including bias weight to regularization term
        grad_subterm = np.zeros(np.shape(self.w)[0])

        for i in range(len(hinge_losses)): 
            if(hinge_losses[i] > 0):
                grad_subterm = np.add(grad_subterm, np.dot(((-1)*y[i]), X[i])) 
        
        # C/N penalty multiplier
        C_over_N = self.c/len(hinge_losses)
        grad = np.add(grad, np.dot(C_over_N, grad_subterm))
        
        return grad # shape : (785,)


    def classify(self, X):
        '''
        --Classify new input data matrix (shape (n,m)).
        --Returns the predicted class labels (shape (n,))
        '''
        # adding bias as column of 1 to x
        bias = np.ones([len(X),1])
        X = np.concatenate((X,bias), axis=1)
        
        # Classify points as +1 or -1 as SIGN of y
        labels = []
        for i in range(len(X)):
            predict = np.dot([X[i]], self.w)[0]
            if(predict > 0):
                labels.append(1)
            else:
                labels.append(-1)
        
        return labels


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    feature_count = np.shape(np.array(train_data))[1]
    # initialize weights w: feature_count+1 for the bias   
    w = np.random.normal(0.0, 0.1, feature_count+1) #shape : (m+1,)
       
    for _ in range(iters):
        '''compute gradient, compute new params with GD, set svm weights/bias to new params.'''

        batch_sampler = BatchSampler(train_data, train_targets, batchsize)
        train_batch, train_label_batch = batch_sampler.get_batch()
        
        SVM_opt = SVM(penalty, w)
        
        new_grad = SVM_opt.grad(train_batch, train_label_batch)
        #new_grad = np.insert(new_grad,len(new_grad),0) 
        # calculating grad does not involve bias. inserting "0" at the end to represent this 0 bias weight for grad
        #new_grad = [np.concatenate((new_grad,[w[0][np.shape(w)[1]-1]]))]    
        w = optimizer.update_params(w, new_grad)

    SVM_opt = SVM(c=penalty, w=w)
    hinge_loss = np.average(SVM_opt.hinge_loss(train_data, train_targets))
    
    return w


# 2.3
def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4)) # digit 4: y = 1
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9)) # digit 9: y = -1

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8)) # 80% training set, 20% test set

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    print("")
    
    return train_data, train_targets, test_data, test_targets



if __name__ == '__main__':
    optimizer_beta_0 = GD_Optimizer(1.0, 0.0)
    optimizer_beta_0_9 = GD_Optimizer(1.0, 0.9)
    
    w_history_beta_0 = optimize_test_function(optimizer_beta_0)
    w_history_beta_0_9 = optimize_test_function(optimizer_beta_0_9)
    
    line_up, = plt.plot(w_history_beta_0,'b', label = 'β = 0.0')
    line_down, = plt.plot(w_history_beta_0_9,'r', label = 'β = 0.9')
    plt.xlabel('steps (t)')
    plt.ylabel('w_t')
    plt.title('w_t for 200 time-steps for β = 0.0 and β = 0.9')
    plt.legend(handles=[line_up, line_down])
    plt.show()
    print("")
    
    

    # 4 vs. 9
    train_data, train_label, test_data, test_label = load_data()
    
    
    print("4 vs. 9 Digit Classification Results\n\n-------------------------------\n\n")
    print("First Model:")
    
    optimizer_beta_0 = GD_Optimizer(0.05, 0.0)
    w_beta_0 = optimize_svm(train_data, train_label, 1.0, optimizer_beta_0, 100, 500)
    
    SVM_opt_0 = SVM(1.0, w_beta_0)
    
    predictions_train_0 = SVM_opt_0.classify(train_data)
    print("")
    print('training accuracy = {}'.format((predictions_train_0 == train_label).mean()))
    
    predictions_test_0 = SVM_opt_0.classify(test_data)
    print('test accuracy = {}'.format((predictions_test_0 == test_label).mean()))
    
    # Note: plt.cmap show SMALLER numbers as DARKER values
    w_square_0 = np.zeros([28,28]); cnt = 0
    for i in range(28):
        for j in range(28):
            w_square_0[i][j] = w_beta_0[cnt]
            cnt += 1

    print("Training loss: ", np.average(SVM_opt_0.hinge_loss(train_data, train_label)))
    print("Test loss: ", np.average(SVM_opt_0.hinge_loss(test_data, test_label)))
    print("")
    plt.imshow(w_square_0, cmap='gray')
    plt.title('Plot of 4 vs. 9 SVM weights for momentum = 0.0')
    plt.show()
    print(""); print("")

    
    print("Second Model:")
    
    optimizer_beta_0_1 = GD_Optimizer(0.05, 0.1)
    w_beta_0_1 = optimize_svm(train_data, train_label, 1.0, optimizer_beta_0_1, 100, 500)
    
    SVM_opt_0_1 = SVM(1.0, w_beta_0_1)
    
    w_square_0_1 = np.zeros([28,28]); cnt = 0
    for i in range(28):
        for j in range(28):
            w_square_0_1[i][j] = w_beta_0_1[cnt]
            cnt += 1
    
    predictions_train_0_1 = SVM_opt_0_1.classify(train_data)
    print("")
    print('training accuracy = {}'.format((predictions_train_0_1 == train_label).mean()))
    
    predictions_test_0_1 = SVM_opt_0_1.classify(test_data)
    print('test accuracy = {}'.format((predictions_test_0_1 == test_label).mean()))
    print("")
    print("Training loss: ", np.average(SVM_opt_0_1.hinge_loss(train_data, train_label)))
    print("Test loss: ", np.average(SVM_opt_0_1.hinge_loss(test_data, test_label)))
    print("")
    plt.imshow(w_square_0_1, cmap='gray')
    plt.title('Plot of 4 vs. 9 SVM weights for momentum = 0.1')
    plt.show()
    