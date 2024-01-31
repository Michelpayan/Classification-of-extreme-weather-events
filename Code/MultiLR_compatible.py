#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin



class MultiLogisticReg(BaseEstimator, ClassifierMixin):
    def __init__(self,stepsize=0.1, n_steps=100,reg=0):
        #self.train_inputs = np.array(train_inputs)
        #self.train_labels = np.array(train_labels)
        #self.label_list = np.unique(train_labels)
        
        #Parameters
        self.stepsize = stepsize
        self.n_steps = n_steps
        self.reg = reg
        
    ## CREATE FIT
    
    def fit(self,train_inputs,train_labels):
        self.train_inputs = np.array(train_inputs)
        self.train_labels = np.array(train_labels)
        self.label_list = np.unique(train_labels) 
        
        return self
        
    def predict(self, test):
        new_X_train = np.insert(np.array(self.train_inputs),0, 1, axis=1)
        self.test = np.insert(np.array(test),0, 1, axis=1)
        m_proba = np.zeros([new_X_train.shape[0] , len(self.label_list)]) #NxM (a cada fila le sacaremos argmax)
        W_total = np.zeros([new_X_train.shape[1] , len(self.label_list)])
        for m in range(len(self.label_list)):
            #initialize weigths (they have bias included)
            W = np.random.normal(0, 1, new_X_train.shape[1])
            y =  self.train_labels.copy()
            y = np.where(y==self.label_list[m] ,1 , 0)
            for epoch in range(self.n_steps):
                z = np.dot(new_X_train, W)
                y_hat = 1 / (1 + np.exp(-z))
                #loss= (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean() #cross entropy
                gradient = np.dot(new_X_train.T, (y_hat - y)) / y.shape[0] + self.reg*W
                
                W -= self.stepsize * gradient
            W_total[:,m] = W #.reshape((new_X.shape[1],1))
            
        #After training make the prediction
        z_test=np.dot(self.test, W_total)
        y_test = 1 / (1 + np.exp(-z_test))
        final_predictions = np.argmax(y_test,axis=1)
        
        return final_predictions
    
    def score(self, features, labels):
        pred = self.predict(features)
        accur = np.sum(pred == labels)/labels.shape[0]
        
        return accur
    
    def get_params(self, deep=True):
         
        return {
            'stepsize': self.stepsize,
            'reg': self.reg,
            'n_steps':self.n_steps
        }
       
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

