#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np



class MultiLogisticReg():
    def __init__(self):
        #self.train_inputs = np.array(train_inputs)
        #self.train_labels = np.array(train_labels)
        #self.label_list = np.unique(train_labels)
        
    ## CREATE FIT
    
    def fit(self,train_inputs,train_labels):
        self.train_inputs = np.array(train_inputs)
        self.train_labels = np.array(train_labels)
        self.label_list = np.unique(train_labels) 
        
    
    def train(self, stepsize, n_steps, reg=0): #stepsize is the learning rate | n_steps is total iterations
        #add 1 to first column in the training features ALSO ON THE TEST SET
        new_X = np.insert(np.array(self.train_inputs),0, 1, axis=1)
        self.stepsize = stepsize
        self.n_steps = n_steps
        self.reg = reg
        #since there are m classes, we will perform m models: ONE VS REST 
        m_proba = np.zeros([new_X.shape[0] , len(self.label_list)]) #NxM (a cada fila le sacaremos argmax)
        self.W_total = np.zeros([new_X.shape[1] , len(self.label_list)]) #DxM (realmente d+1 por los biases)
        for m in range(len(self.label_list)):
            #initialize weigths (they have bias included)
            W = np.random.normal(0, 1, new_X.shape[1])
            y =  self.train_labels.copy()
            y = np.where(y==self.label_list[m] ,1 , 0)
            for epoch in range(self.n_steps):
                z = np.dot(new_X, W)
                y_hat = 1 / (1 + np.exp(-z))
                loss= (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean() #cross entropy
                gradient = np.dot(new_X.T, (y_hat - y)) / y.shape[0] + self.reg*W
                
                W -= self.stepsize * gradient
            self.W_total[:,m] = W #.reshape((new_X.shape[1],1))
            
    def predict(self, test_inputs, test_labels=None, test=False):
        self.test_inputs = np.array(test_inputs)
        self.test_labels = np.array(test_labels)
        
        new_X = np.insert(np.array(self.test_inputs),0, 1, axis=1)
        z_test=np.dot(new_X, self.W_total)
        y_test = 1 / (1 + np.exp(-z_test))
        final_predictions = np.argmax(y_test,axis=1)
        #ACCURACY*******************
        if test==True:
            
            accur = np.sum(final_predictions == self.test_labels)/self.test_labels.shape[0]
            return final_predictions, accur
        else:
            return final_predictions
    
    def hyper_tuning(self,x_val,y_val,x_test, arr_steps_size , arr_reg, n_steps): 
        #save the weights and extract the ones that gave the best acc and apply to test set
        self.accuracy_matrix = np.zeros(( len(arr_steps_size)*len(arr_reg),  3)) #steps|reg|acc
        
        #new_val = np.insert(np.array(x_val),0, 1, axis=1)
        new_test = np.insert(np.array(x_test),0, 1, axis=1)
        
        saved_weights =  np.empty((len(arr_steps_size)*len(arr_reg), new_test.shape[1], len(self.label_list)))
        num = 0
        for st in arr_steps_size:
            for r in arr_reg:
                model=MultiLogisticReg(self.train_inputs, self.train_labels)
                model.train(st, n_steps, r)
                pred, ac = model.predict(x_val , y_val, test=True)
                
                saved_weights[num] = model.W_total
                self.accuracy_matrix[num, 0] = st # learning_rate
                self.accuracy_matrix[num, 1] = r #regularizer
                self.accuracy_matrix[num, 2] = ac
                
                num+=1
                
        self.best_ac = np.argmax(self.accuracy_matrix[:,-1]) #this is an index
        best_W = saved_weights[self.best_ac] #weights extracted from the best accuracy (dxm: d+1xm)
        
        test_prediction = np.dot(new_test, best_W)
        y_test = 1 / (1 + np.exp(-test_prediction))
        final_predictions_test = np.argmax(y_test,axis=1)
        return final_predictions_test , self.accuracy_matrix[self.best_ac,-1]