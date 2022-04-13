# -*- coding: utf-8 -*-

import numpy as np


class LinerRegression(object):

    def __init__(self, learning_rate=0.01, max_iter=100, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.w1 = np.random.normal(1, 0.1)
        self.w2 = np.random.normal(1, 0.1)
        self.b = np.random.normal(1, 0.1)
        self.loss_arr = []

    def fit(self,x1,x2,y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        print(40*'=')
        print('max_iter=',self.max_iter)
        for i in range(self.max_iter):
            self._train_step(i)
            self.loss_arr.append(self.loss())

    def _f(self, x1,x2,w1,w2,b):
        return x1 * w1 + x2 * w2 + b

    def predict(self, x1=None, x2=None):
        if x1 is None:
           x1 = self.x1
        if x2 is None:
           X2 = self.x2
        y_pred = self._f(x1,x2,self.w1,self.w2,self.b)
        return y_pred

    def loss(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict(self.x1,self.x2)
            #print(np.mean((y_true - y_pred)**2))
        return np.mean((y_true - y_pred)**2)

    def _calc_gradient(self):
        d_w1= np.mean((self.x1 * self.w1 + self.x2 * self.w2 + self.b - self.y) * self.x1)
        d_w2= np.mean((self.x1 * self.w1 + self.x2 * self.w2 + self.b - self.y) * self.x2)
        d_b = np.mean(self.x1 * self.w1 + self.x2 * self.w2 + self.b - self.y)
        return d_w1,d_w2,d_b

    def _train_step(self,index):
        d_w1,d_w2,d_b = self._calc_gradient()
        print('training step:',index)
        print('update w1=',self.w1)
        print('update w2=',self.w2)
        print('update b=',self.b)
        print('')
        self.w1 = self.w1 - self.lr * d_w1
        self.w2 = self.w2 - self.lr * d_w2
        self.b = self.b - self.lr * d_b
        return self.w1, self.w2, self.b
