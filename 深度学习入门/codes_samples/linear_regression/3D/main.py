# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from linear_regression_3D import *
from mpl_toolkits.mplot3d import Axes3D


def show_data(x1, x2, y,w1=None, w2=None, b=None):
    ax = Axes3D(plt.figure())
    ax.scatter(x1,x2,y, marker='.')
    if w1 is not None and w2 is not None and b is not None:
        #plt.plot(x1,x2, w1*x1 + w2*x2 + b, c='red')
        Z=w1*x1 + w2*x2 + b
        ax.plot_trisurf(x1,x2,Z, linewidth=0.2, antialiased=True)
    plt.show()


# data generation
np.random.seed(272)
data_size = 300
x1 = np.random.uniform(low=1.0, high=10.0, size=data_size)
x2 = np.random.uniform(low=1.0, high=10.0, size=data_size)
y = x1 * 20 + x2 *5 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)

# train / test split
shuffled_index = np.random.permutation(data_size)
x1 = x1[shuffled_index]
x2 = x2[shuffled_index]
y = y[shuffled_index]

split_index = int(data_size * 0.7)
x1_train = x1[:split_index]
x2_train = x2[:split_index]
y_train = y[:split_index]

x1_test = x1[split_index:]
x2_train = x2[:split_index]
y_test = y[split_index:]

print('x1_train=',x1_train)
print('x2_train=',x2_train)
print('y_train=',y_train)
# visualize data
ax = Axes3D(plt.figure())
ax.scatter(x1_train,x2_train, y_train, marker='.')
plt.show()

# train the liner regression model
regr = LinerRegression(learning_rate=0.01, max_iter=500, seed=314)
regr.fit(x1_train,x2_train, y_train)
print('cost: \t{:.3}'.format(regr.loss()))
print('w1: \t{:.3}'.format(regr.w1))
print('w2: \t{:.3}'.format(regr.w2))
print('b: \t{:.3}'.format(regr.b))
show_data(x1, x2, y, regr.w1, regr.w2, regr.b)

# plot the evolution of cost
plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
plt.show()
