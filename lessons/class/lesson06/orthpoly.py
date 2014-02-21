from pandas import *
import numpy as np
from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation as cv
import sklearn.metrics as metrics
from sklearn.svm import l1_min_c
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
import scipy.linalg as la
from math import pi
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrix
import re
import os
import math

sin_data = DataFrame({'x' : np.linspace(0, 1, 101)})
sin_data['y'] = np.sin(2 * pi * sin_data['x']) + np.random.normal(0, 0.1, 101)

x = sin_data['x']
y = sin_data['y']
Xpoly = dmatrix('C(x, Poly)')
Xpoly1 = Xpoly[:, :2]
Xpoly3 = Xpoly[:, :4]
Xpoly5 = Xpoly[:, :6]
Xpoly25 = Xpoly[:, :26]

polypred1 = sm.OLS(y, Xpoly1).fit().predict()
polypred3 = sm.OLS(y, Xpoly3).fit().predict()
polypred5 = sm.OLS(y, Xpoly5).fit().predict()
polypred25 = sm.OLS(y, Xpoly25).fit().predict()

"""
"""

plt.figure(figsize=(10, 8))
fig, ax = plt.subplots(2, 2, sharex = True, sharey = True)
fig.subplots_adjust(hspace = 0.0, wspace = 0.0)
fig.suptitle('Polynomial fits to noisy sine data', fontsize = 16.0)

# Iterate through panels (a), model predictions (p), and the polynomial 
# degree of the model (d). Plot the data, the predictions, and label
# the graph to indicate the degree used.
for a, p, d in zip(ax.ravel(), [polypred1, polypred3, polypred5, polypred25],
      ['1', '3', '5', '25']):
    a.plot(x, y, '.', color = 'steelblue', alpha = 0.5)
    a.plot(x, p)
    a.text(.5, .95, 'D = ' + d, fontsize = 12,
           verticalalignment = 'top',
           horizontalalignment = 'center',
           transform = a.transAxes)
    a.grid()

# Alternate axes that have tick labels to avoid overlap.
plt.setp(fig.axes[2].get_yaxis().get_ticklabels(), visible = False)
plt.setp(fig.axes[3].get_yaxis(), ticks_position = 'right')   
plt.setp(fig.axes[1].get_xaxis(), ticks_position = 'top')
plt.setp(fig.axes[3].get_xaxis().get_ticklabels(), visible = False)
plt.show()


"""
a home-rolled function to build the orthonormal-polynomial basis functions from a vector. This will test we understand what Poly does. If you check it, you'll find this function and Poly provide the same results.
"""
def poly(x, degree):
    '''
    Generate orthonormal polynomial basis functions from a vector.
    '''
    xbar = np.mean(x)
    X = np.power.outer(x - x.mean(), arange(0, degree + 1))
    Q, R = la.qr(X)
    diagind = np.subtract.outer(arange(R.shape[0]), arange(R.shape[1])) == 0
    z = R * diagind
    Qz = np.dot(Q, z)
    norm2 = (Qz**2).sum(axis = 0)
    Z = Qz / np.sqrt(norm2)
    Z = Z[:, 1:]
    return Z
