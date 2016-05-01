import numpy as np
import matplotlib.pyplot as plt

from pyquantregForest import QuantileForest

def sin_func(X, c=1):
    X = np.asarray(X)
    return c*np.sin(X)

def sin_func(X, c=1):
    X = np.asarray(X)
    return c*X
    
np.random.seed(0)
    
# Sample creation
dim = 1 # Dimension
n_sample = 200
xmin, xmax = 0., 5.
X = np.linspace(xmin, xmax, n_sample).reshape((n_sample, 1))
y = sin_func(X).ravel() + np.random.randn(n_sample)
    
quantForest = QuantileForest().fit(X, y)

n_quantiles = 10
alpha = 0.9
x = np.linspace(xmin, xmax, n_quantiles)
x = 3.
quantiles = quantForest.computeQuantile(x, alpha, do_optim=True)
print quantiles
    
x = np.linspace(xmin, xmax, n_quantiles)
y_cdf = np.linspace(0., 30., 50)
CDFs = quantForest.compute_CDF(x, y_cdf)
print CDFs.shape

if dim == 1:
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(X, y, '.k')
    ax.plot(x, quantiles, 'ob')
    fig.tight_layout()
    plt.show()