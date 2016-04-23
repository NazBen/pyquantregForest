# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.optimize import fmin_cobyla, fmin_slsqp, basinhopping
import sys
import pandas as pd
sys.path.append("../configuration/")
from configuration import shape
import cma
import dask.array as da


class QuantileForest():
    """Quantile Regresion Random Forest.
      This class can build random forest using Scikit-Learn and compute
      conditional quantiles.

      Parameters
      ----------
      inputSample : array
        Input samples used in data

      outputSample : array
        Output samples used in data

      n_estimators : int, optional (default=50)
        The number of trees in the forest.

      max_leaf_nodes : int or None, optional (default=max(10, len(outputSample)/100))
        Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. If not None then max_depth will be ignored. Note: this parameter is tree-specific.

      n_jobs : int, optional (default=4)
        The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.

      numPoints : int, optional (default=0)
        The size of the vector used to determines the quantile. If 0, the vector use is the outputSample.

      outputSample : string, optional (default="Cobyla")
        Name of the Optimisation method to find the alpha-quantile (if the option is chosen in the computeQuantile method). Only "Cobyla" and "SQP" are available.

      random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    """

    def __init__(self,
                 inputSample,
                 outputSample,
                 n_estimators=50,
                 min_samples_leaf=None,
                 n_jobs=4,
                 n_points=0,
                 optMethod="Cobyla",
                 random_state=None,
                 *more_args,
                 **more_kwargs):

        self.setSample(inputSample, outputSample)
        self.computeForest(n_estimators, min_samples_leaf, n_jobs,
                           random_state, *more_args, **more_kwargs)

        self.setPrecisionOfCDF(n_points)
        self._optMethod = optMethod  # Optimisation method

    def _checkRegressor(self, X):
        """

        """
        n = X.shape[0]  # Number of sample
        try:  # Works if X is an array
            d = X.shape[1]  # Dimension of the array
            if d != self._dimension:  # If the dimension is not correct
                if n == self._dimension:  # There is one sample of d dimension
                    d = n
                    n = 1
                else:  # Error
                    raise ValueError("X dimension is different from forest \
                    dimension : %d (X) != %d (forest)" % (d, self._dimension))
        except:  # Its a vector
            d = 1
            if d != self._dimension:  # If the dimension is not correct
                if n == self._dimension:  # There is one sample of d dimension
                    d = n
                    n = 1
                else:  # Error
                    raise ValueError("X dimension is different from forest \
                    dimension : %d (X) != %d (forest)" % (d, self._dimension))

        if (n > 1) & (d == 1):
            X.resize(n, 1)

        return X, n

    def setInputLimits(self, limits=None):
        """
        Parameter
        ----------
        limits
        """
        if not limits:
            limits = [[-np.inf]*self._dimension, [np.inf]*self._dimension]
        elif type(limits) is np.array:
            limits = limits.tolist()
        assert len(limits[0]) == self._dimension, ValueError(
                "Limits dimension different from input sample dimension: \
                %d(limits) != %d (input)" % (len(limits[0]), self._dimension))
        self._inputLimits = limits

    def computeQuantile(self, X, alpha, doOptim=True, verbose=False,
                        doSaveCDF=False):
        """
        Compute the conditional alpha-quantile.
        """

        # If the inputs are scalars
        if type(alpha) is int or type(alpha) is float:
            alpha = [alpha]

        if type(X) is int or type(X) is float:
            X = [X]

        # Converting to array for convenience
        X = np.array(X)
        alpha = np.array(alpha)

        # Number of quantiles to compute
        X, numRegressor = self._checkRegressor(X)

        numAlpha = len(alpha)  # Number of probabilities

        # Matrix of computed quantiles
        quantiles = np.zeros((numRegressor, numAlpha), dtype=np.float)

        if doSaveCDF:
            self._CDF = np.empty((self._yCDF.size, numRegressor))

        # Nodes of the regressor in all the trees
        nodesOfRegressor = self._forest.apply(X).transpose()

        for k in range(numRegressor):  # For each regressor
            # Set to 1 only the samples in the same nodes the regressor,
            # Shape : Matrix (numSample * numTree)
            weightPerTree = (self._nodesOfSamples ==
                             nodesOfRegressor[:, k]) * 1.

            # The proportion in each node
            # Shape : Matrix (numSample * numTree)
            normedWeightPerTree = weightPerTree / weightPerTree.sum(axis=0)

            # The weight of each sample in the trees
            # Shape : Vector (numSample * )
            weight = normedWeightPerTree.sum(axis=1) / self._numTree

            if doOptim:  # Computation is by optimisation
                # The starting point is taking by computing the alpha quantile
                # of the non-null weights.
                y0 = np.percentile(self._outputSample[
                                   weight != 0], alpha * 100)

                for i, alphai in enumerate(alpha):
                    if self._optMethod == "Cobyla":
                        quantiles[k, i] = fmin_cobyla(self._optFunc,
                                                      y0[i],
                                                      [self._ieqFunc],
                                                      args=(weight, alphai),
                                                      disp=verbose)

                    elif self._optMethod == "SQP":
                        epsilon = 1.E-1 * abs(y0[i])
                        quantiles[k, i] = fmin_slsqp(self._optFunc,
                                                     y0[i],
                                                     f_ieqcons=self._ieqFunc,
                                                     args=(weight, alphai),
                                                     disp=verbose,
                                                     epsilon=epsilon)

                    else:
                        raise Exception("Unknow optimisation method %s" %
                                        self._optMethod)
            else:
                CDF = self._infYY.dot(weight).ravel()  # Compute the CDF
                quantiles[k, :] = [self._yCDF.values[CDF >= alphai][0]
                                   for alphai in alpha]
                if doSaveCDF:
                    self._CDF[:, k] = CDF

        if numRegressor == 1 and numAlpha == 1:
            return quantiles[0][0]
        elif numRegressor == 1 or numAlpha == 1:
            return quantiles.ravel()
        else:
            return quantiles

    def _optFunc(self, yi, w, alpha):
        """

        """
        alphai = w[self._outputSample.values <= yi].sum()
        return abs(alphai - alpha)**2

    def _ieqFunc(self, yi, w, alpha):
        """

        """
        alphai = w[self._outputSample.values <= yi].sum()
        return alphai - alpha

    def findMinQuantile(self, alpha, verbose=False, seed=None):
        """

        """
        x0 = self._inputSample.values.mean(axis=0)
        sigma0 = self._inputSample.values.std()
        if self._dimension > 1:
            self._inputLimits = [[-1., -1., -1.], [1., 1., 1.]]
            args = {'bounds': self._inputLimits, "seed": seed}
            es = cma.CMAEvolutionStrategy(x0, sigma0, args)
            es.optimize(self.computeQuantile, args=([alpha]))
            res = es.result()
        else:
            minimizer_kwargs = {"method": "BFGS", "args": (alpha)}
            res = basinhopping(self.computeQuantile, x0,
                               minimizer_kwargs=minimizer_kwargs, disp=verbose)
        return res

# ==============================================================================
# Setters
# ==============================================================================
    def setInputSample(self, inputSample):
        """

        """
        typeInput = type(inputSample)
        error = ValueError("Don't build a Forest with only one sample...")

        assert typeInput is not int or typeInput is not float, error

        if type(inputSample) is not pd.DataFrame:
            inputSample = pd.DataFrame(inputSample)

        assert inputSample.shape[0] > 1, error

        self._inputSample = inputSample
        [self._numSample, self._dimension] = inputSample.shape
        self.setInputLimits()

    def setOutputSample(self, outputSample):
        """

        """
        typeInput = type(outputSample)
        error = ValueError("Don't build a Forest with only one sample...")

        assert typeInput is not int or typeInput is not float, error

        if type(outputSample) is not pd.Series:
            outputSample = pd.Series(outputSample.ravel())

        assert outputSample.shape[0] > 1, error

        self._outputSample = outputSample

    def setSample(self, inputSample, outputSample):
        """

        """

        self.setInputSample(inputSample)
        self.setOutputSample(outputSample)
        numSample = outputSample.shape[0]
        if numSample != self._numSample:
            raise ValueError("Different size of input and output sample %d(in)\
            != %d(out)" % (numSample, self._numSample))

    def computeForest(self,
                      n_estimators=50,
                      min_samples_leaf=None,
                      n_jobs=4,
                      random_state=None,
                      *more_args,
                      **more_kwargs):
        """

        """

        # Minimum number of sample in a leaf
        if not min_samples_leaf:
            min_samples_leaf = max(10, self._numSample / 100)

        # Sklearn regressor forest object created
        forest = RandomForestRegressor(n_jobs=n_jobs,
                                       n_estimators=n_estimators,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=random_state,
                                       *more_args,
                                       **more_kwargs)

        self._forest = forest.fit(self._inputSample, self._outputSample)

        # Nodes of each sample in all the tree
        self._nodesOfSamples = self._forest.apply(self._inputSample)

        trees = self._forest.estimators_  # List of constructed trees

        # The OOB samples are setted to -1
        self._oobID = np.array([-trees[i].indices_ for i in
                                range(n_estimators)], dtype=np.int)
        self._samplesOOB = pd.DataFrame(self._oobID).T
        self._nodesOfSamples[self._samplesOOB.values] = -1

        self._numTree = n_estimators  # Number of trees

    def setPrecisionOfCDF(self, n_points):
        """
        If the value is set at 0, we will take the quantile from the output
        sample. Else we can create new sample to find the quantile
        """
        if n_points == 0:  # We use the outputSample as precision vector
            self._yCDF = self._outputSample.sort(inplace=False)
        else:  # We create a vector
            yymin = self._outputSample.min()
            yymax = self._outputSample.max()
            self._yCDF = pd.Series(np.linspace(yymin, yymax, n_points))

        # Matrix of output samples inferior to a quantile value
        outMatrix = self._outputSample.reshape(self._numSample, 1)
        cdfMatrix = self._yCDF.reshape(self._yCDF.size, 1).T
        self._infYY = pd.DataFrame(outMatrix <= cdfMatrix).T

    def computeImportance(self, alpha):
        """

        """
        baseError = np.empty(self._numTree)
        for i in range(self._numTree):
            oob = self._oobID[i]
            nOOB = sum(oob)
            X_oob = self._inputSample[oob]
            Yobs_oob = self._outputSample[oob]
            Yest_oob = self.computeQuantile(X_oob, alpha)
            baseError[i] = sum(check_function(Yobs_oob, Yest_oob, alpha))/nOOB
        return baseError

def check_function(Yobs, Yest, alpha):
    """

    """
    u = Yobs - Yest
    return u*(alpha - (u <= 0.)*1.)
