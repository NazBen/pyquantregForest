from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.forest import BaseForest, ForestRegressor
import numpy as np
from scipy.optimize import fmin_cobyla, fmin_slsqp, basinhopping
from pathos.multiprocessing import ProcessingPool
from pandas import DataFrame, Series
import pylab as plt

class QuantileForest(RandomForestRegressor):
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
        Grow trees with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None then unlimited
        number of leaf nodes. If not None then max_depth will be ignored.
        Note: this parameter is tree-specific.

      n_jobs : int, optional (default=4)
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

      numPoints : int, optional (default=0)
        The size of the vector used to determines the quantile. If 0, the
        vector use is the outputSample.

      outputSample : string, optional (default="Cobyla")
        Name of the Optimisation method to find the alpha-quantile (if the
        option is chosen in the computeQuantile method). Only "Cobyla" and
        "SQP" are available.

      random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    """


    def fit(self, X, y):
        """

        """
        # We transform X as a np array for use convenience
        X = np.asarray(X)

        # It's a vector
        if X.shape[0] == X.size:
            self._n_sample = X.shape[0]
            self._input_dim = 1
        else:
            self._n_sample, self._input_dim = X.shape

        # The bootstrap is mandatory for the method. Since update 
        # 1.16 of Sklearn, the indices of each element are not 
        # availables. TODO: find a way to get OOB indices.
        self.bootstrap = True

        # Fit the forest
        RandomForestRegressor.fit(self, X, y)

        # Save the data. Necessary to compute the quantiles.
        self._input_sample = DataFrame(X)
        self._output_sample = Series(y)

        # The resulting node of each elements of the sample
        self._sample_nodes = DataFrame(self.apply(X))  

        return self

    def _check_input(self, X):
        """

        """
        n = X.shape[0]  # Number of sample
        try:  # Works if X is an array
            d = X.shape[1]  # Dimension of the array
            if d != self._input_dim:  # If the dimension is not correct
                if n == self._input_dim:  # There is one sample of d dimension
                    d = n
                    n = 1
                else:  # Error
                    raise ValueError("X dimension is different from forest \
                    dimension : %d (X) != %d (forest)" % (d, self._input_dim))
        except:  # Its a vector
            d = 1
            if d != self._input_dim:  # If the dimension is not correct
                if n == self._input_dim:  # There is one sample of d dimension
                    d = n
                    n = 1
                else:  # Error
                    raise ValueError("X dimension is different from forest \
                    dimension : %d (X) != %d (forest)" % (d, self._input_dim))

        if (n > 1) & (d == 1):
            X.resize(n, 1)

        return X, n

    def _compute_weight(self, X_nodes_k, i_tree):
        """
        """
        if i_tree < 0:
            sample_node = self._sample_nodes.values
        else:
            sample_node = self._nodesOfSamples.values[:, i_tree]
        tmp = (sample_node == X_nodes_k)

        # Number of samples in nodes
        n_samples_nodes = tmp.sum(axis=0)

        # The proportion in each node
        # Shape : Matrix (numSample * numTree)
        weight = tmp.astype(float) / n_samples_nodes

        # The weight of each sample in the trees
        # Shape : Vector (numSample * )
        if i_tree < 0:
            return weight.mean(axis=1)
        else:
            return weight

    def get_nodes(self, X, i_tree):
        """
        """
        X, n_quantiles = self._check_input(X)

        # Nodes of the regressor in all the trees
        # Shape : (numTree * numRegressor)
        if i_tree < 0:
            # Sklearn does not like arrays of one values...
            if n_quantiles == 1 and self._input_dim == 1:
                X_nodes = self.apply(X[0]).transpose()
            else:
                X_nodes = self.apply(X).transpose()
        else:
            tree = self.estimators_[i_tree].tree_
            X_nodes = tree.apply(X.astype(np.float32))
            X_nodes.resize((1, n_quantiles))

        return X_nodes

    def compute_CDF(self, X, y, i_tree=-1):
        """
        """
        if type(X) in [int, float]:
            X = [X]
        if type(y) in [int, float]:
            y = [y]

        # Converting to array for convenience
        X = np.asarray(X)
        y = np.asarray(y)
        X, n_X = self._check_input(X)
        n_y = y.shape[0]
        y.resize(n_y, 1)
        
        self._prepare_CDF()

        CDFs = np.zeros((n_y, n_X))
        X_nodes = self.get_nodes(X, i_tree)
        for k in range(n_X):
            weight = self._compute_weight(X_nodes[:, k], i_tree)
            id_pos = weight > 0
            CDFs[:, k] = (weight[id_pos] * (self._output_sample.values[id_pos] <= y)).sum(axis=1)
        #CDF = self._infYY.dot(weight).ravel()  # Compute the CDF
        return CDFs

    def computeQuantile(self, X, alpha, do_optim=True, verbose=False,
                        doSaveCDF=False, i_tree=-1, opt_method="Cobyla"):
        """
        Compute the conditional alpha-quantile.
        """
        if type(alpha) in [int, float]:
            alpha = [alpha]
        if type(X) in [int, float]:
            X = [X]

        # Converting to array for convenience
        alpha = np.asarray(alpha)
        X = np.asarray(X)

        # Number of quantiles to compute
        X, n_quantiles = self._check_input(X)
        n_alphas = alpha.size  # Number of probabilities

        # Matrix of computed quantiles
        quantiles = np.zeros((n_quantiles, n_alphas))

        if doSaveCDF or not do_optim:
            self._prepare_CDF()
        if doSaveCDF:
            self._CDF = np.empty((self._yCDF.size, n_quantiles))

        X_nodes = self.get_nodes(X, i_tree)

        # For each quantiles to compute
        for k in range(n_quantiles):
            weight = self._compute_weight(X_nodes[:, k], i_tree)

            # Compute the quantile by minimising the pinball function
            if do_optim:
                # The starting points are the percentiles
                # of the non-zero weights.
                y0 = np.percentile(self._output_sample[
                                   weight != 0], alpha * 100.)

                # For each alpha
                for i, alphai in enumerate(alpha):
                    # The quantile is obtain by the minimisation of the
                    # weighted check function.
                    if opt_method == "Cobyla":
                        quantiles[k, i] = fmin_cobyla(self._min_function,
                                                      y0[i], [],
                                                      args=(weight, alphai),
                                                      disp=verbose)

                    elif opt_method == "SQP":
                        epsilon = 1.E-1 * abs(y0[i])
                        quantiles[k, i] = fmin_slsqp(self._min_function,
                                                     y0[i],
                                                     args=(weight, alphai),
                                                     disp=verbose,
                                                     epsilon=epsilon)
                    else:
                        raise ValueError("Unknow optimisation method %s" %
                                         opt_method)
            else:
                CDF = self._infYY.dot(weight).ravel()  # Compute the CDF
                quantiles[k, :] = [self._yCDF.values[CDF >= alphai][0]
                                   for alphai in alpha]
                if doSaveCDF:
                    self._CDF[:, k] = CDF

        if n_quantiles == 1 and n_alphas == 1:
            return quantiles[0][0]
        elif n_quantiles == 1 or n_alphas == 1:
            return quantiles.ravel()
        else:
            return quantiles

    def _min_function(self, yi, w, alpha):
        """
        Minimisation function used to compute the conditional quantiles.
        The function need the curret value of $y$, the weight of each observation
        and the alpha value. The check function of the residual between $y_i$ and the
        output sample, pondered with the weight is minimised.
        """
        # Weighted deviation between the current value and the output sample.
        # TODO: Think about using only the non-null weight to increases performances
        u = w*(self._output_sample.values - yi)
        return check_function(u, alpha).sum()
    
# ==============================================================================
# Setters
# ==============================================================================
    def _prepare_CDF(self):
        """
        If the value is set at 0, we will take the quantile from the output
        sample. Else we can create new sample to find the quantile
        """
        self._yCDF = self._output_sample.sort_values(inplace=False)

        # Matrix of output samples inferior to a quantile value
        out_martrix = self._output_sample.reshape(self._n_sample, 1)
        cdf_matrix = self._yCDF.reshape(self._yCDF.size, 1).T
        self._infYY = DataFrame(out_martrix <= cdf_matrix).T

    def _computeImportanceOfTree(self, alpha, i):
        """

        """
        oob = self._oobID[i]
        X_oob = self._inputSample.values[oob, :]
        Yobs_oob = self._outputSample.values[oob]
        Yest_oob = self.computeQuantile(X_oob, alpha, i_tree=i)
        baseError = (check_function(Yobs_oob, Yest_oob, alpha)).mean()

        permError = np.empty(self._input_dim)
        for j in range(self._input_dim):
            X_oob_perm = np.array(X_oob)
            np.random.shuffle(X_oob_perm[:, j])
            Yest_oob_perm = self.computeQuantile(X_oob_perm, alpha, i_tree=i)
            permError[j] = check_function(Yobs_oob, Yest_oob_perm, alpha)\
                .mean()

        return (permError - baseError)

    def compute_importance(self, alpha):
        """

        """
        pool = ProcessingPool(self._numJobs)
        errors = pool.map(self._computeImportanceOfTree,
                          [alpha] * self._numTree, range(self._numTree))
        return np.array(errors).mean(axis=0)


def check_function(u, alpha):
    """

    """
    return u * (alpha - (u < 0.) * 1.)

if __name__ == "__main__":
    """
    The main execution is just an example of the Quantile Regression Forest 
    applied on a sinusoidal function with Gaussian noise.
    """

    def sin_func(X):
        X = np.asarray(X)
        return 3*X
    
    np.random.seed(0)
    dim = 1
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