# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.optimize
import sys
sys.path.append("../configuration")
from configuration import shape
import pandas as pd


class QuantileForest() :
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
  def __init__(self, inputSample,
               outputSample,
               n_estimators=50,
               min_samples_leaf=None,
               n_jobs=4,
               numPoints=0,
               optMethod="Cobyla",
               random_state=None) :

    [numSample, dimension] = shape(inputSample, outputSample)

    # Pandas are faster and better with large array
    inputSample = pd.DataFrame(inputSample)
    outputSample = pd.Series(outputSample.ravel())

    # Minimum number of sample in a leaf
    min_samples_leaf = max(10, numSample/100) if not min_samples_leaf else min_samples_leaf
    # Sklearn regressor forest object created
    forest = RandomForestRegressor(n_jobs=n_jobs,
                                   n_estimators=n_estimators,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=random_state, max_features=1)

    self._forest = forest.fit(inputSample, outputSample)
    self._numTree = n_estimators # Number of trees
    self._numSample = numSample # Number of samples
    self._inputSample = inputSample # Input samples
    self._outputSample = outputSample # Output samples
    self._optMethod = optMethod # Optimisation method

    self._dimension = dimension
    # Nodes of each sample in all the tree
    self._nodesOfSamples = self._forest.apply(self._inputSample)
    # The OOB samples are setted to -1

    samplesOOB = pd.DataFrame([-self._forest.estimators_[i].indices_ for i in range(self._numTree)]).transpose()
    self._nodesOfSamples[samplesOOB.values] = -1

    # TODO : Think about how to find the good quantile...
    # If the value is set at 0, we will take the quantile from the output sample
    # Else we can create new sample to find the quantile
    if numPoints == 0 :
      self._yy = pd.Series(np.sort(outputSample))
    else :
      yymin = outputSample.min()
      yymax = outputSample.max()
      self._yy = np.linspace(yymin, yymax, numPoints)
    # Matix of output samples inferior to a quantile value
    self._infYY = pd.DataFrame( pd.DataFrame(outputSample).values <= pd.DataFrame(self._yy).transpose().values ).transpose()

    self.getNodesDimension()

  def getNodesDimension(self):
    xmin = self._inputSample.min(axis=0)
    xmax = self._inputSample.max(axis=0)

    for t in range(self._numTree):
      decisionTree = self._forest.estimators_[t]

      nodeDimension = np.empty((decisionTree.tree_.node_count, 2*self._dimension), dtype=np.float)
      nodeDimension[:, ::2] = xmin
      nodeDimension[:, 1::2] = xmax

      for j in range(decisionTree.tree_.node_count): # For each node
        i_l = decisionTree.tree_.children_left[j] # Get the left node child
        i_r = decisionTree.tree_.children_right[j] # Get the fith node child
        k = decisionTree.tree_.feature[j] # Dimension of the cut -2 if it's a leaf

        if k != -2 :
          cutValue = decisionTree.tree_.threshold[j] # Value of the cut

          # The child leaves take the dimension of the parent
          nodeDimension[i_l, :] = nodeDimension[j, :]
          nodeDimension[i_r, :] = nodeDimension[j, :]

          nodeDimension[i_l, 2*k+1] = cutValue # The max limit of the left node is the threeshold
          nodeDimension[i_r, 2*k] = cutValue # The min limit of the right node is the threeshold

    #  decisionTree.nodeDimension = nodeDimension
      decisionTree.leafID = np.where(decisionTree.tree_.feature == -2)[0]
      decisionTree.leafDimension = nodeDimension[decisionTree.leafID]

  def computeQuantile(self, X, alpha, doOptim=True, regressorID=[-1]) :
    """
    Compute the conditional alpha-quantile of forest.
    """

    dim = len(regressorID)

    ## If the inputs are scalars
    if type(alpha) is int or type(alpha) is float :
      alpha = [alpha]

    # Converting to array for convenience
    X = np.array(X)
    alpha = np.array(alpha)

    try :
      [n, p] = X.shape
    except : # It's a vector
      try :
        a = len(X)
        if a == dim :
          n = 1
        else :
          n = a
      except : # It's a scalar
        X = [X]
        n = 1

    numRegressor = n # Number of quantiles to compute
    numAlpha = len(alpha) # Number of probabilities

    if n > 1 :
      if X.size == numRegressor :
        X.resize(numRegressor, 1)

    # Matrix of computed quantiles
    quantiles = np.zeros((numRegressor, numAlpha))

    regressorID = np.array([regressorID])
    # Nodes of the regressor in all the trees
#    nodesOfRegressor = self._forest.apply(X).transpose()

    for k in range(numRegressor) : # For each regressor
      weight = np.zeros((self._numSample,), dtype=np.float)
      for t in range(self._numTree): # For each tree
        decisionTree = self._forest.estimators_[t] # The tree object

        # The dimension of the parameters we want to check
        paramDim = decisionTree.leafDimension[:, 4:]
        # The ID of the params
        inRho = np.where((X[k] >= paramDim[:, 0]) & (X[k] <= paramDim[:, 1]))[0]
        # The leaves ID containing the regressor
        leafIDinX = decisionTree.leafID[inRho]

        wi = np.zeros((self._numSample,), dtype=np.float)
        for iD in leafIDinX : # For each leaf
          # the samples in the leaf
          sampleInID = (self._nodesOfSamples[:, t] == iD)*1.
          wi += sampleInID

        # The weight of each sample in the trees
        weight += wi/wi.sum()

      weight /= self._numTree
      if doOptim :
        y0 = self._outputSample[weight != 0].mean()
#        y00 = np.percentile(self._outputSample[weight != 0], alpha*100)
        for i, alphai in enumerate(alpha) :
#          y0 = y00[i]
          if self._optMethod == "Cobyla" :
            quantiles[k, i]  = scipy.optimize.fmin_cobyla(self._optFunc, y0, [self._constrFunc], args=(weight, alphai), disp=0)
          else :
            epsilon = 1.E-1*abs(y0)
            quantiles[k, i]  = scipy.optimize.fmin_slsqp(self._optFunc, y0, f_ieqcons=self._constrFunc, args=(weight, alphai), disp=0, epsilon=epsilon)
      else :
        CDF = self._infYY.dot(weight).ravel() # Compute the CDF
        quantiles[k, :] = [ self._yy.values[CDF >= alphai][0] for alphai in alpha ]
        self._CDF = CDF

    return quantiles

  def _optFunc(self, yi, w, alpha) :
    alphai = w[self._outputSample.values <= yi].sum()
    return abs(alphai - alpha)**2

  def _constrFunc(self, yi, w, alpha) :
    alphai = w[self._outputSample.values <= yi].sum()
    return alphai - alpha