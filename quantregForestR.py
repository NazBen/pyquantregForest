# -*- coding: utf-8 -*-
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects import r
import rpy2.rinterface as ri
import sys
sys.path.append("../configuration")
from configuration import shape

ri.initr()
quantregForest = importr("quantregForest")

class RquantForest() :
  def __init__(self, inputSample, outputSample) :
    [numSample, dimension] = shape(inputSample, outputSample)

    inputSample = np.array(inputSample)
    if inputSample.size == numSample :
      inputSample.resize(numSample, dimension)

    r_x = numpy2ri(inputSample)
    r_y = r['as.numeric'](numpy2ri(outputSample))
    self._quantForest = quantregForest.quantregForest(r_x, r_y)

  def computeQuantile(self, alpha) :
    if type(alpha) is int or type(alpha) is float :
      alpha = [alpha]
    r_alpha = numpy2ri(np.array(alpha))
    return np.array(quantregForest.predict_all(self._quantForest, quantiles=r_alpha))