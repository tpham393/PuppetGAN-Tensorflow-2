import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import pearsonr


def pearson_correlation(A,B):
  # A and B are integer numpy arrays
  corr, _ = pearsonr(A, B)
  return corr


def tensor_pearson_correlation(A,B, sample_axis=0,event_axis=None):
    # A: A numeric Tensor holding samples 
    # B: Optional Tensor with same dtype and shape as A. Default value: None (B is effectively set to A).
    # sample_axis: Scalar or vector Tensor designating axis holding samples
    # event_axis:Axis indexing random events, whose correlation we are interested in. Default value: -1 (rightmost axis holds events).
    return tfp.stats.correlation(x, y)

# testing 
x = tf.random.normal(shape=(100, 2))
y = tf.random.normal(shape=(100, 2))
print(tensor_pearson_correlation(x,y))    
