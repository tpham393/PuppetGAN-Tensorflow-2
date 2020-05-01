import cv2
import numpy as np
from scipy.stats import pearsonr


def pearson_correlation(A,B):
  corr, _ = pearsonr(A, B)
  return corr



# testing
a = cv2.imread("./digit-data/real-digits/train/0000.png")
b = cv2.imread("./digit-data/real-digits/train/0001.png")
a = a.flatten()
b = b.flatten() 

print(a.shape)
c = np.array([1,3,4])
d = np.array([1,4,5])
print(pearson_correlation(a,b))
print(pearson_correlation(c,d))
 
