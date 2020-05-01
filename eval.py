import cv2
import numpy as np
from scipy.stats import pearsonr


def pearson_correlation(A,B):
  corr, _ = pearsonr(A, B)
  return corr

def get_rotation_size(img_8):
  th3 = cv2.threshold(img_8[:, :, 0], 10, 255, cv2.THRESH_BINARY)[1]
  cnts = cv2.findContours(th3, 0, 2)[1]

  if not cnts.all():
    return None, None
  print("Found contours")
  contours = np.concatenate(cnts, axis=0)
  if contours.shape[0] < 10:
    return None, None

  hull = cv2.convexHull(contours)
  area = np.sqrt(cv2.contourArea(hull))
  ellipse = cv2.fitEllipse(contours)
  degree = ellipse[2]
  degree_signed = degree if degree < 90 else degree - 180

  return degree_signed, area

# testing
a = cv2.imread("./digit-data/real-digits/train/0000.png")
b = cv2.imread("./digit-data/real-digits/train/0001.png")
print(get_rotation_size(b)) 
a = a.flatten()
b = b.flatten() 

print(a.shape)
c = np.array([1,3,4])
d = np.array([1,4,5])
#print(pearson_correlation(a,b))
#print(pearson_correlation(c,d))
