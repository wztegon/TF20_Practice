import  numpy as np
# import sys
# import os
# import cv2
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# a = np.arange(24).reshape(2, 3, 4)
# print('a.shape   : {}'.format(a.shape))
# print('a.shape[0]: {:4d}.'.format(a.shape[0]))
# print('a.shape[1]: {:4d}.'.format(a.shape[1]))
# print('a.shape[2]: {:4d}.'.format(a.shape[2]))
#
# b = np.array([-2, -6])
# print(b//2)
print((3, 4) + (5,))
a = np.zeros((3, 4, 5, 2))
b = np.ones((3, 4, 5, 1))
ab = np.concatenate((a, b), axis=-1)
print(ab.shape)
print(ab[0, 0, 0, :])