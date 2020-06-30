import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
'''
# input_image0 = Image.open(input_image0_path)
# input_image1 = Image.open(input_image1_path)
#
# fig = plt.figure()
# ax0 = fig.add_subplot(331)
# ax1 = fig.add_subplot(3, 3, 2)
# ax0.imshow(input_image0)
# ax1.imshow(input_image1)
#
# # plt.axis('on')
# plt.show()
x = np.arange(-np.pi, np.pi, 0.01)
y = np.tan(x)
fig = plt.figure()
# plt.plot(x, y)
# plt.xlim(-np.pi, np.pi)
# plt.ylim(-100, 100)
# plt.show()
plt.scatter(x, y, marker = '.',color = 'red', s = 1 ,label = 'tanx')
plt.ylim(-np.pi, np.pi)
plt.show()
'''
import tensorflow as tf
import cv2
input_path = r'C:\Users\Administrator\Desktop\0109009044_B141959\1.BMP'
input_image = cv2.imread(input_path)
print(input_image.shape)
input_image = input_image[np.newaxis, ...]
print(input_image.shape)
input_image = np.array(input_image, dtype=np.float32)
print(input_image.shape, input_image.dtype)
filters = np.array([[1, -1, 1],
                    [1, -1, 1],
                    [1, -1, 1]])
filters = filters[:,:,np.newaxis,np.newaxis]
print(filters.shape)
output_smooth = tf.nn.conv2d(input_image, filters= filters, strides=[1, 1, 1, 1], padding="VALID")
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)
ax0.imshow(input_image.reshape(1536, 2048, 3))
ax1.imshow(output_smooth.reshape(1536, 2048, 3))
plt.show()