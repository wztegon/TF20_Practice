import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
# import tensorflow as tf

'''
v = tf.Variable(1)
c_1 = tf.constant(2)
c = tf.constant(1)
# a = tf.make_ndarray(c)
print(c)
print(c_1)
print(v)
print(v.name)
tf_a = tf.range(10, delta=1)
print(tf_a)
x = tf.random.normal([4, 32, 32, 3])
print(x.shape)
x_reshape = tf.reshape(x, [4, 3, 32, 32])
tf.transpose()
tf.reshape()
tf.cast()
'''
'''
input_tensor = tf.fill([5,3], 2)
output_tensor = tf.fill([5, 1], 1)
input_train = np.ones((5, 3), dtype= np.int32)
output_train = np.zeros((5, 1), dtype= np.int32)
print(input_tensor.shape, output_tensor.shape)
dataset1 = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
dataset2 = tf.data.Dataset.from_tensor_slices((input_train, output_train))
'''
'''
ta = tf.range(12, dtype=tf.float32)
ta = tf.reshape(ta, (2, 3, 2))
ta_div_10 = ta / 10
print(ta_div_10)
'''
# '''numpy基础
# a = np.ones((2, 3), dtype=np.int32)
# print('strides', a.strides)
# print('ndim', a.ndim)
# print('itemsize', a.itemsize)
# print()
# print('flags:\n{}'.format(a.flags	))#有关数组内存布局的信息。
# print('shape	',a.shape	)         #数组维度的元组。
# print('strides	',a.strides	)         #遍历数组时每个维度中的字节元组。
# print('ndim	',a.ndim	)             #数组维数。
# print('data	',a.data	)             #Python缓冲区对象指向数组的数据的开头。
# print('size	',a.size	)             #数组中的元素数。
# print('itemsize',a.itemsize)          #	一个数组元素的长度，以字节为单位
# print('nbytes	',a.nbytes	)         #数组元素消耗的总字节数。
# print('base	',a.base	)             #如果内存来自其他对象，则为基础对象。
#
#
# print('T	 ：\n', a.T	   )#The transposed array.
# print('real  ：\n', a.real   )#The real part of the array.
# print('imag  ：\n', a.imag   )#The imaginary part of the array.
# print('flat  ：\n', a.flat   )#A 1-D iterator over the array.
# print('ctypes：\n', a.ctypes )#	An object to simplify the interaction of the array with the ctypes module.
#
#
# a_iter = iter(a)
# print(next(a_iter))
# '''
'''
n1 = np.arange(2 * 3 * 4).reshape(2, 3, 2, 2)
print(n1)
print('--------------------')
# print(n1.sum(axis=0))
# print(n1.sum(axis=0).shape)
# print(n1.sum(axis=1))
# print(n1.sum(axis=1).shape)
# print(n1.sum(axis=2))
# print(n1.sum(axis=2).shape)
# print(n1.sum(axis=3))
# print(n1.sum(axis=3).shape)
print(n1.prod(axis=0))
print(n1.prod(axis=0).shape)
print(n1.cumprod(axis=0))
print(n1.cumprod(axis=0).shape)
print('--------------------')
# n_one = np.zeros((2, 3))
# print(n_one)
n_random_int = np.random.randint(0, 100, 24, dtype=np.int64).reshape(4, -1)
print(n_random_int)
lst = [2, 3, 4]
n_lst = np.array(lst, dtype=np.int16)
print(n_lst, n_lst.dtype)
n_dot1 = np.arange(1, 7).reshape(2, 3)
n_dot2 = np.arange(10, 70, step=10).reshape(2, 3)
n_dot_res = np.dot(n_dot1.T, n_dot2)
print(n_dot1)
print(n_dot2)
print(n_dot_res)
print('--------------------')
n_concat1 = np.zeros((2, 2, 4))
n_concat2 = np.ones((2, 3, 4))
n_concat_res = np.concatenate((n_concat1, n_concat2), axis=1)
print(n_concat_res)
'''

# input_image_path = r'C:\Users\Administrator\Desktop\0.JPG'
# input_image = Image.open(input_image_path)
# input_image=cv2.cvtColor(cv2.imread(input_image_path),cv2.COLOR_BGR2RGB)
# cv2.rectangle(input_image, (123, 123), (223, 223), (255, 0, 0), 3)
# fig = plt.figure()
# plt.imshow(input_image)
# plt.show()
# a = np.ones((3, 2))
# print(a.shape)
# a = a.reshape(1, 3, 2)
# print(a.shape)
# a = np.ones(24).reshape((6, 4))
# b = np.ones(6).reshape((6, 1))
# print(a + b)
# dir = r"C:\Users\Administrator\Desktop\2020.07.01__Sensofar测量数据\BULLET.dat"
# with open(dir, 'rb') as f:
# 	lst = f.readlines()
# 	print(len(lst))
# 	print(lst[-1])
L = []
l = []
for _ in range(5):
	for j in range(3):
		l.append(j * _)
	L.append(l)
	l = []
	
print(len(L))
print(L)
a = np.array(L)
print(a.shape)
a_max1 = np.maximum(a, 1)
print(a_max1)