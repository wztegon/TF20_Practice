import  numpy as np
import sys
import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# import tensorflow as tf
# arr0 =np.array([1,2])
# arr1=np.array([[1,3],[2,4],[3,5]])
# arr2=np.array([[4,9],[5,8],[6,7]])
# print(np.vstack((arr0,arr2)))
# print(np.hstack((arr1,arr2)))
#
# s = set(['Adam', 'Lisa', 'Bart', 'Paul'])
# l = []
# for i in s:
#     l.append(i.lower())
# s.update(l)
# print(s)
# str = "hello word!"
# print(str.upper())          # 把所有字符中的小写字母转换成大写字母
# print(str.lower())          # 把所有字符中的大写字母转换成小写字母
# print(str.capitalize())     # 把第一个字母转化为大写字母，其余小写
# print(str.title())          # 把每个单词的第一个字母转化为大写，其余小写
# L = []
# L = list(range(0,101))
# print(type(L))
# def sum(L):
#     s = 0
#     for i in L:
#         s += i*i
#     return s
# print(sum(L))
# import math
# def add(x,y,f):
#     return f(x) + f(y)
# len = abs
# print(add(-1,-2,len),len(1))
# math.sqrt
# def format_name(s):
#     L = []
#     for i in s:
#         i = i[0].upper()+i[1:].lower()
#         L.append(i)
#     return L
#
# print(map(format_name, ['adam', 'LISA', 'barT']))
# import functools
# def cmp_ignore_case(s1, s2):
#     if s1[0].upper() < s2[0].upper():
#         return -1
#     elif s1[0].upper() > s2[0].upper():
#         return 1
#     else:
#         return 0
#
# print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=functools.cmp_to_key(cmp_ignore_case)))
#
# def new_fn(f):
#     def fn(n):
#         print('Call '+ f.__name__+'()')
#         return f(n)
#     return fn
# @new_fn
# def f1(x):
#     return x*2
# print(f1(5))
# import time
# from functools import reduce
#
#
# def performance(f):
#     def fn(*args, **kw):
#         t_start = time.time()
#         r = f(*args, **kw)
#         t_end = time.time()
#         print('call %s() in %.28fs' % (f.__name__, (t_end - t_start)))
#         return r
#     return fn
#
# @performance
# def factorial(n):
#     return reduce(lambda x,y: x*y, range(1, n+1))
#
# print(factorial(10000))
# import time
# from functools import reduce
#
#
# def performance(unit):
#     def performance_decorator(f):
#         def wrapper(*args, **kw):
#             C = {'ms':1000,'s':1}
#             t_start = time.time()
#             r = f(*args, **kw)
#             t_end = time.time()
#             print('call %s() in %f%s' % (f.__name__, (t_end - t_start)*C[unit], unit))
#             return r
#         return wrapper
#     return performance_decorator
#
# @performance('ms')
# def factorial(n):
#     return(reduce(lambda x,y: x*y, range(1, n+1)))
#
# print(factorial(10))
# import functools
# from functools import cmp_to_key
#
#
# sorted_ignore_case = functools.partial(sorted, key=str.lower)
#
# print(sorted_ignore_case(['bob', 'about', 'Zoo', 'Credit']))
# from os.path import isdir, isfile
# print(isdir(r'C:\Windows'))
# print(isfile(r'C:\Windows\notepad.exe'))
# class Person(object):
#     def __init__(self, name, gender, birth, **kw):
#         self.name = name
#         self.gender = gender
#         self.birth = birth
#         for k,w in kw.items():
#             setattr(self,k,w)
# xiaoming = Person('Xiao Ming', 'Male', '1990-1-1', job='Student')
#
# print(xiaoming.name)
# print(xiaoming.job)
# import random
# L = []
# for i in range(0,30):
#     L.append(random.randint(0,10))
# print(sorted(L,reverse=True))
# class Person(object):
#     count=0
#     def __init__(self,name):
#         self.name=name
#         Person.count += 1
# p1 = Person('Bob')
#
# print(p1.count)
#
# p2 = Person('Alice')
# print(Person.count)
#
# p3 = Person('Tim')
# print(Person.count)
# class Person(object):
#     def __init__(self, name, gender):
#         self.name = name
#         self.gender = gender
#
# class Teacher(Person):
#
#     def __init__(self, name, gender, course):
#         super(Teacher, self).__init__(name, gender)
#         self.course = course
# t = Teacher('Alice', 'Female', 'English')
# print(t.name)
# print(t.course)
# import json
#
# class Students(object):
#     def __init__(self):
#         self.n = ["Tim", "Bob", "Alice"]
#     def read(self):
#         return 'l'
#         #return str(self.l)
#
# s = Students()
#
# print(list(filter(lambda x :x[0:2] != '__' ,dir(s))))

# class Person(object):
#     def __init__(self, name, gender):
#         self.name = name
#         self.gender = gender
#
# class Student(Person):
#     def __init__(self, name, gender, score):
#         super(Student, self).__init__(name, gender)
#         self.score = score
#     def whoAmI(self):
#         return 'I am a Student, my name is %s' % self.name
# s = Student('Bob', 'Male', 88)
# print(setattr(s, 'age', 10))
# print( getattr(s, 'age'))
# print(s)
# from filecmp import cmp
#
#
# class Student(object):
#     def __init__(self, name, score):
#         self.name = name
#         self.score = score
#     def __str__(self):
#         return '(%s: %s)' % (self.name, self.score)
#     __repr__ = __str__
#     def __cmp__(self, s):
#         if not isinstance(s, Student):
#             return cmp(self.name, str(s))
#         if self.score > s.score:
#             return -1
#         elif self.score < s.score:
#             return 1
#         else:
#             return 0
#
#
# L = [Student('Tim', 99), Student('Bob', 88), Student('Alice', 99)]
# #L = [Student('Tim', 99), Student('Bob', 88), 100, 'Hello']
# print(sorted(L))
# class Student(object):
#
#     def __init__(self, name, score):
#         self.name = name
#         self.score = score
#
#     def __str__(self):
#         return '(%s: %s)' % (self.name, self.score)
#
#     __repr__ = __str__
#
#     # def __cmp__(self, s):
#     #     if self.score > s.score:
#     #         return -1
#     #     elif self.score < s.score:
#     #         return 1
#     #     elif self.name > s.name:
#     #         return 1
#     #     elif self.name < s.name:
#     #         return -1
#     #     else:
#     #         return 0
#
# L = [Student('Tim', 99), Student('Bob', 88), Student('Alice', 99)]
# print(sorted(L, key = lambda s:(-s.score, s.name)))
# class Rational(object):
#     def __init__(self, p, q):
#         self.p = p
#         self.q = q
#
#     def __add__(self, r):
#         return Rational(self.p * r.q + self.q * r.p, self.q * r.q)
#
#     def __sub__(self, r):
#         return Rational(self.p * r.q - self.q * r.p, self.q * r.q)
#
#     def __mul__(self, r):
#         return Rational(self.p * r.p, self.q * r.q)
#
#     def __truediv__(self, r):
#         return Rational(self.p * r.q, self.q *r.p)
#     @classmethod
#     def gcd(cls, a, b):
#         if b == 0:
#             return a
#         return cls.gcd(b, a % b)
#
#     def __str__(self):
#         g = self.gcd(self.p, self.q)
#         return '%i/%i' % (self.p/g, self.q/g)
#
#     __repr__ = __str__
#
# r1 = Rational(1, 2)
# r2 = Rational(1, 4)
# print(r1 + r2)
# print(r1 - r2)
# print(r1 * r2)
# print(r1 / r2)
# class Rational(object):
#     def __init__(self, p, q):
#         self.p = p
#         self.q = q
#
#     def __int__(self):
#         return self.p // self.q
#
#     def __float__(self):
#         return float(self.p)/float(self.q)
#
# print(int(Rational(7, 2)))
# print(float(Rational(1, 3)))
# class Fib(object):
#     def __init__(self, num):
#         self.nlst = []
#         a = 0
#         b = 1
#         for i in range(num):
#             self.nlst.append(a)
#             a, b = b, a + b
#             # b = self.nlst[i]
#     def __str__(self):
#         return str(self.nlst)
#
#     def __len__(self):
#         return len(self.nlst)
#
#
# f = Fib(10)
# print(f)
# -*- coding:utf-8 -*-
# f = open('test_file.txt', 'r+') #f是文件的文件句柄，它是在内存中的，是内存中的一个对象
# data = f.read()
# data2 = f.read()
# print(data)
# print(len(data2))
# print(f.tell())
# f.seek(2)
# data3 = f.read()
# print(data3)
# f.close()
# f = open('test_file.txt', 'w+')
# f.write('写入字符串第an个字符' )
# f.flush()
# f.close()
# f = open('test_file.txt', 'r+')
# data4 = f.read()
# f.close()
# print(len(data4))
# coding: utf-8
#
# import string
# from collections import namedtuple
#
#
# def str_count(s):
#     '''找出字符串中的中英文、空格、数字、标点符号个数'''
#
#     count_en = count_dg = count_sp = count_zh = count_pu = 0
#     s_len = len(s)
#     for c in s:
#         if c in string.ascii_letters:
#             count_en += 1
#         elif c.isdigit():
#             count_dg += 1
#         elif c.isspace():
#             count_sp += 1
#         elif c.isalpha():
#             count_zh += 1
#         else:
#             count_pu += 1
#     total_chars = count_zh + count_en + count_sp + count_dg + count_pu
#     if total_chars == s_len:
#         return namedtuple('Count', ['total', 'zh', 'en', 'space', 'digit', 'punc'])(s_len, count_zh, count_en, count_sp, count_dg, count_pu)
#     else:
#         print('Something is wrong!')
#         return None
#     return None
#
#
# s = '上面是引用了官网的介绍，意思就是说 TensorBoard 就是一个方便你理解、调试、优化 TensorFlow 程序的可视化工具，你可以可视化你的 TensorFlow graph、学习参数以及其他数据比如图像。'
# count = str_count(s)
# print(s, end='\n\n')
# print('该字符串共有 {} 个字符，其中有 {} 个汉字，{} 个英文，{} 个空格，{} 个数字，{} 个标点符号。'
#       .format(count.total, count.zh, count.en, count.space, count.digit, count.punc))
# L = ['Adam', 'Lisa', 'Bart', 'Paul']
# L = [22] * 100
# L[1:10] = [11] * 9
# print(L)
# print(L[3:66].index(22))
# x = np.array([[[4,2],[1,3],[5,8]]]*4,dtype = np.complex)
# print(x[1,0,0])
# print(x.shape)
# print(x.dtype)
# print(x.itemsize)
# y = np.array((1,2,3))
# x = np.empty((2,3))
# print(y.shape)
# print(y,x)
# print(x.dtype)
#print(np.arange(10000).reshape(100,100))
# a = np.arange(9).reshape(3,3)
# b = np.arange(10,19).reshape(3,3)
# print(a,'\n', b, '\n', a@b, '\n',b@a)
# a = np.ones(12).reshape(3,4)
# b = np.random.random((3,4))
#
# print(a,'\n',b,'\n',a@b.T)
# a = np.ones(24).reshape(3,2,-1)
# b = np.fromfunction(lambda x,y,z:10*x+2*y+z,(3,2,2))
# print(b)
# print(b[:,0,1])
# print(a.reshape(2,2,2,-1))
# a = np.array((1,2,3,4))
# b = np.array((2,3,4,5))
# print(np.column_stack(a.reshape(2,-1)))
# c = b[:, newaxis, newaxis]
# print(c.shape,b[...,],c[...,0])
# print('hello',np.r_[1:4,0,4])
# a = np.arange(48).reshape(3,4,-1)
# print(a)
# print(np.min(a, axis= 2))
# print(np.maximum.reduce(a, axis= 2))
# from PIL import Image, ImageDraw
# input_image_full_name = \
#     r"C:\Users\Administrator\Desktop\TensorFlow2.0-Examples-master\4-Object_Detection\MTCNN\timg2.jpg"
# im = Image.open(input_image_full_name)
# im_ndarray = np.array(im)
# print(im_ndarray[0, 0, ...])
# im = Image.fromarray(im_ndarray)
# im.show()
# img_ndarray = cv2.cvtColor(cv2.imread(input_image_full_name), cv2.COLOR_BGR2RGB)
# print(img_ndarray[0, 0, :])
# im = Image.fromarray(img_ndarray)
# print(type(im))

# im = Image.fromarray(image)
# def jwkj_get_filePath_fileName_fileExt(filename):
#   (filepath,tempfilename) = os.path.split(filename);
#   (shotname,extension) = os.path.splitext(tempfilename);
#   return filepath,shotname,extension
# res = jwkj_get_filePath_fileName_fileExt(full_image)
# output_image_full_name = res[0] + res[1] + '_detect' + res[2]
# res = os.path.splitext(input_image_full_name)

# im.save(full_image.split('.')[-2]+'_detect.jpg' )
# cv2.circle(image, (10, 20),  1, (255, 0, 0), 2)
# cv2.circle(image, (10, 80),  1, (255, 0, 0), 2)
# cv2.circle(image, (100, 20), 1, (255, 0, 0), 2)
# cv2.circle(image, (100, 80), 1, (255, 0, 0), 2)
# Image.fromarray(image).show()
# a = np.arange(100).reshape(10, 10)
# b = a[np.newaxis, 5:, 5:, np.newaxis]
# c = np.ones(np.size(b)).reshape(1,5,5,1)
# # # b_exp = np.expand_dims(b[...], axis= 2)
# # # bc = np.stack((b, c, c), axis= 2)
# # bc_v = np.vstack((b, c))
# # bc_h = np.hstack((b, c))
# # bc_s_0 = np.stack((b, c), axis= -1)
# # bc_conca = np.concatenate((b,c), axis= 0)
# # print(bc_v.shape  )
# # print(bc_h.shape  )
# # print(bc_s_0.shape)
# # print(bc_conca.shape)
#
# bc = np.vstack((b,c))
# bcc = np.vstack((bc, c))
# print(bcc.shape)
# print()
# import tensorflow as tf
# alpha = np.array([[1], [0.1], [0.01]])
# gradent = np.array([[5], [4], [3]])
# print(alpha*gradent)
# import tensorflow as tf
# class mymodel(tf.keras.models.Model):
# 	def __init__(self):
# 		super(mymodel, self).__init__()
# 		self.hidden1_layer = tf.keras.layers.Dense(30, activation="selu")
# 		self.output_layer = tf.keras.layers.Dense(1)
# 	def call(self, input):
# 		hidden1 = self.hidden1_layer(input)
# 		output = self.output_layer(hidden1)
# 		return  output
# model = mymodel()
# model.build(input_shape=(None, 1))
# input = np.arange(256, dtype = np.float)
# input_labels = input*3
# model.compile(loss= 'MSE', optimizer= tf.keras.optimizers.Adam(lr= 0.01))
# model.fit(input,input_labels, epochs= 100)
# print(model.predict([32.]))
# op = tf.keras.optimizers.SGD.apply_gradients()
# a = np.arange(100).reshape(10, -1)
# b = a[::2, ::2]
# for i in b:
# 	print(i)
# print(b)
# import  re
# s = '362.asc'
# if re.search(r'\d[0-5]\d.asc', s):
# 	print('匹配正确')
# else:
# 	print('匹配错误')
# a = b'ni hao a'
# a_list = a.split(' ')
# print(a_list)
# import tensorflow as tf
# ta = tf.constant([1,2,3])
# tb = tf.constant([4,5,6])
# tc = tf.constant([7,8,9])
# l = []
# l.append(ta)
# l.append(tb)
# l.append(tc)
# tf.cast()
# tc_1 = tc[tf.newaxis, :]
# tab = tf.stack([ta, tb])
# print(tab)
# print(list(map(tf.stack, l)))
# sa = tf.constant('你好')
# sb = tf.constant('-2.86')
# sab = tf.strings.join([sa, sb], '\t')
# sr = tf.strings.split(sab, '\t')
# print(sr.shape)
# print(tf.strings.to_number(sa, out_type=tf.float32))
# print(sa)
# def strTOfloat(s):
# 	return 	tf.strings.to_number(s, out_type=tf.float32)
# data_float = tf.map_fn(strTOfloat,  sr, dtype = tf.float32)
# print(data_float)
# na = sa.numpy()
# print(na, type(na))
# print(str(na, encoding='UTF8'))
# def fun():
# 	for i in range(20):
# 	    x=yield i
# 	    print('good',x)
#
# if __name__ == '__main__':
# 	a=fun()
# 	print(next(a))
# 	print(next(a))
# 	print(next(a))
#
# 	x=a.send(5)
# 	print(x)
# def add(a, b):
#     return  a+b
# def option(fun):
#     return fun()
# print(option(lambda : add(1, 3)))
# a = np.arange(240).reshape(20, -1)
# a_0 = a[:, 0::2]
# a_1 = a[:, 1::2]
# print(a)
# print(a_0)
# print(a_1)
# a_01 = np.concatenate([a_0, a_1], axis= -1)
# print(a_01)
# def plot_position_emedding(position_embedding):
# 	plt.pcolormesh(position_embedding, cmap= 'RdBu')
# 	# plt.xlabel()
# 	plt.xlim((0, 12))
# 	plt.colorbar()
# 	plt.show()
# plot_position_emedding(a_01)
# sa = tf.constant('你好')
# a = sa.numpy()
# print(a)
# print(str(a, encoding= "UTF8"))

'''给定数组改变元素正负使得和为要求的值'''
'''  使用递归给出一个解
# l = np.random.randint(1, 11, 4)
l = np.array([8, 5, 6, 8])
l_result = []
def find_operations( target, sublist):
	if (np.sum(np.abs(sublist)) + target) % 2 == 0:
		if np.size(sublist) <= 2:
			if target == sublist[0] - sublist[1]:
				# print(sublist[0], -sublist[1])
				l_result.append(sublist[0])
				l_result.append(-sublist[1])
				return True
			if target == sublist[0] + sublist[1]:
				# print(sublist[0], sublist[1])
				l_result.append(sublist[0])
				l_result.append(sublist[1])
				return True
			if target == -sublist[0] + sublist[1]:
				# print(-sublist[0], sublist[1])
				l_result.append(-sublist[0])
				l_result.append(sublist[1])
				return True
			if target == -sublist[0] - sublist[1]:
				# print(-sublist[0] , -sublist[1])
				l_result.append(-sublist[0])
				l_result.append(-sublist[1])
				return True
			# print('not find operation')
			return False
		else:
			if np.abs(target) <= np.sum(np.abs(sublist[:-1])):

				if find_operations((target + sublist[-1]), sublist[0:-1]):
					# print(-sublist[-1])
					l_result.append(-sublist[-1])
					return True
				elif find_operations((target - sublist[-1]), sublist[0:-1]):
					# print(sublist[-1])
					l_result.append(sublist[-1])
					return True
				else:
					return False
			else:
				return False
	else:
		return False
result = find_operations(1, l)
print(result)
if result:
	print(l_result)
	print(sum(l_result))
'''
'''转换成01背包问题'''
''' 给出所有可能组合
def findways(target, sublist):
	sum = np.sum(np.abs(sublist))
	if (sum + target) % 2 == 1 or target > sum:
		return
	W = (sum + target)// 2
	dp = np.zeros((np.size(sublist) + 1, W + 1), dtype= np.int)
	dp[:,  0] = 1
	for i in range(1, np.size(dp, axis= 0)):
		for j in range(1, np.size(dp, axis= 1)):
			if sublist[i -1] <= j:
				dp[i, j] = dp[i - 1, j] + dp[i-1, j - sublist[i-1]]
			else:
				dp[i, j] = dp[i - 1, j]
	return  dp
l = np.random.randint(1, 10, 6)
# l = np.arange(1, 51)
l = np.array([8, 5, 6, 8])
print(l)
dp = findways(1, l)
# if type(dp) != type(None):
if np.any(dp != None):
	# print('  ',np.arange(np.size(dp, axis= 1)))
	head_indx = np.arange(np.size(dp, axis=1), dtype = np.int).reshape(1, -1)
	l_p = np.hstack((np.array((0, 0)), l)).reshape(6, 1)
	print(head_indx.shape, dp.shape)
	dp_p = np.vstack((head_indx, dp))
	print(l_p.shape, dp_p.shape)
	dp_p = np.hstack((l_p, dp_p))
	print(dp_p)
else:
	print("no solution ")
'''
# ta = tf.ones((2, 3, 4))
# ta_sum = tf.reduce_sum(ta, axis=-1)
# print('ta_sum.shape: {}'.format(ta_sum.shape))
# print('ta_sum : {}'.format(ta_sum.numpy()))
#
# x = np.arange(24).reshape(2, 3, 4)
# print('x : {}'.format(x))
# print('x.sum-axis = 0 : {}'.format(x.sum(axis = 0)))
# print('x.sum-axis = 1 : {}'.format(x.sum(axis = 1)))
# @tf.function
# def f(x):
#   if x > 0:
#     import pdb
#     pdb.set_trace()
#     x = x + 1
#   return x
#
# tf.config.experimental_run_functions_eagerly(True)
# f(tf.constant(1))
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))


class MyLayer(keras.layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


model = tf.keras.Sequential([
    MyLayer(10)])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)





























