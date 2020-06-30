import numpy as np
import _pickle as cPickle
import os
import matplotlib.pyplot as plt


CIFAR_DIR = r"C:\Users\Administrator\.keras\datasets\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))
def load_cifar_file(filename):
	"""read dat from cifar file"""
	with open(filename, 'rb') as f:
		datadict = cPickle.load(f, encoding='bytes')
		print(type(datadict))
		print(datadict.keys())
		print(type(datadict[b'data']))
		print(datadict[b'data'].shape)
		return datadict[b'data'], datadict[b'labels']
data, labels = load_cifar_file(os.path.join(CIFAR_DIR, "data_batch_1"))
"""
labels_all = []
for i in range(10):
	labels_i = [value for value in labels if value ==i]
	labels_all.append(len(labels_i))
	print(len(labels_i))
print("labels  nums: ", len(labels_all))
print("labels sum: ", sum(labels_all))
print(labels_all)
"""
single_image = data[300].reshape((3, 32, 32))
single_image = single_image.transpose((1, 2, 0))
plt.imshow(single_image)
plt.show()