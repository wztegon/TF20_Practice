import numpy as np
import _pickle as cPickle
import os
import matplotlib.pyplot as plt



CIFAR_DIR = r"C:\Users\Administrator\.keras\datasets\cifar-10-batches-py"

layer_neure_nums = np.array([6, 5, 4, 3, 2, 1])#the nums of neure of each layer
weight_lst = []  # save each layer weight
bias_lst = []  # save each layer bias
activate_fun = ["relu", "relu", "relu", "relu", "relu", "sigmoid"]
Z_all_layer = []
A_all_layer = []
epochs = 100
learn_rate = 0.01

def load_cifar_file(filename):
	"""read dat from cifar file"""
	with open(filename, 'rb') as f:
		datadict = cPickle.load(f, encoding='bytes')
		return datadict[b'data'], datadict[b'labels']
def load_two_labels_file_data(filenames):
	"""load two kind label data"""
	data_0  = []
	label_0 = []
	data_1  = []
	label_1 = []
	for filename in filenames:
		if len(data_0) == 100 and len(data_1) == 100:
			break
		data, label = load_cifar_file(filename)
		for single_data, single_label in zip(data, label):
			if single_label == 0 and len(data_0) <100 :
				data_0.append(single_data)
				label_0.append(single_label)
			elif single_label == 1 and len(data_1) <100:
				data_1.append(single_data)
				label_1.append(single_label)
			if len(data_0) == 100 and len(data_1) == 100:
				break
		
	return np.array(data_0), np.array(label_0), np.array(data_1), np.array(label_1)



def sigmoid(Z):
	if Z >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
		return 1.0 / (1 + np.exp(-Z))
	else:
		return np.exp(Z) / (1 + np.exp(Z))

def relu(Z):
	return np.maximum(0, Z)


def sigmoid_backward(dA, Z):
	sig = sigmoid(Z)
	return dA * sig * (1 - sig)


def relu_backward(dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z <= 0] = 0;
	return dZ;


def creat_weight_bias(lst, input_shape):
	"""init weight and bias of each layer
		n_back: back层的神经元个数
		n_curr：当前层的神经元个数
	"""
	n_back = input_shape
	for i, n_curr in enumerate(lst):
		weight = np.random.normal(0, 1, n_curr * n_back).reshape(n_curr, n_back)
		weight_lst.append(weight)
		bias = np.random.normal(0, 1, n_curr).reshape(n_curr, 1)
		bias_lst.append(bias)
		n_back = n_curr
		# weight = np.random.randn(n_curr, n_back) * 0.1
		# weight_lst.append(weight)
		# bias = np.random.randn(n_curr, 1) * 0.1
		# bias_lst.append(bias)
		#n_back = n_curr
		



def forward(lst, sameples):
	"""neure network forward compute"""

	for sameple in sameples:
		a_back = sameple.reshape(-1, 1)
		

		for i, element in enumerate(lst):
			Z = np.matmul(weight_lst[i], a_back) + bias_lst[i]
			
			#Z_single_layer.append(Z)
			if activate_fun[i] == "relu":
				a = relu(Z)
			elif activate_fun[i] == "sigmoid":
				a = sigmoid(Z)
			else:
				raise Exception('Non-supported activation function')
			#A_single_layer.append(a)
			a_back = a

			
			Z_all_layer.append(Z)
			A_all_layer.append(a)
	

def single_layer_backward(A_back, Z_curr, W_pre, dZ_pre, activate='relu'):
	"""这里采用的SGD 最后一位表示传入的总体样本数目m 当前层的神经元个数为n_curr
	:param A_back: 上一层的激活值 shape:(n_back, m)
	:param Z_curr: 当前层的Z值   shape:(n_curr, m)
	:param W_pre: 下一层的W      shape:(n_pre, n_curr)
	:param dZ_pre: 下一层的dZ    shape:(n_pre, m)
	:param activate: 当前层所用的激活函数
	:return:dZ_curr 需要传到上一次做运算
	        dW_curr, db_curr 用于更新当前层的W,b参数 
	"""
	m = A_back.shape[1]
	#计算后的shape：（n_curr, m）
	# print("W_pre shape: {0}".format(W_pre.shape))
	# print("dZ_pre shape: {0}".format(dZ_pre.shape))
	# print("Z_curr shape: {0}".format(Z_curr.shape))
	
	dA_curr = np.matmul(W_pre.T, dZ_pre)
	
	if activate == "relu":
		g_apostrophe = relu_backward(dA_curr, Z_curr)# g_apostrophe = g'
	elif activate == "sigmoid":
		g_apostrophe = sigmoid_backward(dA_curr, Z_curr)
	else:
		raise Exception('Non-supported activation function')
	#计算后的shape:(n_curr, m)
	dZ_curr = dA_curr * g_apostrophe
	#计算后的shape:(n_curr, n_back)
	dW_curr = np.matmul(dZ_curr, A_back.T)/m
	#计算后的shape:(n_curr, 1)
	db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
	return dZ_curr, dW_curr, db_curr


def full_backward(a_first, A, Y):
	"""最后一层的参数需要自己计算"""
	m = Y.shape[1]
	#最后
	W_last = np.array(weight_lst[-1])
	
	
	dZ_last = A-Y
	
	W_pre, dZ_pre = W_last, dZ_last


	for i in range(layer_neure_nums.shape[0] -2 , -1, -1):
		if i == 0:
			A_back = a_first
		else:
			A_back = np.array(A_all_layer[i-1::6]).reshape(m, -1).T
		Z_curr = np.array(Z_all_layer[i::6]).reshape(m, -1).T

		dZ_curr, dW_curr, db_curr = \
			single_layer_backward(A_back, Z_curr, W_pre, dZ_pre, activate=activate_fun[i])
		W_pre = weight_lst[i]
		dZ_pre = dZ_curr
		updata(dW_curr, db_curr, i)
	

def updata(dW_curr, db_curr, i_layer):
	"""更新参数"""
	weight_lst[i_layer] -= learn_rate * dW_curr
	#print("dW_curr shape: {0}".format(dW_curr.shape))
	bias_lst[i_layer] -= learn_rate * db_curr
	#print("db_curr shape: {0}".format(db_curr.shape))
	
def train(input_datas, input_labels):
	"""训练过程"""
	#初始化参数
	m = input_labels.shape[1]
	input_shape = input_datas.shape[1]
	creat_weight_bias(layer_neure_nums, input_shape)
	
	
	a_first = input_datas.T
	
	for i in range(epochs):
		forward(layer_neure_nums, input_datas)
		a_last = np.array(A_all_layer[5::6]).reshape(m, -1).T
	
		
		full_backward(a_first, a_last, input_labels)
		
		cost = get_cost_value(a_last, input_labels)
		accuracy = get_accuracy_value(a_last, input_labels)
		print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
		#清空本轮保存的Z和A
		A_all_layer.clear()
		Z_all_layer.clear()
		
def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    print("m 的值：{}".format(m))
    # calculation of the cost according to the formula
    cost = (-1.0 / m) * (np.sum(Y* np.log(Y_hat)) + np.sum((1.0 - Y)* np.log(1.0 - Y_hat)))
    return cost

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
def train_tf(input_datas, input_labels):
	""""""
	import tensorflow as tf
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(6, input_dim=3072,activation='relu'),
		tf.keras.layers.Dense(5, activation='relu'),
		tf.keras.layers.Dense(4, activation='relu'),
		tf.keras.layers.Dense(3, activation='relu'),
		tf.keras.layers.Dense(2, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')]
	)
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
	history = model.fit(input_datas, input_labels, epochs=200, verbose=1)


def main():

	data_0, label_0, data_1, label_1 = load_two_labels_file_data([os.path.join(CIFAR_DIR, filename) for
	                                          filename in ["data_batch_{}".format(i) for i in range(1, 6)]])
	input_datas = np.concatenate([data_0, data_1], axis=0)
	input_labels = np.concatenate([label_0, label_1], axis=0)
	# print(input_datas.shape)
	# print(input_labels.shape)
	
	input_datas = input_datas/127.5 - 1
	p = np.random.permutation(input_labels.shape[0])
	input_datas = input_datas[p]
	input_labels = input_labels[p]
	#train_tf(input_datas, input_labels)
	train(input_datas, input_labels.reshape(1, -1))
	
	# #init weight and bias
	# input_shape = input_datas.shape[1]
	# creat_weight_bias(layer_neure_nums, input_shape)
	# #forward compute
	# all_a = forward(layer_neure_nums, input_datas[0:6])
	# print(all_a[1::6])
	# print(Z_all_layer[0][2])
	# a = np.array(A_all_layer[3::6])
	# print(a.shape)
if __name__ == '__main__':
	main()