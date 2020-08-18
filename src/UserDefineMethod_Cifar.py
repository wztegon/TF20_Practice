import os
import re
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import tensorflow as tf
from Resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

lr = 0.0001
EPOCHS = 50

# 选择小模型进行测试
model = ResNet18()
optimizer = tf.keras.optimizers.Adam(lr)

# 对所有数据进行归一化 因为是测量的实际值 范围在200微米左右  这里除以100
sacla_rate = 100

# 损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# 选择指标: 损失和正确率
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

tst_loss = tf.keras.metrics.Mean(name='test_loss')
tst_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# 使用tf.GradientTape函数进行自定义训练过程
@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images, training=True)
		# print("=> label shape: ", labels.shape, "pred shape", predictions.shape)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	train_loss(loss)
	train_accuracy(labels, predictions)


@tf.function
def tst_step(images, labels):
	predictions = model(images)
	t_loss = loss_object(labels, predictions)
	tst_loss(t_loss)
	tst_accuracy(labels, predictions)


def get_input_filenames(input_dir):
	path_list = os.listdir(input_dir)  # 001 002 003...006
	train_filenames_list = []
	test_labels_lsit = []
	path_list = map(lambda name: os.path.join(input_dir, name), path_list)
	for full_label_path in path_list:
		label_name = os.path.split(full_label_path)[-1]
		file_names = os.listdir(full_label_path)
		for file_name in file_names:
			full_label_name = os.path.join(full_label_path, file_name)
			if os.path.isfile(full_label_name):
				if re.search(r'\d[6-9]\d.asc', full_label_name):
					test_labels_lsit.append(full_label_name)
				else:
					train_filenames_list.append(full_label_name)
	
	return train_filenames_list, test_labels_lsit


def process_path(file_path):
	label = tf.strings.to_number(tf.strings.split(file_path, '\\')[-2], out_type=tf.int32)
	data = tf.io.read_file(file_path)
	# 读取asc文件时最后一行为空 所以要去掉最后一个元素
	data_line = tf.strings.split(data, '\r\n')[0: -1]
	data_line = tf.strings.reduce_join(data_line, separator='\t')
	data_line = tf.strings.split(data_line, '\t')
	data_float = tf.map_fn(lambda s: tf.strings.to_number(s, out_type=tf.float32), data_line, dtype=tf.float32)
	data_re = tf.reshape(data_float, (1000, 300, 1))
	# 这里使用最大最小值是否对痕迹统一性有影响？？？
	data_re = (data_re - tf.reduce_min(data_re)) / (tf.reduce_max(data_re) - tf.reduce_min(data_re))
	return data_re, label


def load_cifar_file(filename):
	"""read dat from cifar file"""
	with open(filename, 'rb') as f:
		datadict = cPickle.load(f, encoding='bytes')
		return datadict[b'data'], datadict[b'labels']


def load_two_labels_file_data(filenames):
	"""load two kind label data"""
	data_0 = []
	label_0 = []

	for filename in filenames:

		data, label = load_cifar_file(filename)
		for single_data, single_label in zip(data, label):
			if single_label >= 0 and single_label <= 5:
				data_0.append(single_data)
				label_0.append(single_label)

	return np.array(data_0), np.array(label_0)


def main():
	# 准备数据 进行归一化等处理
	# 采用CIFAR10的6类作为数据
	CIFAR_DIR = r"C:\Users\Administrator\.keras\datasets\cifar-10-batches-py"
	data, label = load_two_labels_file_data([os.path.join(CIFAR_DIR, filename) for
	                                          filename in ["data_batch_{}".format(i) for i in range(1, 6)]])
	
	data = np.transpose(data.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
	data = data.astype(np.float32)
	#label = label.astype(np.float32)
	print(label.shape)
	a = set(label)
	print(a)
	print(data.shape)
	# train_set = zip(data[0:-100], label[0:-100])
	# test_set = zip(data[-100:], label[-100:])
	
	# print(data[1].shape)
	# image = data[10].reshape((3, 32, 32))
	# image = np.transpose(image, (1, 2, 0))
	# plt.imshow(image)
	# plt.show()
	# train_set = tf.data.Dataset.from_tensor_slices(train_dat)
	# test_set = tf.data.Dataset.from_tensor_slices(test_dat)
	# train_set.shuffle(3000)
	# test_set.shuffle(3000)
	# train_set = train_set.batch(16)
	# test_set = test_set.batch(16)
	batch_num = 116
	batch_size = 256
	for epoch in range(EPOCHS):
		for item in range(batch_num):
			train_step(data[item*batch_size : (item + 1)*batch_size], label[item*batch_size : (item + 1)*batch_size])
			
			
			tst_step(data[batch_size * batch_num:], label[batch_size * batch_num:])
			
			template = '=> Epoch {}, Batch {}, Loss: {:.4}, Accuracy: {:.2%}, tst Loss: {:.4}, tst Accuracy: {:.2%}'
			print(template.format(epoch + 1,
			                      item +1,
			                      train_loss.result(),
			                      train_accuracy.result(),
			                      tst_loss.result(),
			                      tst_accuracy.result()))
			# 重新设置指标
			train_loss.reset_states()
			train_accuracy.reset_states()
			tst_loss.reset_states()
			tst_accuracy.reset_states()


if __name__ == '__main__':
	main()