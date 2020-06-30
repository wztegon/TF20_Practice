import tensorflow as tf
import os
import re
import numpy as np

def get_input_filenames(input_dir):
	path_list = os.listdir(input_dir)#001 002 003...006
	train_filenames_list = []
	test_labels_lsit = []
	path_list = map(lambda name :os.path.join(input_dir, name), path_list)
	for full_label_path in path_list:
		label_name = os.path.split(full_label_path)[-1]
		file_names = os.listdir(full_label_path)
		for file_name in file_names:
			full_label_name = os.path.join(full_label_path, file_name)
			if os.path.isfile(full_label_name):
				if re.search(r'\d[6-9]\d.asc', full_label_name) :
					test_labels_lsit.append(full_label_name)
				else:
					train_filenames_list.append(full_label_name)
			
	return train_filenames_list, test_labels_lsit

#对所有数据进行归一化 因为是测量的实际值 范围在200微米左右  这里除以100
sacla_rate = 100
def process_path(file_path):
	label = tf.strings.to_number(tf.strings.split(file_path, '\\')[-2], out_type = tf.int32 )
	data = tf.io.read_file(file_path)
	#读取asc文件时最后一行为空 所以要去掉最后一个元素
	data_line = tf.strings.split(data, '\r\n')[0: -1]
	data_line = tf.strings.reduce_join(data_line, separator = '\t')
	data_line = tf.strings.split(data_line, '\t')
	data_float = tf.map_fn(lambda s:tf.strings.to_number(s, out_type=tf.float32), data_line, dtype = tf.float32)
	data_re = tf.reshape(data_float, (1000, 300, 1))
	data_re = (data_re - tf.reduce_min(data_re)) / (tf.reduce_max(data_re) - tf.reduce_min(data_re))
	return data_re, label

def main():
	input_dir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对300x1000\0209060559'
	train_filenames, test_filenames = get_input_filenames(input_dir)
	train_files_ds = tf.data.Dataset.list_files(train_filenames)
	test_files_ds = tf.data.Dataset.list_files(test_filenames)
	train_set = train_files_ds.map(process_path)
	test_set = test_files_ds.map(process_path)
	train_set.shuffle(3000)
	test_set.shuffle(3000)
	train_set = train_set.batch(4)
	test_set = test_set.batch(4)
	model = tf.keras.models.Sequential([
		#after this layer the tensor shape: (, 499, 298, 64)
		tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 1), activation='selu', input_shape=(1000, 300, 1)),
		#after this maxpooling the tensor shape: (, 249, 149, 64)
		tf.keras.layers.MaxPooling2D(2, 2),
		#after this layer the tensor shape: (, 247, 147, 128)
		tf.keras.layers.Conv2D(128, (3, 3), activation='selu'),
		#after maxpooling the tensor shape: (, 123, 73, 128)
		tf.keras.layers.MaxPooling2D(2, 2),
		#after this layer the tensor shape: (, 121, 71, 128)
		tf.keras.layers.Conv2D(128, (3, 3), activation='selu'),
		#after maxpooling the tensor shape: (, 60, 35, 128)
		tf.keras.layers.MaxPooling2D(2, 2),
		#after this layer the tensor shape: (, 58, 33, 256)
		tf.keras.layers.Conv2D(256, (3, 3), activation='selu'),
		#after the last maxpooling the tensor shape: (, 29, 16, 256)
		tf.keras.layers.MaxPooling2D(2, 2),
		#after flatten layre the tensor change to one dimension :(118784)
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='selu'),
		tf.keras.layers.Dense(6, activation='softmax')
	])
	model.summary()
	# tf.keras.utils.plot_model(model, 'asc_classify_model_with_shape_info.png', show_shapes=True)
	model.compile(optimizer= tf.optimizers.Adam(learning_rate= 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	logdir =r'C:\Users\Administrator\Desktop\1000x300_classify_result'
	if not os.path.exists(logdir):
	    os.mkdir(logdir)
	output_model_file = os.path.join(logdir, "1000x300_classify_result.h5")
	callbacks = [
	    tf.keras.callbacks.TensorBoard(logdir),
	    tf.keras.callbacks.EarlyStopping(monitor= "loss",patience=3,min_delta=1e-3),
	    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only= True)
	   ]
	history = model.fit(train_set, epochs=200,
	                    validation_data=test_set, callbacks=callbacks)
	

	# train_set = csv_reader_dataset(train_filenames, batch_size=3)
	# print(len(train_filenames), '\n', len(test_filenames))


if __name__ == '__main__':
	main()