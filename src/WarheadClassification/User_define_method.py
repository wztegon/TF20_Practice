import os
import re
import numpy as np
import tensorflow as tf
from Resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

lr = 0.0001
EPOCHS = 10

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

def main():
    #准备数据 进行归一化等处理
    input_dir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对300x1000\0209060559'
    train_filenames, test_filenames = get_input_filenames(input_dir)
    train_files_ds = tf.data.Dataset.list_files(train_filenames)
    test_files_ds = tf.data.Dataset.list_files(test_filenames)
    train_set = train_files_ds.map(process_path)
    test_set = test_files_ds.map(process_path)
    train_set.shuffle(3000)
    test_set.shuffle(3000)
    train_set = train_set.batch(1)
    test_set = test_set.batch(1)
    
    
    for epoch in range(EPOCHS):
        for images, labels in train_set:
            train_step(images, labels)
    
        for tst_images, tst_labels in test_set:
            tst_step(tst_images, tst_labels)
    
        template = '=> Epoch {}, Loss: {:.4}, Accuracy: {:.2%}, tst Loss: {:.4}, tst Accuracy: {:.2%}'
        print(template.format(epoch+1,
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