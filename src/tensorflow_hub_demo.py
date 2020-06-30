import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
# image_tensor = tf.zeros(shape=[1,96,96,3])
# num_classes = 20
# handle = "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_050_96/feature_vector/4"
# m = hub.load(handle)
# print(list(m.signatures.keys()))
# result = m(image_tensor)
# print(result)
# print(type(result))
# print(result.shape)
tf_array = tf.ones((3, 3), dtype=tf.float32)
print(tf_array)
tf_sub = (tf_array - 0.5) / 0.3
print(tf_sub)
