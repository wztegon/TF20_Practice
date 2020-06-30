import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
print(tf.__version__)
for module in mpl, np, pd, sklearn, tf, keras:
	print(module.__name__, module.__version__)
class MyModel(tf.keras.models):
	def __init__(self, input):
		super(MyModel, self).__init__()
		self.Conv1 = keras.layers.Conv2D(filter=64, kernel_size=3, strides=1, padding='Vaild')
		self.Conv2 = keras.layers.Conv2D(filter=128, kernel_size=3, strides=1, padding='Vaild')
		self.Conv2 = keras.layers.Conv2D(filter=128, kernel_size=3, strides=1, padding='Vaild')
		self.maxpool1 = keras.layers.MaxPooling2D(3)