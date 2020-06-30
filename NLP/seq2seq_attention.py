import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import unicodedata
import re
from sklearn.model_selection import train_test_split
from tensorflow import keras
print(tf.__version__)
for module in mpl, np, pd, sklearn, tf, keras:
	print(module.__name__, module.__version__)
'''
    1.preprocessing data
    2.build model
    2.1 encoder
    2.2 attention
    2.3 decoder
    3. envaluation
    3.1 given sentence, return translated results
    3.2 visualize results(attention)
'''
en_spa_file_path = r'C:\Users\Administrator\jupyter_program_file\Chapter10\data_spa_en\spa.txt'

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    #标点符号前后加空格
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    #多余的空格变成一个空格
    s = re.sub(r'[" "]+', " ", s)
    #除了标点符号和字母外都是空格
    s = re.sub(r'[^a-zA-Z?.!,¿]', " ", s)
    #去掉前后空格
    s = s.rstrip().strip()
    #加收尾加标识符
    s = '<start> ' + s + ' <end>'
    return s
def parse_data(filename):
    lines = open(filename, encoding = "UTF_8").read().strip().split("\n")
    sentence_pairs = [line.split('\t')[0:2] for line in lines]
    preprocess_sentence_pairs = [(preprocess_sentence(en), preprocess_sentence(sp)) for en, sp in sentence_pairs]
    return zip(*preprocess_sentence_pairs)
en_dataset, sp_dataset = parse_data(en_spa_file_path)
def tokenizer(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(num_words = None, filters='', split=' ')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return  tensor, lang_tokenizer
input_tensor, input_tokenizer = tokenizer(sp_dataset[0: 30000])
output_tensor, output_tokenizer = tokenizer(en_dataset[0:30000])
def max_length(tensor):
    return max(len(t) for t in tensor)
max_length_input = max_length(input_tensor)
max_length_output = max_length(output_tensor)
print(max_length_input, max_length_output)

input_train, input_eval, output_train, output_eval = train_test_split(input_tensor, output_tensor, test_size=0.2)
len(input_train), len(input_eval), len(output_train), len(output_eval)
def convert(example, tokenizer):
    for t in example:
        if t!= 0:
            print('%d --> %s'%(t, tokenizer.index_word[t]))
convert(input_train[2], input_tokenizer)
print()
convert(output_train[2], output_tokenizer)
def make_dataset(input_tensor, output_tensor, batch_size, epochs, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset
batch_size = 64
epochs = 20
train_dataset = make_dataset(input_train, output_train, batch_size, epochs, True)
eval_dataset = make_dataset(input_eval, output_eval, batch_size, 1, False)
