import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
from PIL import Image
import cv2
import os
import tensorflow as tf
print(tf.__version__)
#加载并解析asc文件返回头部和点云数据
def AscToMatrix(filename):
    with open(filename,'r') as ascFile:
        reader = ascFile.readlines()
        head = reader[0:16]
        data_str = reader[16:]
        data_float = []
        for line in  data_str:
            L = line.split('\t')
            data_float.append([float(x) for x in L])
    ascFile.close()
    return head,data_float
#save result img as asc file
def save_asc(full_filename, heads, datas):
    with open(full_filename, "wt", encoding= "utf-8") as f:
        f.writelines(heads)
        for row_index in range(datas.shape[0]):
            f.write("\t".join([repr(col) for col in datas[row_index]]))
            f.write("\n")
    f.close()
    
   
input_dir = 'C:\\Users\\Administrator\\Desktop\\ASC_Filter'
filename_input = "001.asc"
full_filename = os.path.join(input_dir, filename_input)
heads, input_list = AscToMatrix(full_filename)
input_list = np.array(input_list, dtype=np.float32)
#输入的asc文件进行归一化和显示
input_nparray = np.array(input_list)
input_nparray = (input_nparray - input_nparray.min())/(input_nparray.max() - input_nparray.min()) * 255
#BilateralFilter
bilateralFilter_img =  cv2.bilateralFilter(input_nparray, 3, 80, 80)
print(bilateralFilter_img.max())
print(bilateralFilter_img.min())
#diff_img of input and bilateralfilter
result_img = input_nparray - bilateralFilter_img
print(result_img.max())
print(result_img.min())
#show the image of three steps
fig = plt.figure(figsize=(24, 8))
ax0 = fig.add_subplot(131)
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)
ax0.imshow(input_nparray, cmap=plt.cm.gray, interpolation='nearest')
ax1.imshow(bilateralFilter_img, cmap=plt.cm.gray, interpolation='nearest')
ax2.imshow(result_img, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
print(result_img.shape)

#高斯平滑滤波器
filter_smooth = np.array([[np.math.exp(-4),np.math.exp(-2.5),np.math.exp(-2),np.math.exp(-2.5),np.math.exp(-4)],
           [np.math.exp(-2.5),np.math.exp(-1),np.math.exp(-0.5),np.math.exp(-1),np.math.exp(-2.5)],
           [np.math.exp(-2),np.math.exp(-0.5), 1, np.math.exp(-0.5),np.math.exp(-2)],
           [np.math.exp(-2.5),np.math.exp(-1),np.math.exp(-0.5),np.math.exp(-1),np.math.exp(-2.5)],
           [np.math.exp(-4),np.math.exp(-2.5),np.math.exp(-2),np.math.exp(-2.5),np.math.exp(-4)]])
filters_smooth = filter_smooth[:,:,np.newaxis,np.newaxis]

# 保存差异文件前需要做一下高斯平滑
# save_asc(full_filename_output, heads, diff_data)
#平滑滤波前也需要做padding操作 使用原始数据进行填充
input_smooth_padding_updown = np.r_[result_img[:2, :], result_img, result_img[-2:, :]]
input_smooth_padding_udlr = np.c_[input_smooth_padding_updown[:, :2],
                                  input_smooth_padding_updown,
                                  input_smooth_padding_updown[:, -2:]]
input_smooth = input_smooth_padding_udlr[np.newaxis, :, :, np.newaxis]
output_smooth = tf.nn.conv2d(input_smooth, filters= filters_smooth, strides=[1, 1, 1, 1], padding="VALID")
output_smooth_nparray= output_smooth.numpy().reshape(input_nparray.shape)

# output_smooth_nparray = (output_smooth_nparray - output_smooth_nparray.min())/(
# 		output_smooth_nparray.max() - output_smooth_nparray.min()) * 255
# 定义处处文件的路径 和 文件保存函数
output_path_ext = os.path.splitext(full_filename)
output_asc_full_filename = output_path_ext[0] + '.asd'
save_asc(output_asc_full_filename, heads, output_smooth_nparray)
