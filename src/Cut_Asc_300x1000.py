import os
import numpy as np


def AscDirToMatrix(input_filedir, output_filedir, offset):
	if not os.path.exists(output_filedir):
		os.mkdir(output_filedir)
		
	all_name_list = os.listdir(input_filedir)
	for name in all_name_list:
		filename = os.path.join(input_filedir, name)
		if os.path.isfile(filename) and ".asc" in name:  # 如果是asc文件就剪切并保存的行的文件夹中
			output_last_dir = '%03d'%((int(os.path.splitext(name)[-2]) - 1 - offset)%6 +1)
			print(os.path.splitext(name)[-2])
			output_filedir_bullet = os.path.join(output_filedir, output_last_dir)
			
			cut_save_asc_300x1000(filename, output_filedir_bullet)
			
r'''output_filedir:  0209060559\001  0209060559\002  ...  0209060559\006'''
r'''最后生成的文件：100.asc 101.asc 110.asc ... 390.asc 391.asc 其中百分位表示这是第几个弹头 十分位表示列分快数 个位表示行分块数'''
def cut_save_asc_300x1000(filename, output_filedir):#每次生成按顺序
	if not os.path.exists(output_filedir):
		os.mkdir(output_filedir)
	with open(filename, 'r') as ascFile:
		item_bullet = filename.split('\\')[-2]
		item_bullet = item_bullet.split('-')[-1]
		reader = ascFile.readlines()
		data_str = reader[16:]
		data_float = []
		for line in data_str:
			L = line.split('\t')
			data_float.append([float(x) for x in L])
		data_float = np.array(data_float)
		for i in range(2):
			for j in range(10):
				data_300x1000 = data_float[i::2 , j::10]
				output_full_filename = os.path.join(output_filedir, '%03d.asc' %(100*int(item_bullet) + j*10  + i))
				save_asc_300x1000(output_full_filename, data_300x1000)
		
def save_asc_300x1000(full_filename, datas):
	with open(full_filename, "wt") as f:
		for row_index in range(datas.shape[0]):
			f.write("\t".join([repr(col) for col in datas[row_index]]))
			f.write("\n")
	f.close()


def main():
	input_filedir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对3000x2000\0209060559'
	bullet_num = 3#the num of bullets from the same gun
	rifling_num = 6#the num of rifling lines of a gun
	input_filedir_list = []
	for item in range(1, bullet_num + 1):
		input_filedir_list.append(input_filedir + '-' + str(item))
	bullet_num_offset = [0, 0, 3]
	output_filedir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对300x1000\0209060559'
	for item ,input_dir in enumerate(input_filedir_list):
		AscDirToMatrix(input_dir, output_filedir, bullet_num_offset[item])
	
	


if __name__ == '__main__':
	main()