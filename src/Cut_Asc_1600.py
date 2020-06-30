import os
import numpy as np
def AscDirToMatrix(input_filedir, output_filedir):
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
    all_name_list = os.listdir(input_filedir)
    for name in all_name_list:
        filename = os.path.join(input_filedir, name)
        if os.path.isfile(filename) and ".asc" in name:#如果是asc文件就剪切并保存的行的文件夹中
            cut_save_asc_1600(filename, output_filedir)
        if os.path.isdir(filename):
            AscDirToMatrix(filename, os.path.join(output_filedir, os.path.split(filename)[-1]))
         
         
def cut_save_asc_1600(filename, output_filedir):
    with open(filename, 'r') as ascFile:
        cut_size = 1600
        reader = ascFile.readlines()
        head = reader[0:16]
        head[4] = head[4][:-len(head[4].split(r' ')[-1])] + "1600\n"
        head[5] = head[5][:-len(head[5].split(r' ')[-1])] + "1600\n"
        
        data_str = reader[16:]
        data_float = []
        for line in data_str:
            L = line.split('\t')
            data_float.append([float(x) for x in L])
        data_float = np.array(data_float)
        if len(data_float) >= cut_size and len(data_float[0]) >= cut_size:
            cut_height_start = (len(data_float) -cut_size)// 2
            cut_height_end = cut_height_start + cut_size
            cut_length_start = (len(data_float[0]) - cut_size) // 2
            cut_length_end = cut_length_start + cut_size
            data_2 = data_float[cut_height_start:cut_height_end, cut_length_start:cut_length_end]
            output_filename = os.path.join(output_filedir, os.path.split(filename)[-1])
            save_asc_1600(output_filename, head, data_2)
        else:
            ascFile.close()
            
def save_asc_1600(full_filename, heads, datas):
    with open(full_filename, "wt") as f:
        f.writelines(heads)
        for row_index in range(datas.shape[0]):
            f.write("\t".join([repr(col) for col in datas[row_index]]))
            f.write("\n")
    f.close()
def main():
    input_filedir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对3000x2000'
    output_filedir = r'C:\Users\Administrator\Desktop\马尔数据-单体比对1600x1600'
    AscDirToMatrix(input_filedir, output_filedir)
if __name__ == '__main__':
    main()