import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# porkets_file = r"C:\Users\Administrator\Desktop\测试数据.txt"
porkets_file = r"C:\Users\Administrator\Desktop\temp_data.txt"
out_temp_narray_file = r"C:\Users\Administrator\Desktop\temp_fire_narray.txt"
with open(porkets_file, 'r') as pf:
	con = pf.read()
	# con = con.replace(r'[\n]', ' ')
	temp_nums = con.split(' ')
# with open(out_temp_narray_file, 'w') as out_file:
# 	for i in range(120):
# 		for j in range(160):
# 			out_file.write(temp_nums[i*160 + j])
# 			out_file.write(' ')
# 		out_file.write('\n')
	
	
temp_nums = [float(temp) for temp in temp_nums]
temp_narray = np.array(temp_nums, dtype=np.float16)



print(len(temp_narray))
print(np.max(temp_narray), np.min(temp_narray))
# temp_narray = temp_narray.reshape((120, 160))
# f, ax = plt.subplots(figsize=(8, 6))
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
# sns.heatmap(temp_narray, cmap=cmap, linewidths=.05, ax=ax)
# plt.show()