import cv2
import os
import sys
input_dir = r'C:\Users\Administrator\Desktop\autoImages'
output_dir = r'C:\Users\Administrator\Desktop\autoImages_cut'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
input_images_list = os.listdir(input_dir)
for image in input_images_list:
	img = cv2.imread(os.path.join(input_dir, image))
	print(img.shape)
	cropped = img[:, 0:1080]
	cv2.imwrite(os.path.join(output_dir, image), cropped)
	# cropped = img[0:128, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]
## 文件重命名
# path = r'C:\Users\Administrator\Desktop\autoImages'
# filelist = os.listdir(path)
# for i,item in enumerate(filelist):
#     # print('item name is ',item)
#     if item.endswith('.jpg'):
#         name = item.split('.',1)[0]
#         src = os.path.join(os.path.abspath(path), item)
#         dst = os.path.join(os.path.abspath(path), str(i) + '.jpg')
#     try:
#         os.rename(src, dst)
#         print('rename from %s to %s' %(src, dst))
#     except:
#         continue