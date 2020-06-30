import os
from PIL import Image
input_dir = r'C:\Users\Administrator\Desktop\autoImages_cut'
output_dir = r'C:\Users\Administrator\Desktop\autoImages_resize'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
size=(448,448)

input_images_list = os.listdir(input_dir)

#遍历
for img_name in input_images_list:
    pri_image = Image.open(os.path.join(input_dir, img_name))
    tmppath = os.path.join(output_dir, img_name)

    #保存缩小的图片
    pri_image.resize(size, Image.ANTIALIAS).save(tmppath)