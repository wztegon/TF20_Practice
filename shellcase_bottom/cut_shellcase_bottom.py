import cv2
import os
import sys
input_dir = r'C:\Users\Administrator\Desktop\0109009044_B141959'
output_dir = r'C:\Users\Administrator\Desktop\0109009044_B141959_cut'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
input_images_list = os.listdir(input_dir)
print(input_images_list)
for image in input_images_list:
	print(os.path.join(input_dir, image))
	img = cv2.imread(os.path.join(input_dir, image))
	print(img.shape)
	start = int((img.shape[1] - img.shape[0])/2)
	end = start + img.shape[0]
	cropped = img[:, start:end]
	print(cropped.shape)
	cv2.imwrite(os.path.join(output_dir, image), cropped)