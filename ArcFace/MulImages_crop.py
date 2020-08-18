import os
from PIL import Image
x_start = 15
y_start = 15
w = 224
h = 224
def crop_images(origin_dir, crop_dir):
	persions_list = os.listdir(origin_dir)
	for single_persion in persions_list:
		single_persion_dir = os.path.join(origin_dir, single_persion)
		crop_single_persion_dir = os.path.join(crop_dir, single_persion)#every persion dir after crop
		single_persion_all_images = os.listdir(single_persion_dir)
		for single_persion_single_image in single_persion_all_images:
			image_dir = os.path.join(single_persion_dir, single_persion_single_image)
			im = Image.open(image_dir)
			cut_image = im.crop((x_start, y_start, x_start + w, y_start + h))#old image shape:(255, 255)
			#crop_dir = os.path.join(crop_dir, single_persion_single_image)#crop image dir
			if not os.path.exists(crop_single_persion_dir):
				os.mkdir(crop_single_persion_dir)
			crop_image_path = os.path.join(crop_single_persion_dir, single_persion_single_image)
			cut_image.save(crop_image_path)
			

def main():
	origin_dir = r"E:\lfw-deepfunneled\multiple_images"
	cut_dir = r"E:\lfw-deepfunneled\multiple_images_crop"
	crop_images(origin_dir, cut_dir)
if __name__ == '__main__':
	main()