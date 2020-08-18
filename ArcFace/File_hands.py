import os
import shutil
def move_single_file(old_path, new_path):
	
	persions_list = os.listdir(old_path)
	for single_persion in persions_list:
		single_persion_dir = os.path.join(old_path, single_persion)
		single_persion_list = os.listdir(single_persion_dir)
		
		if 1 == len(single_persion_list):
			new_dir = os.path.join(new_path, single_persion_list[0])
			if not os.path.exists(new_dir):
				shutil.move(single_persion_dir, new_path)
		
		# shutil.move()
def main():
	old_path = r"E:\lfw-deepfunneled\multiple_images"
	new_path = r"E:\lfw-deepfunneled\single_images"
	move_single_file(old_path, new_path)
if __name__ == '__main__':
	main()
