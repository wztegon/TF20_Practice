import sys
import os
import glob
import xml.etree.ElementTree as ET

class_id = {'drillrod': 0,}
# create yolov3 format files
xml_list = glob.glob(r'C:\Users\Administrator\Desktop\autoImages_resize_label\*.xml')
if len(xml_list) == 0:
  print("Error: no .xml files found in ground-truth")
  sys.exit()
L = []
for tmp_file in xml_list:
  #print(tmp_file)
  # 1. create new file (yolov3 format)
  with open(tmp_file, "r") as new_f:
    root = ET.parse(tmp_file).getroot()
    for obj in root.findall('object'):
      obj_name = obj.find('name').text
      bndbox = obj.find('bndbox')
      left = bndbox.find('xmin').text
      top = bndbox.find('ymin').text
      right = bndbox.find('xmax').text
      bottom = bndbox.find('ymax').text
      tem_str = tmp_file + " " + left + "," + top + "," + right + "," + bottom + "," + str(class_id[obj_name]) + '\n'
      L.append(tem_str)
      # new_f.write(obj_name + " " + left + " " + top + " " + right + " " + bottom + '\n')
with open(r'C:\Users\Administrator\Desktop\myname.txt', "wt") as f:
  for line in L:
    f.write(line)
print("Conversion completed!")