import  tensorflow as tf

flowers_root =r'C:\Users\Administrator\Desktop\马尔数据-单体比对300x1000\0209060559'
list_ds = tf.data.Dataset.list_files(flowers_root)

# def process_path(file_path):
#   label = tf.strings.split(file_path, '/')[-2]
#   return tf.io.read_file(file_path), label
# labeled_ds = list_ds.map(process_path)

for f in list_ds.take(5):
  print(f)
  
# for image_raw, label_text in labeled_ds.take(1):
#   print(repr(image_raw.numpy()[:100]))
#   print()
#   print(label_text.numpy())