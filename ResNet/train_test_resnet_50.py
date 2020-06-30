import tensorflow as tf
import tensorflow.keras as keras
import resnet_model

Num_classes = 6
train_epochs = 10
steps_per_epoch = 16
num_eval_steps = 16
#dataset
(train_images, train_lables), (test_images, test_lables) = tf.keras.datasets.cifar10.load_data()
train_images, valid_images = train_images[: 45000], train_images[45000:]
train_lables, valid_lables = train_lables[: 45000], train_lables[45000:]
print(train_images.shape, valid_images.shape)
print(train_lables.shape, valid_lables.shape)


# model = resnet_model.resnet50(
# 	num_classes= Num_classes)
# model.compile(
# 	loss='sparse_categorical_crossentropy',
# 	optimizer='adam',
# 	metrics=['accuracy'])
# callbacks =  [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
# history = model.fit(train_images,
#                     train_lables,
#                     epochs=train_epochs,
#                     steps_per_epoch=steps_per_epoch,
#                     callbacks=callbacks,
#                     validation_steps=num_eval_steps,
#                     validation_data=(),
#                     verbose=2)