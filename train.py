from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from libs.DataGenerator import MaskedDataGenerator
from libs.PartialConvUNet import PartialConvUNet
from libs.util import torch_preprocessing
import matplotlib.pyplot as plt
import numpy as np
TRAIN_DIR   = "E:\\Python\\data_256"
VAL_DIR     = "E:\\Python\\val_256"
VGG16_WEIGHTS   = r'E:\Python\Image-inpainting-using-PConv\vgg16_pytorch2keras.h5'
WEIGHTS_DIR     = "weights/"

BATCH_SIZE      = 4
STEPS_PER_EPOCH = 5000
EPOCHS_STAGE1   = 13
EPOCHS_STAGE2   = 13
STEPS_VAL       = 10
IMAGE_SHAPE      = (256, 256)
STAGE_1         = True # Initial training if True, Fine-tuning if False 
LOAD            = True
LAST_CHECKPOINT =  WEIGHTS_DIR + "initial/weights.13-4.41-2.23.hdf5"

train_datagen = MaskedDataGenerator(preprocessing_function=torch_preprocessing, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE
)
val_datagen = MaskedDataGenerator(preprocessing_function=torch_preprocessing)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    seed=2137,
    shuffle=False
)

if STAGE_1:
	lr = 0.0002
	fine_tuning = False
	initial_epoch = 13
	stage_name = 'initial/'
	epochs = 13+10
else: 
	lr = 0.00005
	fine_tuning = True
	initial_epoch = EPOCHS_STAGE1
	stage_name = 'fine_tuning/'
	epochs = EPOCHS_STAGE2
model = PartialConvUNet(fine_tuning=fine_tuning, lr=lr, image_shape=IMAGE_SHAPE, vgg16_weights=VGG16_WEIGHTS)
if LOAD:
	model.load_weights(LAST_CHECKPOINT)
model.fit(
	train_generator,
	steps_per_epoch=STEPS_PER_EPOCH,
	initial_epoch=initial_epoch,
	epochs=epochs,
	validation_data=val_generator,
	validation_steps=STEPS_VAL,
	callbacks=[
		ModelCheckpoint(WEIGHTS_DIR+stage_name+'weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
	]
)
