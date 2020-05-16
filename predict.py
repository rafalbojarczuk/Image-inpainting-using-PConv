from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.preprocessing.image import load_img, img_to_array
from libs.DataGenerator import MaskedDataGenerator
from libs.PartialConvUNet import PartialConvUNet
from libs.util import torch_preprocessing, torch_postprocessing
import matplotlib.pyplot as plt
import numpy as np

TEST_DIR     = "E:\\Python\\test_256"
VGG16_WEIGHTS   = r'E:\Python\Image-inpainting-using-PConv\vgg16_pytorch2keras.h5'
WEIGHTS_DIR     = "weights/"

BATCH_SIZE      = 4
IMAGE_SHAPE      = (256, 256)

LAST_CHECKPOINT =  WEIGHTS_DIR + "initial/weights.23-4.40-2.06.hdf5"

test_datagen = MaskedDataGenerator(preprocessing_function=torch_preprocessing)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE
)

model = PartialConvUNet(predict_only=True, image_shape=IMAGE_SHAPE)
model.load_weights(LAST_CHECKPOINT)


(input_img, mask), orig_img = next(test_generator)
output_img = model.predict([input_img, mask])

# Post-processing:
orig_img   = torch_postprocessing(orig_img)
input_img  = torch_postprocessing(input_img) * mask # the (0,0,0) masked pixels are made grey by post-processing
output_img = torch_postprocessing(output_img)
output_comp = input_img.copy()
output_comp[mask == 0] = output_img[mask == 0]

fig, axes = plt.subplots(input_img.shape[0], 3, figsize=(15, 29))
for i in range(input_img.shape[0]):
    #axes[i,0].imshow(orig_img[i])
    axes[i,0].imshow(input_img[i])
    axes[i,1].imshow(output_img[i])
    axes[i,2].imshow(orig_img[i])
    axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
axes[0,0].set_title('Masked image')
axes[0,1].set_title('Prediction')
plt.tight_layout()
plt.show()