import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from libs.util import MaskGenerator, torch_preprocessing, torch_postprocessing
from libs.DataGenerator import MaskedDataGenerator
 # mask generation example


generator = MaskGenerator(256, 256, 3)

# get mask
mask = generator.sample()

# display with opencv
cv2.imshow('mask displayed with opencv', mask*255)
print("Cheeck")

# get many masks
masks = np.stack([
    generator.sample()
    for _ in range(3)], axis=0
)

# display them with matplotlib
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(masks[0,:,:,:]*255)
axes[1].imshow(masks[1,:,:,:]*255)
axes[2].imshow(masks[2,:,:,:]*255)
axes[0].set_title('mask 0')
axes[1].set_title('mask 1')
axes[2].set_title('mask 2')
fig.suptitle('Masks displayed with matplotlib', fontsize=12)

# create random image and apply first mask
img = np.random.rand(256, 256, 3)
masked = deepcopy(img)
masked[mask==0] = 1
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(img)
axes[1].imshow(mask*255)
axes[2].imshow(masked)
axes[0].set_title('original')
axes[1].set_title('mask')
axes[2].set_title('masked')
fig.suptitle('Masked image displayed with matplotlib', fontsize=12)
plt.show()

# wait for user's click
cv2.waitKey(0)

BATCH = 4
val_datagen = MaskedDataGenerator(
	preprocessing_function=torch_preprocessing, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_generator = val_datagen.flow_from_directory(
    r"E:\Python\val_256",
    target_size=(256, 256),
    batch_size=BATCH
)

test_data = next(train_generator)
(masked, mask), ori = test_data
masked = torch_postprocessing(masked)
ori = torch_postprocessing(ori)

for i in range(BATCH):
	fig, axes = plt.subplots(1, 3, figsize=(10, 4))
	axes[0].imshow(masked[i])
	axes[1].imshow(mask[i]*255)
	axes[2].imshow(ori[i])
	axes[0].set_title('masked image')
	axes[1].set_title('mask')
	axes[2].set_title('original image')
	fig.suptitle('Input and expected output for network', fontsize=12)
	plt.show()
