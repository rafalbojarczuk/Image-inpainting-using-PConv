import cv2
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from libs.util import MaskGenerator
 # mask generation example

if __name__ == '__main__':
    generator = MaskGenerator(512, 512, 3)

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
    img = np.random.rand(512, 512, 3)
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
