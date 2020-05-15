import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from libs.util import MaskGenerator
 # mask reduction in partial convolutions: example


generator = MaskGenerator(256, 256, 3)

# get mask
mask = generator.sample()

clusterSize = (2,2)

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for step in range(3):
    axes[step].imshow(mask*255)
    newmask = deepcopy(mask)
    for i in range(clusterSize[0], 256 - clusterSize[0]):
        for j in range(clusterSize[1], 256 - clusterSize[1]):
            for k in range(i-clusterSize[0], i+clusterSize[0]):
                for m in range(j-clusterSize[1], j+clusterSize[1]):
                    if mask[k,m][0] == 1:
                        newmask[i,j] = (1,1,1)
    mask = newmask

axes[3].imshow(mask*255)

plt.show()



# wait for user's click
cv2.waitKey(0)
