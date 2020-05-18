import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from libs.util import MaskGenerator, torch_preprocessing, torch_postprocessing
from libs.DataGenerator import MaskedDataGenerator
 # mask generation example

noTests = 100
generator = MaskGenerator(256, 256, 3)
for minCoverage in np.arange(0, 0.59, 0.15):
    maxCoverage = minCoverage + 0.15
    for test in range(noTests):
        generator._generate_mask(minCoverage, maxCoverage)
