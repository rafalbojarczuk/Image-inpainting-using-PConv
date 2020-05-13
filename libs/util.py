import os
from random import randint, seed
import itertools
import numpy as np
import cv2

def torch_preprocessing(x):
    """
    Image pre-processing function used in PyTorch.

    For an RGB image:
       1) 1/255 scaling,
       2) per-channel mean subtraction (0.485, 0.456, 0.406),
       3) per-channel scaling by std (0.229, 0.224, 0.225).

    Parameters
        x: numpy array with an image or a batch of images.
    """
    x /= 255.
    x[..., 0] -= 0.485
    x[..., 1] -= 0.456
    x[..., 2] -= 0.406
    x[..., 0] /= 0.229
    x[..., 1] /= 0.224
    x[..., 2] /= 0.225
    return x


def torch_postprocessing(x, return_int=True):
    """
    Post-processing function to invert `torch_preprocessing` and clip values autside [0..255].

    Parameters
        x: numpy array with an image or a batch of images.
        return_int: boolean, whether to cast the image to `uint8`, returns `float` if False.

    Comments
        The color scaling is done in-place.
    """

    x[..., 0] *= 0.229
    x[..., 1] *= 0.224
    x[..., 2] *= 0.225
    x[..., 0] += 0.485
    x[..., 1] += 0.456
    x[..., 2] += 0.406
    x *= 255.
    if return_int:
        return x.clip(0, 255).astype('uint8')
    else:
        return x.clip(0, 255)

#def PSNR(y_true, y_pred):
#    return -10.0*K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)



class MaskGenerator():

    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)

        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)

        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

        return 1-img

    # def _load_mask(self, rotation=True, dilation=True, cropping=True):
    #     """Loads a mask from disk, and optionally augments it"""
    #
    #     # Read image
    #     mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))
    #
    #     # Random rotation
    #     if rotation:
    #         rand = np.random.randint(-180, 180)
    #         M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
    #         mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    #
    #     # Random dilation
    #     if dilation:
    #         rand = np.random.randint(5, 47)
    #         kernel = np.ones((rand, rand), np.uint8)
    #         mask = cv2.erode(mask, kernel, iterations=1)
    #
    #     # Random cropping
    #     if cropping:
    #         x = np.random.randint(0, mask.shape[1] - self.width)
    #         y = np.random.randint(0, mask.shape[0] - self.height)
    #         mask = mask[y:y+self.height, x:x+self.width]
    #
    #     return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        # if self.filepath and len(self.mask_files) > 0:
        #     return self._load_mask()
        # else:
        #     return self._generate_mask()
        return self._generate_mask()


