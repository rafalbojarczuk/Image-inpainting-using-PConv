import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from libs.util import MaskGenerator
from copy import deepcopy

class MaskedDataGenerator(ImageDataGenerator):

	def flow_from_directory(self, directory, mask_generator=MaskGenerator(256,256),  *args, **kwargs):

		generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        
		seed = None if 'seed' not in kwargs else kwargs['seed']

		while True:
            
			original_image = next(generator)

			mask = np.stack([mask_generator.sample(seed) for _ in range(original_image.shape[0])], axis=0)
			masked_image = deepcopy(original_image)
			masked_image *= mask

         
			yield [masked_image, mask], original_image