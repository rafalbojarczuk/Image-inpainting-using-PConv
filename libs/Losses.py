from keras import backend as K
from keras.applications.vgg16 import VGG16 
from keras.models import Model

def vgg16_features(layers, weights='imagenet'):
    """
    Feature exctraction VGG16 model.
  
    weights: ether "imagenet" or path to the file with weights. 

    # Returns
        features_model: keras.models.Model instance to extract the features.

    """
    assert isinstance(flayers,list), "First argument 'layers' must be a list"
    assert len(flayers) > 1, "Length of 'layers' must be > 1."
  
    base_model = VGG16(include_top=False, weights=weights)

    vgg16_outputs = [base_model.get_layer(layer).output for layer in layers]

    features_model = Model(inputs=[base_model.input], outputs=vgg16_outputs, name='vgg16_features')
    features_model.trainable = False
    features_model.compile(loss='mse', optimizer='adam')
    
    return features_model

def gram_matrix(x):
    """Calculate gram matrix used in style loss"""
    
    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
    
    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[3], shape[1], shape[2]
    
    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    gram = K.batch_dot(features, features, axes=2)
    
    # Normalize with channels, height and width
    gram /= K.cast(C * H * W, x.dtype)
    
    return gram

def total_loss_function(mask, vgg16_weights='imagenet'):
	"""
	I_gt - ground truth image - y_true
	I_out - predicted image - y_pred
	I_comp - non-masked part is ground truth and masked region is taken from predicted image
	"""

	vgg16_layers = ['block1_pool', 'block2_pool', 'block3_pool']
	vgg = vgg16_feature_model(vgg16_layers, weights=vgg16_weights)
	def loss(y_true, y_pred):
		I_comp = mask*y_true + (1-mask) * y_pred
		vgg_gt   = vgg_model(y_true)
        vgg_out  = vgg_model(y_pred)
        vgg_comp = vgg_model(y_comp)

		l_valid = valid_loss(y_true, y_pred, mask)
		l_hole  = hole_loss(y_true, y_pred, mask)
		l_perceptual  = perceptual_loss(vgg_out, vgg_gt, vgg_comp)
		l_style_out = style_loss(vgg_out, vgg_gt) 
		l_style_comp = style_loss(vgg_comp, vgg_gt)
		l_tv = total_variance_loss(mask, y_comp)

		return l_valid + 6.*l_hole + 0.05*l_perceptual + 120.*(l_style_out + l_style_comp) + 0.1*l_tv		

def l1_loss(y_true, y_pred):
	if K.ndim(y_true) == 4:
	    return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
	elif K.ndim(y_true) == 3:
	    return K.mean(K.abs(y_pred - y_true), axis=[1,2])
	else:
	    raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

def valid_loss(y_true, y_pred, mask):
	return l1_loss(mask*y_true, mask*y_pred)

def hole_loss(y_true, y_pred, mask):
	return l1_loss((1-mask)*y_true, (1-mask)*y_pred)

def perceptual_loss(vgg_out, vgg_gt, vgg_comp):
	loss = 0
	for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
	    loss += self.l1(o, g) + self.l1(c, g)
	return loss

def style_loss(output, vgg_gt):
    """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
    loss = 0
    for o, g in zip(output, vgg_gt):
        loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
    return loss


def total_variance_loss(mask, y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""

    # Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
    dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

    # Cast values to be [0., 1.], and compute dilated hole region of y_comp
    dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
    P = dilated_mask * y_comp

    # Calculate total variation loss     
    return l1_loss(P[:,1:,:,:], P[:,:-1,:,:]) + l1_loss(P[:,:,1:,:], P[:,:,:-1,:])