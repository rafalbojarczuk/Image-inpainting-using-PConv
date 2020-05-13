from .PartialConv2DLayer import PartialConv2D
from .Losses import total_loss_function
#from .util import PSNR
from keras.layers import BatchNormalization, Input
from keras.layers import ReLU, LeakyReLU, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam

def encoder_block(image_input, mask_input, filters, kernel_size, batch_norm=True, training_with_bn=True):
	img_conv, mask_conv = PartialConv2D(filters, 
		kernel_size, 
		strides=2, 
		padding='same',
		use_bias=True,
		kernel_initializer='he_uniform')([image_input, mask_input])
	if batch_norm:
		img_conv = BatchNormalization()(img_conv, training=training_with_bn)
	img_conv = ReLU()(img_conv)

	return img_conv, mask_conv

def decoder_block(image_input, mask_input, enc_image, enc_mask, filters, kernel_size, last_layer=False):
	img_upsampled = UpSampling2D(size=(2,2))(image_input)
	mask_upsampled = UpSampling2D(size=(2,2))(mask_input)
	concatenated_image = Concatenate(axis=3)([img_upsampled, enc_image])
	concatenated_mask = Concatenate(axis=3)([mask_upsampled, enc_mask])

	if last_layer:
		return PartialConv2D(filters,
			kernel_size,
			strides=1, 
			padding='same',
			use_bias=True, 
			kernel_initializer='he_uniform',
			last_layer=last_layer,
			)([concatenated_image, concatenated_mask])

	img_conv, mask_conv = PartialConv2D(filters,
			kernel_size,
			strides=1, 
			padding='same',
			use_bias=True, 
			kernel_initializer='he_uniform')([concatenated_image, concatenated_mask])

	img_conv = BatchNormalization()(img_conv)
	img_conv = LeakyReLU(alpha=0.2)(img_conv)
	return img_conv, mask_conv

def PartialConvUNet(fine_tuning=False, lr=0.0002, predict_only=False, image_shape=(256, 256), vgg16_weights='imagenet'):
	image_input = Input(shape=(image_shape[0], image_shape[1], 3))
	mask_input = Input(shape=(image_shape[0], image_shape[1], 3))

	#Encoder
	enc_img_1, enc_mask_1 = encoder_block(image_input, mask_input, 64, 7, batch_norm=False)
	enc_img_2, enc_mask_2 = encoder_block(enc_img_1, enc_mask_1, 128, 5, training_with_bn=(not fine_tuning))
	enc_img_3, enc_mask_3 = encoder_block(enc_img_2, enc_mask_2, 256, 5, training_with_bn=(not fine_tuning))
	enc_img_4, enc_mask_4 = encoder_block(enc_img_3, enc_mask_3, 512, 3, training_with_bn=(not fine_tuning))
	enc_img_5, enc_mask_5 = encoder_block(enc_img_4, enc_mask_4, 512, 3, training_with_bn=(not fine_tuning))
	enc_img_6, enc_mask_6 = encoder_block(enc_img_5, enc_mask_5, 512, 3, training_with_bn=(not fine_tuning))
	enc_img_7, enc_mask_7 = encoder_block(enc_img_6, enc_mask_6, 512, 3, training_with_bn=(not fine_tuning))
	enc_img_8, enc_mask_8 = encoder_block(enc_img_7, enc_mask_7, 512, 3, training_with_bn=(not fine_tuning))

	#Decoder
	dec_img_9, dec_mask_9 = decoder_block(enc_img_8, enc_mask_8, enc_img_7, enc_mask_7, 512, 3)
	dec_img_10, dec_mask_10 = decoder_block(dec_img_9, dec_mask_9, enc_img_6, enc_mask_6, 512, 3)
	dec_img_11, dec_mask_11 = decoder_block(dec_img_10, dec_mask_10, enc_img_5, enc_mask_5, 512, 3)
	dec_img_12, dec_mask_12 = decoder_block(dec_img_11, dec_mask_11, enc_img_4, enc_mask_4, 512, 3)
	dec_img_13, dec_mask_13 = decoder_block(dec_img_12, dec_mask_12, enc_img_3, enc_mask_3, 256, 3)
	dec_img_14, dec_mask_14 = decoder_block(dec_img_13, dec_mask_13, enc_img_2, enc_mask_2, 128, 3)
	dec_img_15, dec_mask_15 = decoder_block(dec_img_14, dec_mask_14, enc_img_1, enc_mask_1, 64, 3)
	dec_img_16 = decoder_block(dec_img_15, dec_mask_15, image_input, mask_input, 3, 3, last_layer=True)

	model = Model(inputs=[image_input, mask_input], outputs=dec_img_16)



	if not predict_only:
		model.compile(Adam(lr=lr), loss=total_loss_function(mask_input, vgg16_weights=vgg16_weights))

	return model