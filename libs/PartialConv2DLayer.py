from keras import backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers import Conv2D


class PartialConv2D(Conv2D):
  
    def __init__(self, *args, n_channels=3, last_layer=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_layer = last_layer
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """
        Adapted from original _Conv() layer of Keras.
        Parameters
            input_shape: list of dimensions for [img, mask].
        """
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last', "data format should be `channels_last`"
        channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        self.input_dim = input_shape[0][channel_axis]
        self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.input_dim
        # Image kernel:
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel  = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel:
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None
        # Image bias:
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True



    def call(self, inputs):
        #inputs is tuple containing (image, mask)
        assert isinstance(inputs, list) and len(inputs) == 2, f"PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: + {str(inputs)}"
        image_in, mask_in = inputs

        img_conv = K.conv2d(image_in * mask_in,
                              self.kernel,
                              strides=self.strides, 
                              padding=self.padding, 
                              data_format=self.data_format)

        #This is sum(M)
        self.update_mask = K.conv2d(mask_in, 
                         self.kernel_mask, 
                         strides=self.strides, 
                         padding=self.padding, 
                         data_format=self.data_format)

        self.mask_ratio = self.window_size / (self.update_mask + 1e-8)

        #updated mask (equals 1 where sum(M) > 0)
        self.update_mask = K.clip(self.update_mask, 0, 1)

        # Remove values where there are holes
        self.mask_ratio = self.mask_ratio*self.update_mask

        #img output is zero where sum(M) is 0

        if self.use_bias:
            img_conv = K.bias_add(
                img_conv,
                -self.bias,
                data_format=self.data_format)
            img_conv *= self.mask_ratio
            img_conv = K.bias_add(
                img_conv,
                self.bias,
                data_format=self.data_format)
            img_conv *= self.update_mask
        else:
            img_conv *= self.mask_ratio

        # Apply activations on the image
        if self.activation is not None:
            img_conv = self.activation(img_conv)
            
        if self.last_layer:
            return img_conv

        return [img_conv, self.update_mask ]



    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last'
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
        if self.last_layer:
            return new_shape
        return [new_shape, new_shape]