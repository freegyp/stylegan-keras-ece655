import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation, Layer
from tensorflow.keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

class AdaInstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]



def gen_block(_input, style, noise, filters, up = True):
	s = Dense(filters, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
	s = Reshape([1,1,filters])(s)
	b = Dense(filters, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
	b = Reshape([1,1,filters])(b)

	n = Conv2D(filters = filters, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)

	if up:
		out = UpSampling2D(interpolation = 'bilinear')(_input)
		out = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
	else:
		out = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(_input)

	out = add([out,n])

	out = AdaInstanceNormalization()([out,s,b])

	out = LeakyReLU(0.01)(out)

	s = Dense(filters, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
	s = Reshape([1,1,filters])(s)
	b = Dense(filters, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
	b = Reshape([1,1,filters])(b)

	n = Conv2D(filters = filters, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)

	out = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)

	out = add([out,n])

	out = AdaInstanceNormalization()([out,s,b])

	out = LeakyReLU(0.01)(out)

	return out

def _generator(img_size):
	latents = []
	_size = img_size
	while _size>=4:
		latents.append(Input(shape = [512]))
		_size = _size//2

	style_layers = len(latents)

	input_noise = Input(shape = [img_size,img_size,1])
	noise = [Activation('linear')(input_noise)]
	cur_size = img_size
	while cur_size>4:
		cur_size = cur_size//2
		noise.append(Cropping2D(int(cur_size/2))(noise[-1]))

	inp = Input(shape = [1])
	x = Dense(4*4*img_size,kernel_initializer = 'ones', bias_initializer = 'zeros')(inp)
	x = Reshape([4,4,img_size])(x)
	x = gen_block(x,latents[0],noise[-1],img_size,up=False)

	if img_size>=1024:
		x = gen_block(x,latents[-8],noise[7],512)
	if img_size>=512:
		x = gen_block(x,latents[-7],noise[6],384)
	if img_size>=256:
		x = gen_block(x,latents[-6],noise[5],256)
	if img_size>=128:
		x = gen_block(x,latents[-5],noise[4],192)
	if img_size>=64:
		x = gen_block(x,latents[-4],noise[3],128)

	x = gen_block(x,latents[-3],noise[2],64)
	x = gen_block(x,latents[-2],noise[1],32)
	x = gen_block(x,latents[-1],noise[0],16)

	x = Conv2D(filters=3, kernel_size=1, padding='same', activation='sigmoid', bias_initializer = 'zeros')(x)

	return style_layers,Model(inputs = latents+[input_noise,inp],outputs = x)

def _style_mapping(latent_size=512,layers = 6):
	inp = Input(shape = [latent_size])

	x = Dense(512,kernel_initializer = 'he_normal',bias_initializer = 'zeros')(inp)

	for _ in range(layers-1):
		x = LeakyReLU(0.01)(x)
		x = Dense(512,kernel_initializer = 'he_normal',bias_initializer = 'zeros')(x)

	return Model(inputs = inp,outputs = x)



