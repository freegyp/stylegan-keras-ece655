import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

def dis_block(_input, filters, pool=True):
	x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(_input)
	x = LeakyReLU(0.01)(x)

	if pool:
		x = AveragePooling2D()(x)

	x = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
	x = LeakyReLU(0.01)(x)

	return x

def _discriminator(img_size):
	inp = Input(shape = [img_size,img_size,3])

	x = dis_block(inp, 16)
	x = dis_block(x,32)
	x = dis_block(x,64)

	if img_size>32:
		x = dis_block(x,128)
	if img_size>64:
		x = dis_block(x,192)
	if img_size>128:
		x = dis_block(x,256)
	if img_size>256:
		x = dis_block(x,384)
	if img_size>512:
		x = dis_block(x,512)

	x = Flatten()(x)
	x = Dense(128,kernel_initializer = 'he_normal',bias_initializer = 'zeros')(x)
	x = LeakyReLU(0.01)(x)

	x = Dropout(0.01)(x)
	x = Dense(1,kernel_initializer = 'he_normal',bias_initializer = 'zeros')(x)

	return Model(inputs = inp, outputs = x)


