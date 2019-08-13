import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from tensorflow.keras.models import Model,model_from_json
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import tensorflow_datasets as tfds
import tensorflow_gan as tfgan

from stylegan.generator import _generator,_style_mapping,AdaInstanceNormalization
from stylegan.discriminator import _discriminator
from stylegan.adamlr import Adam_lr_mult

from functools import partial

import numpy as np
import json
from PIL import Image

#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight, sample_weight=None):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)

def safeDownload(addr):
	_comps = addr.split('/')
	comps = []
	for k in _comps:
		if len(k)>0:
			comps.append(k)
	if comps[0]=="gs:":
		from google.cloud import storage
		client = storage.Client(project="deep-learning-capstone-course")
		bucket = client.get_bucket(comps[1])
		blob = storage.Blob("/".join(comps[2:]),bucket)
		with open("img_bucket/"+"/".join(comps[2:]),'w') as file:
			client.download_blob_to_file(blob, file)

def safeUpload(addr):
	_comps = addr.split('/')
	comps = []
	for k in _comps:
		if len(k)>0:
			comps.append(k)
	if comps[0]=="gs:":
		from google.cloud import storage
		client = storage.Client(project="deep-learning-capstone-course")
		bucket = client.get_bucket(comps[1])
		blob = storage.Blob("/".join(comps[2:]),bucket)
		blob.upload_from_filename("img_bucket/"+"/".join(comps[2:]),client=client)


class GAN(object):
	def __init__(self, img_size=1024, batch_size = 8, latent_size = 512, latent_layers = 4, steps = 0, lr = 0.0001, decay = 0.00001, preTrained = False, bucket = "image-bucket"):
		temp = (1-decay)**steps
		self.lr = lr*temp
		self.steps = steps
		self.img_size = img_size
		self.latent_size = latent_size
		self.latent_layers = latent_layers
		self.batch_size = batch_size
		self.bucket = bucket

		ss = img_size
		self.style_layers = 0
		while ss>2:
			ss //= 2
			self.style_layers += 1

		self._gen,self._styler,self._dis = None,None,None
		self.single_gen,self.single_sty,self.single_dis = None,None,None
		self.mix_gen,self.mix_sty,self.mix_dis = None,None,None

		if not preTrained:
			self.singleGen()
			self.singleStyle()
			self.singleDis()
			self.mixGen()
			self.mixStyle()
			self.mixDis()
		else:
			self.loadAll()

	def gen(self):
		if self._gen==None:
			self.style_layers,self._gen = _generator(self.img_size)
			print("Generator:")
			self._gen.summary()
		return self._gen

	def styler(self):
		if self._styler==None:
			self._styler = _style_mapping(latent_size = self.latent_size, layers = self.latent_layers)
			print("Styler:")
			self._styler.summary()
		return self._styler

	def dis(self):
		if self._dis==None:
			self._dis = _discriminator(self.img_size)
			print("Discriminator:")
			self._dis.summary()
		return self._dis

	def singleGen(self):
		if self.single_gen==None:
			self.dis().trainable = False
			for layer in self.dis().layers:
				layer.trainable = False

			self.gen().trainable = True
			for layer in self.gen().layers:
				layer.trainable = True

			self.styler().trainable = False
			for layer in self.styler().layers:
				layer.trainable = False

			latent_inp = Input(shape = [self.latent_size])
			latents = self.styler()(latent_inp)
			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()([latents]*self.style_layers + [const_inp,const_1_inp])
			dis_out = self.dis()(gen_out)

			#opt = Adam_lr_mult(lr = self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = {"model_2":0.1})
			#print(opt.__dict__.keys())

			self.single_gen = Model(inputs = [latent_inp,const_inp,const_1_inp], outputs = dis_out)
			self.single_gen.compile(optimizer = Adam(self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')
		return self.single_gen

	def singleStyle(self):
		if self.single_sty==None:
			self.dis().trainable = False
			for layer in self.dis().layers:
				layer.trainable = False

			self.gen().trainable = False
			for layer in self.gen().layers:
				layer.trainable = False

			self.styler().trainable = True
			for layer in self.styler().layers:
				layer.trainable = True

			latent_inp = Input(shape = [self.latent_size])
			latents = self.styler()(latent_inp)
			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()([latents]*self.style_layers + [const_inp,const_1_inp])
			dis_out = self.dis()(gen_out)

			#opt = Adam_lr_mult(lr = self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = {"model_2":0.1})
			#print(opt.__dict__.keys())

			self.single_sty = Model(inputs = [latent_inp,const_inp,const_1_inp], outputs = dis_out)
			self.single_sty.compile(optimizer = Adam(self.lr*0.01, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')
		return self.single_sty

	def singleDis(self):
		if self.single_dis==None:
			self.dis().trainable = True
			for layer in self.dis().layers:
				layer.trainable = True

			self.gen().trainable = False
			for layer in self.gen().layers:
				layer.trainable = False

			self.styler().trainable = False
			for layer in self.styler().layers:
				layer.trainable = False

			real_inp = Input(shape = [self.img_size, self.img_size, 3])
			dout_real = self.dis()(real_inp)

			latent_inp = Input(shape = [self.latent_size])
			latents = self.styler()(latent_inp)
			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()([latents]*self.style_layers + [const_inp,const_1_inp])
			dout_fake = self.dis()(gen_out)

			partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = real_inp, weight = 50)

			self.single_dis = Model(inputs = [real_inp,latent_inp,const_inp,const_1_inp], outputs = [dout_real,dout_fake,dout_real])
			self.single_dis.compile(optimizer = Adam(self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = ['mse','mse',partial_gp_loss])
		return self.single_dis

	def mixGen(self):
		if self.mix_gen==None:
			self.dis().trainable = False
			for layer in self.dis().layers:
				layer.trainable = False

			self.gen().trainable = True
			for layer in self.gen().layers:
				layer.trainable = True

			self.styler().trainable = False
			for layer in self.styler().layers:
				layer.trainable = False

			latents_inp = []
			latents = []
			for _ in range(self.style_layers):
				latents_inp.append(Input(shape = [self.latent_size]))
				latents.append(self.styler()(latents_inp[-1]))

			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()(latents + [const_inp,const_1_inp])
			dis_out = self.dis()(gen_out)

			#opt = Adam_lr_mult(lr = self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = {"model_2":0.1})
			#print(opt.__dict__.keys())

			self.mix_gen = Model(inputs = latents_inp + [const_inp,const_1_inp],outputs = dis_out)
			self.mix_gen.compile(optimizer = Adam(self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')
		return self.mix_gen

	def mixStyle(self):
		if self.mix_sty==None:
			self.dis().trainable = False
			for layer in self.dis().layers:
				layer.trainable = False

			self.gen().trainable = False
			for layer in self.gen().layers:
				layer.trainable = False

			self.styler().trainable = True
			for layer in self.styler().layers:
				layer.trainable = True

			latents_inp = []
			latents = []
			for _ in range(self.style_layers):
				latents_inp.append(Input(shape = [self.latent_size]))
				latents.append(self.styler()(latents_inp[-1]))

			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()(latents + [const_inp,const_1_inp])
			dis_out = self.dis()(gen_out)

			#opt = Adam_lr_mult(lr = self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001, multipliers = {"model_2":0.1})
			#print(opt.__dict__.keys())

			self.mix_sty = Model(inputs = latents_inp + [const_inp,const_1_inp],outputs = dis_out)
			self.mix_sty.compile(optimizer = Adam(self.lr*0.01, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')
		return self.mix_sty

	def mixDis(self):
		if self.mix_dis==None:
			self.dis().trainable = True
			for layer in self.dis().layers:
				layer.trainable = True

			self.gen().trainable = False
			for layer in self.gen().layers:
				layer.trainable = False

			self.styler().trainable = False
			for layer in self.styler().layers:
				layer.trainable = False

			real_inp = Input(shape = [self.img_size,self.img_size,3])
			dout_real = self.dis()(real_inp)

			latents_inp = []
			latents = []
			for _ in range(self.style_layers):
				latents_inp.append(Input(shape = [self.latent_size]))
				latents.append(self.styler()(latents_inp[-1]))

			const_inp = Input(shape = [self.img_size,self.img_size,1])
			const_1_inp = Input(shape = [1])

			gen_out = self.gen()(latents + [const_inp,const_1_inp])
			dout_fake = self.dis()(gen_out)

			partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = real_inp, weight = 50)

			self.mix_dis = Model(inputs = [real_inp] + latents_inp + [const_inp,const_1_inp],outputs = [dout_real,dout_fake,dout_real])
			self.mix_dis.compile(optimizer = Adam(self.lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = ['mse','mse',partial_gp_loss])
		return self.mix_dis

	def _saveModel(self,model,name):
		_bucket = self.bucket if self.bucket[:3]!="gs:" else "img_bucket"
		_json = model.to_json()
		f = open("{0}/stylegan_model/{1}.json".format(_bucket,name),'w')
		f.write(_json)
		f.close()
		model.save_weights("{0}/stylegan_model/{1}_{2}.h5".format(_bucket,name,self.steps))
		safeUpload("{0}/stylegan_model/{1}.json".format(self.bucket,name))
		safeUpload("{0}/stylegan_model/{1}_{2}.h5".format(self.bucket,name,self.steps))

	def _loadModel(self,name,steps):
		_bucket = self.bucket if self.bucket[:3]!="gs:" else "img_bucket"
		safeDownload("{0}/stylegan_model/{1}.json".format(self.bucket,name))
		safeDownload("{0}/stylegan_model/{1}_{2}.h5".format(self.bucket,name,steps))

		f = open("{0}/stylegan_model/{1}.json".format(_bucket,name),'r')
		_json = f.read()
		f.close()

		mod = model_from_json(str(_json), custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
		mod.load_weights("{0}/stylegan_model/{1}_{2}.h5".format(_bucket,name,steps))

		return mod

	def saveAll(self):
		print("Saving model for step {0}.".format(self.steps))
		self._saveModel(self.gen(),"gen")
		self._saveModel(self.styler(),"styler")
		self._saveModel(self.dis(),"dis")

	def loadAll(self):
		self._gen = self._loadModel("gen",self.steps)
		self._styler = self._loadModel("styler",self.steps)
		self._dis = self._loadModel("dis",self.steps)

	def saveGenerated(self,img):
		_bucket = self.bucket if self.bucket[:3]!="gs:" else "img_bucket"
		img = Image.fromarray(np.uint8(img*255),mode = 'RGB')
		img.save("{0}/generated_images/{1}.jpg".format(_bucket,self.steps))
		safeUpload("{0}/generated_images/{1}.jpg".format(self.bucket,self.steps))

def train_forever(gan, ds, show_fn=None):
	#tb = TensorBoard(log_dir="{0}/tf_logs".format(gan.bucket),batch_size=8)
	ones = np.ones((8,1),dtype=np.float32)
	onesEval = np.ones((gan.style_layers*3,1),dtype=np.float32)
	zeros = np.zeros((8,1),dtype=np.float32)
	zerosEval = np.zeros((gan.style_layers*3,1),dtype=np.float32)
	nones = -ones
	nonesEval = -onesEval
	g_losses,d_losses = [],[]

	while True:
		if gan.steps%10<5:
			g1 = gan.singleGen().train_on_batch([np.random.normal(size = [8,gan.latent_size]),np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones],ones)
			g2 = gan.singleStyle().train_on_batch([np.random.normal(size = [8,gan.latent_size]),np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones],ones)
			dis_data = [next(iter(ds)),np.random.normal(size = [8,gan.latent_size]),np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones]
			d = gan.singleDis().train_on_batch(dis_data,[ones,nones,ones])
		else:
			n1,n2 = [],[]
			threshold1 = np.int32(np.random.uniform(0.0, gan.style_layers, size = [8]))
			threshold2 = np.int32(np.random.uniform(0.0, gan.style_layers, size = [8]))
			_n1 = np.random.normal(size = [8,gan.latent_size])
			_n2 = np.random.normal(size = [8,gan.latent_size])
			_n3 = np.random.normal(size = [8,gan.latent_size])
			_n4 = np.random.normal(size = [8,gan.latent_size])
			for i in range(gan.style_layers):
				n1back,n2back = [],[]
				for j in range(8):
					if i<threshold1[j]:
						n1back.append(_n1[j])
					else:
						n1back.append(_n2[j])
					if i<threshold2[j]:
						n2back.append(_n3[j])
					else:
						n2back.append(_n4[j])
				n1back = np.array(n1back)
				n2back = np.array(n2back)
				n1.append(n1back)
				n2.append(n2back)
			g1 = gan.mixGen().train_on_batch(n1+[np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones],ones)
			g2 = gan.mixStyle().train_on_batch(n1+[np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones],ones)
			d = gan.mixDis().train_on_batch([next(iter(ds))]+n2+[np.random.uniform(size = [8,gan.img_size,gan.img_size,1]),ones],[ones,nones,ones])
		#tb.on_epoch_end(gan.steps,{"gen_loss":g,"dis_loss0":d[0],"dis_loss1":d[1],"dis_loss2":d[2]})
		if gan.steps%11==0:
			print("At step {0}, generator loss: {1}, discriminator loss: {2}.".format(gan.steps,[g1,g2],d))
		d_losses.append(d)
		g_losses.append([g1,g2])
		gan.steps += 1
		if gan.steps%1000==0:
			_n1,_n2 = np.random.normal(size = [gan.style_layers,gan.latent_size]),np.random.normal(size = [gan.style_layers,gan.latent_size])
			latents = []
			for i in range(gan.style_layers):
				nn = []
				for j in range(gan.style_layers):
					nn.append(_n1[j])
				for j in range(gan.style_layers):
					nn.append(_n2[j])
				for j in range(gan.style_layers):
					if j<=i:
						nn.append(_n1[j])
					else:
						nn.append(_n2[j])
				nn = np.array(nn)
				latents.append(gan.styler().predict(nn))
			images = gan.gen().predict(latents+[np.random.uniform(size = [gan.style_layers*3,gan.img_size,gan.img_size,1]),onesEval])
			image_grid = tfgan.eval.python_image_grid(images, grid_shape=(3,gan.style_layers))
			if show_fn!=None:
				show_fn(image_grid)
			gan.saveGenerated(image_grid)
			gan.saveAll()
			_bucket_addr = gan.bucket
			if gan.bucket[:3]=="gs:":
				_bucket_addr = "img_bucket"
			rf = open("{0}/records/{1}_g.txt".format(_bucket_addr,gan.steps),'w')
			for dd in d_losses:
				rf.write("{0}\n".format(dd))
			rf.close()
			safeUpload("{0}/records/{1}_g.txt".format(gan.bucket,gan.steps))
			rf2 = open("{0}/records/{1}_d.txt".format(_bucket_addr,gan.steps),'w')
			for gg in g_losses:
				rf2.write("{0} ".format(gg))
				rf2.write("\n")
			rf2.close()
			d_losses,g_losses = [],[]
			safeUpload("{0}/records/{1}_d.txt".format(gan.bucket,gan.steps))

def _predict(gan,show_fn=None):
	onesEval = np.ones((1,1),dtype=np.float32)
	n1 = np.random.normal(size = [1,gan.latent_size])
	latents = []
	for i in range(gan.style_layers):
		latents.append(gan.styler().predict(n1))
	print(latents[0])
	images = gan.gen().predict(latents+[np.random.uniform(size = [1,gan.img_size,gan.img_size,1]),onesEval])
	image_grid = tfgan.eval.python_image_grid(images, grid_shape=(1,1))
	if show_fn!=None:
		show_fn(image_grid)
	#gan.saveGenerated(image_grid)

def _preprocess(element):
	images = (tf.cast(element['image'], tf.float32)) / 255.0
	return images

def main():
	gan = GAN(img_size=256,lr=0.0001)#, steps=9000,preTrained=True)
	dataset_dir = "{0}/datasets".format("gs://face-images-ece655")
	ds = tfds.load('celeb_a_hq/256',split = 'train', data_dir=dataset_dir).map(_preprocess, num_parallel_calls=4).shuffle(
        buffer_size=1000).batch(8)
	ds = tfds.as_numpy(ds)
	train_forever(gan, ds)















