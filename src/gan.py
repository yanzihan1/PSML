# -*- coding: utf-8 -*-
import tensorflow as tf
#tf.disable_v2_behavior()

import pickle
print("banben",tf.__version__)


class alignGAN():
	#########################################################################################
	# parameters init
	#########################################################################################
	def xavier_init(self, size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		aaa = tf.random_normal(shape=size, stddev=xavier_stddev, dtype=tf.float64)
		return tf.to_float(aaa)

	def init_paras_1hidden(self, sess):
		h1_dim = 64
		h2_dim = 1
		input_dim = tf.to_int32(self.input_dim).eval(session=sess)
		h1_dim = tf.to_int32(h1_dim).eval(session=sess)
		h2_dim = tf.to_int32(h2_dim).eval(session=sess)
		W1 = tf.Variable(self.xavier_init([input_dim, h1_dim]))
		W2 = tf.Variable(self.xavier_init([h1_dim, h2_dim]))
		b1 = tf.Variable(tf.zeros(shape=[h1_dim]))
		b2 = tf.Variable(tf.zeros(shape=[h2_dim]))
		return W1, W2, b1, b2

	def __init__(self, args, sess):
		self.input_dim = args.emb_dim
		self.train_batch_size = args.train_batch_size
		self.learning_rate = 1e-4
		self.lambda_c = 0.2
		if args.param_theta == 'None':
			self.param_theta = None
		else:
			self.param_theta = pickle.load(open(args.param_theta))
		if args.param_G == 'None':
			self.param_G = None
		else:
			self.param_G = pickle.load(open(args.param_G))

		self.build_model(sess)

	def variable_summaries(self, var, name):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope(name + 'summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar(name + 'mean', mean)
			with tf.name_scope(name + 'stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar(name + 'stddev', stddev)
			tf.summary.scalar(name + 'max', tf.reduce_max(var))
			tf.summary.scalar(name + 'min', tf.reduce_min(var))
			tf.summary.histogram(name + 'histogram', var)
	def writee(self,num_2_acc_top_10):
		with open('num_2_acc_top_10','w') as f:
			f.write(num_2_acc_top_10+'\n')

	def build_model(self, sess):
		#for train



		self.left_embedding_D = tf.placeholder(tf.float32, shape=(self.train_batch_size,self.input_dim))
		self.right_embedding_D = tf.placeholder(tf.float32, shape=(self.train_batch_size,self.input_dim))
		self.left_embedding_A = tf.placeholder(tf.float32, shape=(None,self.input_dim))
		self.right_embedding_A = tf.placeholder(tf.float32, shape=(None,self.input_dim))
		#for test
		self.node_left = tf.placeholder(tf.int32,name="node_left")
		self.node_right = tf.placeholder(tf.int32, name="node_right")
		self.batch_test_left = tf.placeholder(tf.float32,shape=(None,self.input_dim))
		self.right_test_embedding = tf.placeholder(tf.float32,shape=(None,self.input_dim))
		self.batch_test_right = tf.placeholder(tf.float32, shape=(None,self.input_dim))
		self.left_test_embedding = tf.placeholder(tf.float32, shape=(None,self.input_dim))
		self.neibor=tf.placeholder(dtype=tf.float32, shape=[None,128],name='neibor')
		self.ips0 = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='if_psu_1')#if pseudo 0?
		self.ips1 = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='if_psu_2')#if pseudo anchor1
		self.ifps = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='if_ps')


		with tf.variable_scope('gan'):
			if self.param_theta == None:
				W1,W2,b1,b2 = self.init_paras_1hidden(sess)
				self.theta = [W1,W2,b1,b2]
			else:
				self.theta = tf.Variable(self.param_theta,name='theta')

		self.params_theta = self.theta #discriminator参数

		with tf.variable_scope('sub'):
			if self.param_G == None:
				init_orthogonal = tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32) 
				self.G = tf.get_variable('G', shape=[self.input_dim, self.input_dim], initializer=init_orthogonal)
			else:
				self.G = tf.Variable(self.param_G[0], name="G")
		self.w0 = tf.get_variable('w_0', initializer=[[0.015] * 128] * 128)
		self.w1 = tf.get_variable('w_1', initializer=[[0.015] * 128] * 128)

		self.variable_summaries(self.G, 'G')
		self.params_G = [self.G] #G矩阵参数
		#########################################################################################
		# draw graph for tensorflow
		#########################################################################################

		#PSML
		self.new_add = tf.matmul(tf.multiply(self.neibor, self.ips0), self.w0) \
					   + tf.matmul(tf.multiply(self.neibor, self.ips1), self.w1)
		self.g_mapping = tf.transpose(tf.matmul(self.G, self.left_embedding_D, transpose_a=False,transpose_b=True))
		self.g_mapping= self.g_mapping+tf.multiply(tf.nn.relu(self.new_add),self.ifps)
		#print("shape",self.g_mapping) #60*100

		self.d_fake = self.discriminator(self.g_mapping,self.theta)
		self.d_real = self.discriminator(self.right_embedding_D,self.theta)

		#实现式（2）
		self.d_loss = tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake)
		#实现式（3）
		self.g_loss = -tf.reduce_mean(self.d_fake)
		#实现式（4）
		shape = tf.to_float(tf.shape(self.left_embedding_A))[0]
		self.a_mapping = tf.transpose(tf.matmul(self.G, self.left_embedding_A, transpose_a=False, transpose_b=True))
		Eu_distance_a = tf.sqrt(tf.reduce_sum(tf.square(self.a_mapping-self.right_embedding_A),1))
		self.a_loss = (self.lambda_c/shape)*(tf.reduce_sum(Eu_distance_a))

		tf.summary.scalar('d_loss', self.d_loss)
		tf.summary.scalar('g_loss', self.g_loss)
		self.d_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)\
			.minimize(-self.d_loss, var_list=self.params_theta)
		self.g_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) \
			.minimize(self.g_loss+self.a_loss, var_list=self.params_G)
		self.params_theta_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.params_theta]
		self.params_theta = self.params_theta_clip

		# for test

		self.l2r_acc,self.r2l_acc,self.node1_left,self.top30= self.get_acc()
		self.merged_summary_op = tf.summary.merge_all()
		self.test_summary_op = tf.summary.scalar('l2r_acc_top50', self.l2r_acc[4])
		# with open('num_2_acc_top_10', 'w') as f:
		# 	print type(num_2_acc_top_10)
		# 	f.write(num_2_acc_top_10 + '\n')
		# train_log.write(str(epoch_all) + "\t"
		# 				+ "train accuracy:\t" + buf + '\n')


	def get_acc(self):

		l2r_test_mapping = tf.transpose(tf.matmul(self.G, self.batch_test_left, transpose_a=False, transpose_b=True))
		self.test_all_score_l2r = tf.matmul(l2r_test_mapping,self.right_test_embedding,transpose_a=False,transpose_b=True)
		l2r_acc_intop_10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_l2r, self.node_right, 10), tf.float32))

		num_l2r_acc_top_10=tf.nn.top_k(self.test_all_score_l2r,10)
		l2r_acc_intop_20 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_l2r, self.node_right, 20), tf.float32))
		num_l2r_acc_top_20=tf.nn.top_k(self.test_all_score_l2r,20)
		l2r_acc_intop_30 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_l2r, self.node_right, 30), tf.float32))
		num_l2r_acc_top_30=tf.nn.top_k(self.test_all_score_l2r,30)
		l2r_acc_intop_40 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_l2r, self.node_right, 40), tf.float32))
		num_l2r_acc_top_40=tf.nn.top_k(self.test_all_score_l2r,40)
		#print(123123123123123)
		l2r_acc_intop_50 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_l2r, self.node_right, 50), tf.float32))
		num_l2r_acc_top_50=tf.nn.top_k(self.test_all_score_l2r,50)
		# f.append(a)
		# f.append(b)
		# f.append(ccc)
		# f.append(d)
		# f.append(e)

		#sess=tf.Session()
		#tf.initialize_all_variables().run()
		#sess.run(tf.initialize_all_variables())

		#print("node:right :",sess.run(self.node_right))
		#print("num_l2r_acc_top_10  :",sess.run(num_l2r_acc_top_10))
		#print("num_l2r_acc_top_20: ",sess.run(num_l2r_acc_top_20))
		#print("num_l2r_acc_top_30 :",sess.run(num_l2r_acc_top_30))
		#print("num_l2r_acc_top_40 :",sess.run(num_l2r_acc_top_40))
		#print("num_l2r_acc_top_50",sess.run(num_l2r_acc_top_50))

		l2r_acc = [l2r_acc_intop_10, l2r_acc_intop_20, l2r_acc_intop_30,l2r_acc_intop_40, l2r_acc_intop_50]

		r2l_test_mapping = tf.transpose(tf.matmul(self.G, self.left_test_embedding, transpose_a=False, transpose_b=True))
		#print("this is ",r2l_test_mapping)
		self.test_all_score_r2l = tf.matmul(self.batch_test_right,r2l_test_mapping, transpose_a=False,transpose_b=True)
		#print("321321321",self.test_all_score_r2l)
		r2l_acc_intop_10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 10), tf.float32))
		num_2_acc_top_10 = tf.nn.top_k(self.test_all_score_r2l, 10)
		r2l_acc_intop_20 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 20), tf.float32))
		num_2_acc_top_20 = tf.nn.top_k(self.test_all_score_r2l, 20)
		r2l_acc_intop_30 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 30), tf.float32))
		#top30=tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 30)
		#r2l_acc_intop_30 = tf.reduce_mean(tf.cast(top30, tf.float32))

		#num_2_acc_top_30 = tf.nn.top_k(self.test_all_score_r2l, 30)
		r2l_acc_intop_40 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 40), tf.float32))
		num_2_acc_top_40 = tf.nn.top_k(self.test_all_score_r2l, 40)
		r2l_acc_intop_50 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.test_all_score_r2l, self.node_left, 50), tf.float32))
		num_2_acc_top_50 = tf.nn.top_k(self.test_all_score_r2l, 50)
		r2l_acc = [r2l_acc_intop_10, r2l_acc_intop_20, r2l_acc_intop_30,r2l_acc_intop_40, r2l_acc_intop_50]
		pp=tf.convert_to_tensor([1,2,3])
		return l2r_acc,r2l_acc,self.node_left,pp

	def discriminator(self, batch_embedding,theta):
		W1 = theta[0]
		W2 = theta[1]
		b1 = theta[2]
		b2 = theta[3]
		D_h1 = tf.nn.relu(tf.matmul(batch_embedding, W1) + b1)
		out = tf.matmul(D_h1, W2) + b2
		return out

	def save_model_theta(self, sess, filename):
		param = sess.run(self.params_theta)
		pickle.dump(param, open(filename, 'wb'))

	def save_model_G(self, sess, filename):
		param = sess.run(self.params_G)
		pickle.dump(param, open(filename, 'wb'))

