import tensorflow as tf
import numpy as np
import random
from collections import deque
#-----------------------------------------------------------
# Fix seed for test
def seed(n):
	tf.set_random_seed(n)
	np.random.seed(n)
	random.seed(n)

#------------------------------------------------------------
'''
Replay memory wrapper. Auto length check
'''
class replay_buf:
	REPLAY_SIZE = 50000
	def __init__(self):
		self.buf = deque()

	def append(self,item):
		self.buf.append(item)
		if len(self.buf) > replay_buf.REPLAY_SIZE:
			self.buf.popleft()
#------------------------------------------------------------------
'''
s: session
input_size: determine x shape([None,input_size])
output_size: determine y shape([None,output_size])
name: this network's name. default 'main'
h_size: int. number of hidden units. default 10
learning_rate: learning rate. default 0.1
'''
class simple_dqn:
	def __init__(self,s,input_size,output_size,name='main',h_size=10,learning_rate=0.1):
		self.s = s
		self.input_size = input_size
		self.output_size = output_size
		self.name = name
		self.build(h_size,learning_rate)
		self.sync = None
		s.run(tf.global_variables_initializer())

	def build(self,h_size,learning_rate):
		with tf.variable_scope(self.name):
			self.x = tf.placeholder(tf.float32,[None,self.input_size])
			self.y = tf.placeholder(tf.float32,[None,self.output_size])

			w1 = tf.get_variable('w1',[self.input_size,h_size],initializer=tf.contrib.layers.xavier_initializer())
			l1 = tf.nn.relu(tf.matmul(self.x,w1))

			w2 = tf.get_variable('w2',[h_size,self.output_size],initializer=tf.contrib.layers.xavier_initializer())
			self.Qpred = tf.matmul(l1,w2)

			self.loss = tf.reduce_mean(tf.square(self.y - self.Qpred))
			self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	# predict
	def predict(self,x):
		return self.s.run(self.Qpred,feed_dict={self.x:x})

	# one hot
	def one_hot(self,s):
		return np.eye(self.input_size)[s:s+1]

	# reshape for CartPole
	def reshapeForCartPole(self,state):
		return np.reshape(state,(-1,self.input_size))

	# train
	# return (loss, train)
	def fit(self,x,y):
		return self.s.run([self.loss,self.train],feed_dict={self.x:x,self.y:y})

	# experience replay
	# it must called main network
	def experience_replay(self,target_dqn,buf,dis):
		batch = random.sample(buf.buf,10)

		x = np.empty(0).reshape(0,self.input_size)
		y = np.empty(0).reshape(0,self.output_size)

		for s,a,r,ns,d in batch:
			Q = self.predict(s)

			if d:
				Q[0][a] = r
			else:
				Q[0][a] = r + dis * np.max(target_dqn.predict(ns))

			x = np.vstack([x,s])
			y = np.vstack([y,Q])

		return self.fit(x,y)

	# sync networks
	def sync_networks(self,src_scope_name='main',dest_scope_name='target'):
		if self.sync == None:
			op_holder = []

			src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=src_scope_name)
			dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=dest_scope_name)

			for src_var,dest_var in zip(src_vars,dest_vars):
				op_holder.append(dest_var.assign(src_var.value()))

			self.sync = op_holder

		self.s.run(self.sync)

	# play use this network
	def bot_play(self,env,preprocess_func):
		s = env.reset()
		reward_sum = 0
		while True:
			env.render()
			s = preprocess_func(s)
			s,r,d,i = env.step(np.argmax(self.predict(s)))
			reward_sum += r
			if d:
				print 'Total score:',reward_sum
				break



