import gym
import tensorflow as tf
import numpy as np
import rl

env = gym.make('CartPole-v0').env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = .99
episodes = 300

with tf.Session() as s:
	buf = rl.replay_buf()
	main_dqn = rl.simple_dqn(s,input_size,output_size,name='main')
	target_dqn = rl.simple_dqn(s,input_size,output_size,name='target')

	s.run(tf.global_variables_initializer())

	main_dqn.sync_networks()

	for i in xrange(episodes):
		done = False
		state = main_dqn.reshapeForCartPole(env.reset())
		steps = 0

		while not done:
			action = np.argmax(main_dqn.predict(state) + np.random.randn(1,env.action_space.n) / (i+1))

			new_state,reward,done,_ = env.step(action)
			new_state = main_dqn.reshapeForCartPole(new_state)

			if done: reward = -100
			buf.append((state,action,reward,new_state,done))

			state = new_state
			steps += 1
			if steps >= 10000: break

		print 'Episode:',i,'Steps:',steps
		if i%10==0 and i!=0:
			for _ in xrange(50):
				loss,_ = main_dqn.experience_replay(target_dqn,buf,dis)	
			print 'Loss:',loss
			main_dqn.sync_networks()

	main_dqn.bot_play(env,main_dqn.reshapeForCartPole)
