import readchar
import gym
import numpy as np

# MACROS
LEFT = 0
RIGHT = 1

# Key mapping
arrow_keys = {
	'\x1b[C':RIGHT,
	'\x1b[D':LEFT}


action = {0:'LEFT',1:'RIGHT'}


env = gym.make('CartPole-v0')
env.reset()

episodes = 10
for i in xrange(episodes):
	rewardSum = 0
	d = False
	env.reset()

	while not d:
		env.render()
		k = readchar.readkey()
		a = arrow_keys[k]
		print 'action:',action[a]
		s,r,d,_ = env.step(a)
		rewardSum += r
	print 'rewardSum:',rewardSum
