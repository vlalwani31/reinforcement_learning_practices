import gym
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import Adam


# 1. It renders instance for 1000 timesteps, perform random actions
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
goal_steps = 400
score_requirements = 57
initial_games = 10000
#for num_iterations in range(20):
#    observation = env.reset()
#    for t in range(100):
#        env.render()
#        print(observation)
#        observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
def initial_population():
    training_data = []
    scores = []
    
env.close()
