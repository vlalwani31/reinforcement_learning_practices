import gym
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, backend
import numpy as np

runner = models.load_model("../savedparameters/BiPedalWalker_DDPG/Best_actor_model.h5")
runner.summary()
env = gym.make('BipedalWalker-v3')
done = False
observation = env.reset()
score = 0
while not done:
    env.render()
    action = runner.predict((np.array(observation).reshape(1,24)))
    #print(action[0])
    observation, reward, done, _ = env.step(action[0])
    score += reward
print('Final Score is: ', score)
env.close()
