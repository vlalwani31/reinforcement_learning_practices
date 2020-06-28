import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import Adam
from keras.models import load_model

#model.load("../savedparameters/BipedalWalkermodel.h5")
runner = load_model("../savedparameters/BipedalWalkermodel.h5")
runner.summary()
env = gym.make('BipedalWalker-v3')
done = False
observation = env.reset()
score = 0
while not done:
    env.render()
    action = runner.predict(np.array(observation).reshape(-1, len(observation)))[0]
    observation, reward, done, _ = env.step(action)
    score += reward
print('Final Score is: ', score)

env.close()
print(int(9/10))
