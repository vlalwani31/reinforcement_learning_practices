import gym
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, backend
import numpy as np

runner = models.load_model("../savedparameters/Pendulum_DDPG/actor_model.h5")
runner.summary()
env = gym.make('Pendulum-v0')
done = False
observation = env.reset()
score = 0
while not done:
    env.render()
    action = runner.predict(np.array(observation).reshape(1,3))
    observation, reward, done, _ = env.step(action)
    score += reward
print('Final Score is: ', score)
env.close()
