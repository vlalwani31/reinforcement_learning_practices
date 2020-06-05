import gym
import random
import statistics
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import Adam
from keras.models import model_from_json



def NNmodel(input_size, output_size):
    regressor = Sequential()
    regressor.add(Dense(units = 128, input_dim = input_size, activation='relu'))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units = 192, activation='relu'))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units = 256, activation='relu'))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units = 192, activation='relu'))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units = 128, activation='relu'))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units = output_size, activation='tanh'))
    regressor.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['mae','accuracy'])
    return regressor


def WalkingTrial():
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = 2000
    number_of_games = 201
    epsilon_value = 1.0
    number_of_epochs = 10
    model = NNmodel(24, 4)
    for j in range(number_of_epochs):
        score = 0
        total_rewards = np.zeros(number_of_games)

        for i in range(number_of_games):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                #env.render()
                action = env.action_space.sample() if (np.random.random() < epsilon_value) else model.predict(np.array(observation).reshape(-1, len(observation)))
                observation, reward, done, _ = env.step()
                score += reward
            total_rewards[i] = score

        median_score = statistics.median(total_rewards)
        epsilon_value = epsilon_value - (2/number_of_games) if (epsilon_value > 0.01) else 0.01
        if (((i+1) % 4) == 0):
            print('Episode: ', (i+1), ' Maximum Value: ', max(total_rewards))

    print(total_rewards)
    env.close()


WalkingTrial()
