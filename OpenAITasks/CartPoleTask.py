import gym
import statistics
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import Adam


# 1. It renders instance for 1000 timesteps, perform random actions
env = gym.make('CartPole-v0')
#print(env.action_space)
#print(env.observation_space)
goal_steps = 400
score_requirements = 50
initial_games = 10000
env.reset()
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
    accepted_score = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        previous_observartion = []
        for _ in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action) # take a random action
            if len(previous_observartion) > 0:
                game_memory.append([previous_observartion, action])
            previous_observartion = observation
            score += reward
            if done:
                break
        if score >= score_requirements:
            accepted_score.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                else:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted_score: ', statistics.mean(accepted_score))
    print('Median accepted_score: ', statistics.median(accepted_score))
    return training_data

initial_population()
env.close()
