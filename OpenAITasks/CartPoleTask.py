import gym
import random
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
    np.save('./savedparameters/saved.npy', training_data_save)
    print('Average accepted_score: ', statistics.mean(accepted_score))
    print('Median accepted_score: ', statistics.median(accepted_score))
    return training_data

def NNmodel(input_size):
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
    regressor.add(Dense(units = 2, activation='softmax'))
    regressor.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['mae','accuracy'])
    return regressor

def train_NNmodel(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    Y = np.array([i[1] for i in training_data])
    print(X.shape)
    print(Y.shape)
    if not model:
        model = NNmodel(len(X[0]))

    model.fit(x= X,y= Y, epochs = 5, steps_per_epoch = 500)
    return model

training_data = initial_population()
model = train_NNmodel(training_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    previous_observartion = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if(len(previous_observartion) == 0):
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.array(previous_observartion).reshape(-1, len(previous_observartion))))
        choices.append(action)
        new_observation, reward, done, _ = env.step(action)
        previous_observartion = new_observation
        #game_memory.append([new_observation, action)
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

env.close()
