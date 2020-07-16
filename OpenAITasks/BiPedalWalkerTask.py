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
    regressor.add(Dense(units = 100, input_dim = input_size, activation='relu'))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units = 100, activation='relu'))
    regressor.add(Dropout(0.3))
    #regressor.add(Dense(units = 240, activation='relu'))
    #regressor.add(Dropout(0.25))
    #regressor.add(Dense(units = 256, activation='relu'))
    #regressor.add(Dropout(0.25))
    #regressor.add(Dense(units = 160, activation='relu'))
    #regressor.add(Dropout(0.25))
    #regressor.add(Dense(units = 128, activation='relu'))
    #regressor.add(Dropout(0.25))
    regressor.add(Dense(units = output_size, activation='tanh', kernel_regularizer='l2'))
    #regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse','accuracy'])
    regressor.compile(optimizer='adam', loss='mean_absolute_error',metrics=['mae'])
    return regressor


def WalkingTrial():
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = 2000
    #number_of_games = 1001
    number_of_games = 20
    epsilon_value = 1.0
    number_of_episodes = 20
    model = NNmodel(24, 4)
    for i in range(number_of_episodes):
        score = 0
        total_rewards = np.zeros(number_of_games)
        training_data = []
        all_games = []

        for j in range(number_of_games):
            done = False
            observation = env.reset()
            score = 0
            game_memory = []
            previous_observartion = []
            while not done:
                #env.render()
                action = env.action_space.sample() if (np.random.random() < epsilon_value) else (model.predict(np.array(observation).reshape(-1, len(observation)))[0])
                observation, reward, done, _ = env.step(action)
                if (len(previous_observartion) > 0):
                    game_memory.append([previous_observartion, action])
                previous_observartion = observation
                score += reward
            total_rewards[j] = score
            if (len(game_memory) != 0):
                all_games.append([game_memory, score])


        #percentile_score = np.percentile(total_rewards,(96 - i))
        for one_game in all_games:
            if (one_game[1] > 0):
                #print(one_game[1])
                training_data.append(one_game[0])
        all_games.clear()
        #print(len(training_data))# Total number of games
        #print(len(training_data[0]))# Number of steps in a game
        #print(len(training_data[0][0]))# Number of options in a step
        #print(len(training_data[0][0][0]))# Number of observations
        X = []
        Y = []
        for game_count,games in enumerate(training_data):
            #print('Episode: ', (i+1), ' Game Number: ', game_count)
            #X = np.array([i[0] for i in games]).reshape(-1, len(games[0][0]))
            #Y = np.array([i[1] for i in games])
            #model.fit(x= X,y= Y, epochs = 2)
            X = X + [i[0] for i in games]
            Y = Y + [i[1] for i in games]
        print('Episode: ', (i + 1))
        #X_for_training = np.array(X).reshape(-1,len(training_data[0][0][0]))
        if X:
            X_for_training = np.array(X).reshape(-1,len(training_data[0][0][0]))
            #print(X)
            Y_for_training = np.array(Y)
            if (i < 7):
                model.fit(x= X_for_training, y= Y_for_training, epochs= 10)
            else:
                model.fit(x= X_for_training, y= Y_for_training, epochs= 4)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = model.predict(np.array(observation).reshape(-1, len(observation)))[0]
            observation, reward, done, _ = env.step(action)
            score += reward
        print('Score is: ', score)
        epsilon_value = epsilon_value - (1.24/(number_of_episodes)) if (epsilon_value > 0.01) else 0.01

    model.save("./savedparameters/BipedalWalkermodel.h5")
    print('Saved model to Disk')
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = model.predict(np.array(observation).reshape(-1, len(observation)))[0]
        observation, reward, done, _ = env.step(action)
        score += reward
    print('Final Score is: ', score)

    env.close()


WalkingTrial()
