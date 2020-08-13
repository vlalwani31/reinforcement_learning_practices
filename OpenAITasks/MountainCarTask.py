import gym
import random
import statistics
import numpy as np
import pickle

pos_space = np.linspace(-1.2,0.6,20)
vel_space = np.linspace(-0.07,0.07,20)

def get_state(observation):
    pos,vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0,1,2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return action

def mountain_trial():
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    number_of_games = 32000
    alpha = 0.1
    gamma = 0.99
    epsilon_value = 1.0
    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos,vel))

    Q = {}
    for state in states:
        for action in [0,1,2]:
            Q[state, action] = 0

    score = 0
    total_rewards = np.zeros(number_of_games)
    for i in range(number_of_games):
        done = False
        observation = env.reset()
        state = get_state(observation)
        if ((i % 1000 == 0) and (i > 0)):
            print('Episode ', i, ' score ', score, ' epsilon %.3f' % epsilon_value)
        score = 0
        while not done:
            action = np.random.choice([0,1,2]) if (np.random.random() < epsilon_value) else max_action(Q, state)
            obs_, reward, done, _ = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = max_action(Q, state_)
            Q[state, action] = Q[state, action] + (alpha*(reward + (gamma*Q[state_, action_]) - Q[state, action]))
            state = state_
        total_rewards[i] = score
        epsilon_value = epsilon_value - (1.4/number_of_games) if (epsilon_value > 0.01) else 0.01

    with open('savedparameters/mountaincarQ.pkl','wb') as f:
        pickle.dump(Q,f,pickle.HIGHEST_PROTOCOL)
    observation = env.reset()
    state = get_state(observation)
    done = False
    while not done:
        env.render()
        action = max_action(Q, state)
        observation, reward, done, _ = env.step(action)
        state = get_state(observation)
    env.close()


mountain_trial()
