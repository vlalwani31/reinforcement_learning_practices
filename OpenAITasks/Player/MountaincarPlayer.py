import gym
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

def player():
    env = gym.make('MountainCar-v0')
    Q = {}
    with open('../savedparameters/mountaincarQ.pkl','rb') as f:
        Q = pickle.load(f)
    observation = env.reset()
    state = get_state(observation)
    done = False
    while not done:
        env.render()
        action = max_action(Q, state)
        observation, reward, done, _ = env.step(action)
        state = get_state(observation)
    env.close()


player()
