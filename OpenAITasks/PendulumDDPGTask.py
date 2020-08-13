import gym
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, backend
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Pendulum-v0')
np.random.seed(0)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
# (OA stands for Ornstein Uhlenbeck)
class OUActionNoise(object):
    def __init__(self, mean, sigma=0.15, std_dev=0.2, dt=1e-2, x0=None):
        self.mean = mean
        self.sigma = sigma
        self.std_dev = std_dev
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.std_dev*(self.mean - self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)

class ReplayBuffer(object):
    def __init__(self, input_shape, n_actions, batch_size = 64, maxsize=1000000):
        self.mem_size = maxsize
        self.batch_size = batch_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size, 1))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_step(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter = self.mem_counter + 1


    def learn(self):
        # Get Sampling Range:
        record_range = min(self.mem_counter, self.mem_size)
        # Get random sample indexs
        batch_indices = np.random.choice(record_range, size=self.batch_size)
        # Convert to Tensors
        state_batch = tf.convert_to_tensor(self.state_memory[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_memory[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_memory[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.new_state_memory[batch_indices])
        # Train and Update Actor and Critic Network
        # First Critic
        with tf.GradientTape() as tape:
            # Use Target Actor with next state
            target_actions = target_actor(tf.cast(next_state_batch,tf.float32))
            # Use Target Critic with Target Actor's output and calculate Q-Value
            y = reward_batch + (gamma * target_critic([tf.cast(next_state_batch,tf.float32), target_actions]))
            # Get Critic's Output with previous state
            critic_value = critic_model([tf.cast(state_batch,tf.float32), tf.cast(action_batch,tf.float32)])
            # Calculate L2 loss between Target Critic's and Critic's output
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
        # Second Actor
        with tf.GradientTape() as tape:
            # Use Actor with previous state and get actions
            actions = actor_model(tf.cast(state_batch,tf.float32))
            # Use Critic with previous state and Actor's output
            critic_value = critic_model([tf.cast(state_batch,tf.float32), actions])
            # Wanted to maximize Critic's Q-value, so we try to minimize
            # -Q-value
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

# Updates Target Networks
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append((variable * tau) + (target_variables[i] * (1 - tau)))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append((variable * tau) + (target_variables[i] * (1 - tau)))

    target_actor.set_weights(new_weights)

# Outputs model for Actor Network
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    initializer = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    #last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(3,))
    out = layers.Dense(400,activation='relu')(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(300,activation='relu')(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1,activation='tanh',kernel_initializer=initializer)(out)
    # The upper_bound is 2.0 for Pendulum
    outputs = layers.Lambda(lambda x: x*2.0)(outputs)
    #outputs = outputs * upper_bound

    model = models.Model(inputs, outputs)
    return model

def get_critic():
    # Using States as Input
    state_input = layers.Input(shape=(3,))
    state_out = layers.Dense(16,activation='relu')(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32,activation='relu')(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Including Actions as Input
    action_input = layers.Input(shape=(1,))
    action_out = layers.Dense(32,activation='relu')(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Concatenate State and Action
    concat = layers.Concatenate()([state_out,action_out])

    out = layers.Dense(400,activation='relu')(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(300,activation='relu')(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single Q-value for a given state-action pair
    model = models.Model([state_input,action_input], outputs)
    return model

def policy(state, noise_object):
    #sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = actor_model(tf.cast(state,tf.float32))
    # Initialize a Noise Object
    noise = noise_object()
    # Convert the Tensor output of Model to an numpy array
    sampled_actions = backend.eval(sampled_actions)
    # Adding noise to action
    sampled_actions = sampled_actions + noise
    # Making sure that actions are within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

# All Training Parameters
ou_noise = OUActionNoise(mean=np.zeros(1))

# Main Models
actor_model = get_actor()
critic_model = get_critic()

# Target Models
target_actor = get_actor()
target_critic = get_critic()

# Making Weights similar
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rates for actor-critic
actor_lr = 0.001
critic_lr = 0.002

actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
buffer = ReplayBuffer([num_states],num_actions)
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
for ep in range(total_episodes):
    prev_state = env.reset()
    score = 0
    done = False
    while not done:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        state, reward, done, info = env.step(action)
        buffer.store_step(prev_state, action, reward, state, done)
        score += reward
        buffer.learn()
        update_target(tau)
        prev_state = state
    ep_reward_list.append(score)
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
# Ploting Results
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
# Saving Models
actor_model.save('./savedparameters/Pendulum_DDPG/actor_model.h5')
critic_model.save('./savedparameters/Pendulum_DDPG/critic_model.h5')
target_actor.save('./savedparameters/Pendulum_DDPG/target_actor.h5')
target_critic.save('./savedparameters/Pendulum_DDPG/target_critic.h5')



# Actor Network: Takes in the Observation, outputs the action
# To Train Actor Network, use Actor Network's input and output as inputs for
# Critic Network to obtain the output, which is Q-value. Then try to maximize
# the Q-value or minimize (-Q)-Value.
