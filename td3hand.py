from Env import CompressionEnv
import gym
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import layers
import gym


env = CompressionEnv()

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in batch])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

def build_actor(state_shape, action_shape):
    inputs = layers.Input(shape=state_shape)
    out = layers.Dense(400, activation='relu')(inputs)
    out = layers.Dense(300, activation='relu')(out)
    outputs = layers.Dense(action_shape[0], activation='tanh')(out)
    model = tf.keras.Model(inputs, outputs)
    return model

def build_critic(state_shape, action_shape):
    state_input = layers.Input(shape=state_shape)
    action_input = layers.Input(shape=action_shape)
    concat = layers.Concatenate()([state_input, action_input])
    
    out = layers.Dense(400, activation='relu')(concat)
    out = layers.Dense(300, activation='relu')(out)
    outputs = layers.Dense(1)(out)
    
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

class DDPGAgent:
    def __init__(self, state_shape, action_shape, action_bounds, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        
        self.actor = build_actor(state_shape, action_shape)
        self.critic = build_critic(state_shape, action_shape)
        
        self.target_actor = build_actor(state_shape, action_shape)
        self.target_critic = build_critic(state_shape, action_shape)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.update_target(self.target_actor.variables, self.actor.variables, tau=1)
        self.update_target(self.target_critic.variables, self.critic.variables, tau=1)
    
    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1 - tau))
    
    def policy(self, state, noise_std=0.2):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self.actor(state).numpy()[0]
        noise = np.random.normal(0, noise_std, size=self.action_shape)
        action = np.clip(action + noise, -self.action_bounds, self.action_bounds)
        return action
    
    def train(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])
            target_q_values = rewards + self.gamma * target_q_values * (1 - dones)
            
            q_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            critic_value = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(critic_value)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic.variables, self.tau)

state_shape = env.observation_space.shape
action_shape = env.action_space.shape
action_bounds = env.action_space.high

agent = DDPGAgent(state_shape, action_shape, action_bounds)
replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=64)

episodes = 100
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(200):
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        
        agent.train(replay_buffer)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode: {episode}, Reward: {episode_reward}")
