import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Conv2D

# global variables for A3C
global episode
episode = 0
EPISODES = 8000000
# In case of BreakoutDeterministic-v3, always skip 4 frames
# Deterministic-v4 version use 4 actions
try:
    import gym
    test_env = gym.make("BreakoutDeterministic-v4")
    test_env.close()
    env_name = "BreakoutDeterministic-v4"
except:
    env_name = "ALE/Breakout-v5"

# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, action_size):
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        # optimizer parameters
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        self.threads = 8

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # TensorFlow 2.x - use eager execution (no session needed)
        self.actor_optimizer_obj = RMSprop(learning_rate=self.actor_lr, rho=0.99, epsilon=0.01)
        self.critic_optimizer_obj = RMSprop(learning_rate=self.critic_lr, rho=0.99, epsilon=0.01)

        # TensorBoard setup - modern TF2 API
        self.summary_writer = tf.summary.create_file_writer('summary/breakout_a3c')

    def train(self):
        # self.load_model("./save_model/breakout_a3c")
        agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic],
                        self.actor_optimizer_obj, self.critic_optimizer_obj,
                        self.discount_factor, self.summary_writer) for _ in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            time.sleep(60*10)
            self.save_model("./save_model/breakout_a3c")

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

# make agents(local) and start training
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, actor_optimizer, critic_optimizer, discount_factor, summary_writer):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.discount_factor = discount_factor
        self.summary_writer = summary_writer

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = self.build_localmodel()

        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.t_max = 20
        self.t = 0

    # Thread interactive with environment
    def run(self):
        # self.load_model('./save_model/breakout_a3c')
        global episode

        env = gym.make(env_name)

        step = 0

        while episode < EPISODES:
            done = False
            dead = False
            # 1 episode = 5 lives
            score, start_life = 0, 5
            observe = env.reset()
            if isinstance(observe, tuple):
                observe = observe[0]
            next_observe = observe

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, 30)):
                observe = next_observe
                result = env.step(1)
                next_observe = result[0]

            # At start of episode, there is no preceding frame. So just copy initial states to make history
            state = pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1
                observe = next_observe
                # get action for the current history and go one step in environment
                action, policy = self.get_action(history)
                # change action to real_action
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                step_result = env.step(real_action)
                next_observe = step_result[0]
                reward = step_result[1]
                done = step_result[2]
                # Handle both old and new gym API
                if len(step_result) == 5:
                    # New gym: (obs, reward, terminated, truncated, info)
                    done = step_result[2] or step_result[3]
                    info = step_result[4]
                else:
                    # Old gym: (obs, reward, done, info)
                    info = step_result[3]
                # pre-process the observation --> history
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                self.avg_p_max += np.amax(self.actor.predict(np.float32(history / 255.), verbose=0))

                # if the ball is fall, then the agent is dead --> episode is not over
                if 'ale.lives' in info and start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                self.memory(history, action, reward)

                # if agent is dead, then reset the history
                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                #
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_localmodel()
                    self.t = 0

                # if done, plot the score over episodes
                if done:
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:", step)

                    # Write summary for tensorboard
                    with self.summary_writer.as_default():
                        tf.summary.scalar('Total Reward/Episode', score, step=episode)
                        tf.summary.scalar('Average Max Prob/Episode',
                                        self.avg_p_max / float(step), step=episode)
                        tf.summary.scalar('Duration/Episode', step, step=episode)
                        self.summary_writer.flush()

                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1] / 255.), verbose=0)[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_model(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.critic.predict(states, verbose=0)
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values
        actions = np.array(self.actions)

        # Train actor
        with tf.GradientTape() as tape:
            policy = self.actor(states, training=True)
            action_prob = tf.reduce_sum(actions * policy, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-10)
            loss = tf.reduce_sum(cross_entropy * advantages)
            entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-10))
            actor_loss = loss + 0.01 * entropy

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Train critic
        with tf.GradientTape() as tape:
            values = self.critic(states, training=True)
            values = tf.reshape(values, [-1])
            critic_loss = tf.reduce_mean(tf.square(discounted_rewards - values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.states, self.actions, self.rewards = [], [], []

    def build_localmodel(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.set_weights(self.actor.get_weights())
        critic.set_weights(self.critic.get_weights())

        actor.summary()
        critic.summary()

        return actor, critic

    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_actor.predict(history, verbose=0)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3)
    global_agent.train()
