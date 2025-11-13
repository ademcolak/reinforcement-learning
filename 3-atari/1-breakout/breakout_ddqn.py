import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.losses import Huber

EPISODES = 50000


class DDQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # TensorFlow 2.x - use eager execution (no session needed)
        self.optimizer = RMSprop(learning_rate=0.00025, epsilon=0.01)

        self.avg_q_max, self.avg_loss = 0, 0

        # TensorBoard setup - modern TF2 API
        self.summary_writer = tf.summary.create_file_writer('summary/breakout_ddqn')

        if self.load_model:
            self.model.load_weights("./save_model/breakout_ddqn.h5")

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        # Use Huber loss (combines MSE and MAE)
        model.compile(loss=Huber(), optimizer=self.optimizer)
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history, verbose=0)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, self.action_size))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        # Get current Q values
        target = self.model.predict(history, verbose=0)
        # Get Q values from target model for next state
        value = self.model.predict(next_history, verbose=0)
        target_value = self.target_model.predict(next_history, verbose=0)

        # Update Q values for actions taken
        for i in range(self.batch_size):
            if dead[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i][action[i]] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]

        # Train the model
        loss = self.model.train_on_batch(history, target)
        self.avg_loss += loss

    def save_model(self, name):
        self.model.save_weights(name)

    # Write summary for tensorboard
    def write_summary(self, score, step, episode):
        with self.summary_writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                            self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                            self.avg_loss / float(step), step=episode)
            self.summary_writer.flush()


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    # Deterministic-v4 version use 4 actions
    try:
        env = gym.make('BreakoutDeterministic-v4')
    except:
        env = gym.make('ALE/Breakout-v5')  # Fallback to newer gym version

    agent = DDQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()
        if isinstance(observe, tuple):
            observe = observe[0]

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            result = env.step(1)
            observe = result[0]

        # At start of episode, there is no preceding frame.
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            step_result = env.step(real_action)
            observe = step_result[0]
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
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.), verbose=0)[0])

            # if the agent missed ball, agent is dead --> episode is not over
            if 'ale.lives' in info and start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, dead)
            # every some time interval, train model
            agent.train_replay()
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                if global_step > agent.train_start:
                    agent.write_summary(score, step, e + 1)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        if e % 1000 == 0:
            agent.save_model("./save_model/breakout_ddqn.h5")
