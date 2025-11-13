import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D

EPISODES = 50000


class TestAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.no_op_steps = 20

        self.model = self.build_model()
        # TensorFlow 2.x uses eager execution (no session needed)

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

        return model

    def get_action(self, history):
        if np.random.random() < 0.01:
            return random.randrange(3)
        history = np.float32(history / 255.0)
        q_value = self.model.predict(history, verbose=0)
        return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    try:
        env = gym.make('BreakoutDeterministic-v4')
    except:
        env = gym.make('ALE/Breakout-v5')

    agent = TestAgent(action_size=3)
    agent.load_model("./save_model/breakout_dqn_5.h5")

    for e in range(EPISODES):
        done = False
        dead = False
       
        step, score, start_life = 0, 0, 5
        observe = env.reset()
        if isinstance(observe, tuple):
            observe = observe[0]

        for _ in range(random.randint(1, agent.no_op_steps)):
            result = env.step(1)
            observe = result[0]

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            step += 1

            action = agent.get_action(history)

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            if dead:
                real_action = 1
                dead = False

            step_result = env.step(real_action)
            observe = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            # Handle both old and new gym API
            if len(step_result) == 5:
                done = step_result[2] or step_result[3]
                info = step_result[4]
            else:
                info = step_result[3]

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if 'ale.lives' in info and start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward

            history = next_history

            if done:
                print("episode:", e, "  score:", score)

