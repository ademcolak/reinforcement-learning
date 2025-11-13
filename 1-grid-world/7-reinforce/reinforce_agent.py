import copy
import pylab
import numpy as np
import tensorflow as tf
from environment import Env
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

EPISODES = 2500


# this is REINFORCE Agent for GridWorld
class ReinforceAgent:
    def __init__(self):
        self.load_model = True
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model


    # get action from policy network
    def get_action(self, state):
        policy = self.model.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network using TF2 GradientTape
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        states = np.array(self.states)
        actions = np.array(self.actions)

        with tf.GradientTape() as tape:
            # Forward pass
            policy = self.model(states, training=True)
            # Calculate action probabilities
            action_prob = tf.reduce_sum(actions * policy, axis=1)
            # Calculate loss (policy gradient)
            cross_entropy = tf.math.log(action_prob + 1e-10) * discounted_rewards
            loss = -tf.reduce_sum(cross_entropy)

        # Backward pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # fresh env
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # update policy neural network for each episode
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")
