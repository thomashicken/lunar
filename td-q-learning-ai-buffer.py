#!/usr/bin/env python3
#

import numpy as np 
import gymnasium as gym

import sys
import argparse
import logging
import os.path
import joblib
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

import collections

class QFunction:
    def __init__(self, state_shape, n_actions):
        self.Q = self.create_Q_function(state_shape, n_actions)
        self.state_shape = state_shape
        self.num_actions = n_actions
        self.replay_buffer = collections.deque(maxlen=10000)
        return

    def record_experience(self, state, action, reward, next_state, epoch_done, epoch_truncated):
        self.replay_buffer.append((state, action, reward, next_state, epoch_done, epoch_truncated))
        return

    def n_actions(self):
        return self.num_actions

    def actions(self):
        return [action for action in range(self.n_actions())]

    def create_Q_function(self, state_shape, n_actions):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=state_shape))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(n_actions, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        return model

    def sample_replay_buffer(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones, truncateds = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)]
        return states, actions, rewards, next_states, dones, truncateds


    def train_from_experiences(self, batch_size, gamma):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, truncateds = self.sample_replay_buffer(batch_size)
        this_predictions = self.Q.predict(states, verbose=0)
        next_predictions = self.Q.predict(next_states, verbose=0)
        max_next_predictions = np.max(next_predictions, axis=1)
        completeds = dones | truncateds
        target_values = (rewards + (1-completeds)*gamma*max_next_predictions)
        target_values = target_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.num_actions)
        new_values = mask * target_values + (1-mask)*this_predictions
        self.Q.fit(states, new_values, epochs=1, verbose=0)
        return

    def update(self, state, action, next_state, reward, gamma, prediction, next_prediction):
        """
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        action is a python integer (0,1)
        next_state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        reward is a python float
        gamma is a python float

        """
        # This is what we want the quality value for action in state to be.
        target_quality_value = reward + gamma * np.max(next_prediction)
        # target_quality_value is a numpy float

        # Change for action's value to be target
        target_vec = prediction
        target_vec[0][action] = target_quality_value

        # cause the state to be a list of 1 state, because the fit() method needs lists of inputs and target outputs
        state = np.array([state])
        
        # Cause the network to update its weights to attempt to give this target value.
        self.Q.fit(state, target_vec, epochs=1, verbose=0)
        return

    def predict(self, state):
        #print("PR predict(state={}, shape={})".format(state, state.shape))
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        return prediction
    
    def get_best_action(self, state):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        action = np.argmax(prediction[0])
        return action

    def get_Q_value(self, state, action):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        return prediction[0][action]

    def get_best_action_value(self, state):
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        best_action = np.argmax(prediction[0])
        best_Q_value = prediction[0][action]
        return best_action, best_Q_value

    def show(self):
        # for state in self.states():
        #     state = np.array([state])
        #     prediction = self.Q.predict(state)
        #     print(prediction)
        state = np.array([[0.0,0.0,0.0,0.0]])
        prediction = self.Q.predict(state, verbose=0)
        print(prediction)
        return

    def save(self, model_file):
        self.Q.save(model_file)
        return

    def load(self, model_file):
        self.Q = keras.models.load_model(model_file)
        return
    
    def get_reward(self, state, action, next_state, reward):
        # Calculate reward components based on different criteria
        proximity_reward = self.calculate_proximity_reward(state, next_state)
        speed_reward = self.calculate_speed_reward(next_state)
        tilt_penalty = self.calculate_tilt_penalty(next_state)
        leg_contact_reward = self.calculate_leg_contact_reward(next_state)
        side_engine_penalty = self.calculate_side_engine_penalty(action)
        main_engine_penalty = self.calculate_main_engine_penalty(action)

        # Combine reward components
        reward += proximity_reward
        reward += speed_reward
        reward -= tilt_penalty
        reward += leg_contact_reward
        reward -= side_engine_penalty
        reward -= main_engine_penalty

        return reward

    def calculate_proximity_reward(self, state, next_state):
        # Calculate the distance to the landing pad
        distance_to_pad = np.sqrt((next_state[0] ** 2) + (next_state[1] ** 2))
        
        # Check if both thrusters are terminated and the lander is between the flags
        if next_state[5] == 0 and next_state[6] == 0 and next_state[0] > -0.2 and next_state[0] < 0.2:
            # Reward for being between the flags only if both thrusters are terminated
            proximity_reward = 100
        else:
            # Default proximity reward calculation
            proximity_reward = 10.0 * (1.0 - distance_to_pad)
        
        return proximity_reward

    def calculate_speed_reward(self, next_state):
        # Calculate the speed of the lander
        speed = np.sqrt((next_state[2] ** 2) + (next_state[3] ** 2))
        # Decrease reward as speed increases
        speed_reward = -0.1 * speed
        return speed_reward

    def calculate_tilt_penalty(self, next_state):
        # Decrease reward for tilting
        tilt_penalty = abs(next_state[4]) * 10.0  # Assuming angle is in radians
        return tilt_penalty

    def calculate_leg_contact_reward(self, next_state):
        # Increase reward for each leg in contact with the ground
        leg_contact_reward = 10 * (next_state[6] + next_state[7])
        return leg_contact_reward

    def calculate_side_engine_penalty(self, action):
        # Decrease reward if a side engine is firing
        side_engine_penalty = 0.03 if action in [1, 3] else 0.0
        return side_engine_penalty

    def calculate_main_engine_penalty(self, action):
        # Decrease reward if the main engine is firing
        main_engine_penalty = 0.3 if action == 2 else 0.0
        return main_engine_penalty

    def record_experience(self, state, action, reward, next_state, epoch_done, epoch_truncated):
        reward = self.get_reward(state, action, next_state, reward)
        self.replay_buffer.append((state, action, reward, next_state, epoch_done, epoch_truncated))
        return

def get_model_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-model.keras".format(environment_name)
    return model_file

def get_rewards_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-rewards.csv".format(environment_name)
    return model_file

# The openai gym environment is loaded
def load_environment(my_args):
    if my_args.environment == 'lunar':
        env = gym.make('LunarLander-v2', render_mode='human') # render_mode = None, 'ansi', 'human'
    else:
        raise Exception("Unexpected environment: {}".format(my_args.environment))
    # env.observation.n, env.action_space.n gives number of states and action in env loaded
    return env

def learn_epoch(Q, env, chance_epsilon, gamma, batch_size, my_args):
    action_list = Q.actions()

    # Reset environment, getting initial state
    state, info = env.reset()
    prediction = Q.predict(state)
    
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table temporal difference learning algorithm
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q table
        # To facilitate learning, have chance of random action
        # instead of always choosing the best action
        chance = np.random.sample(1)[0]
        if chance < chance_epsilon:
            action = np.random.choice(action_list)
        else:
            action = Q.get_best_action(state)

        # Take action, get the new state and reward
        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        reward = Q.get_reward(state, action, next_state, reward)
        Q.record_experience(state, action, reward, next_state, epoch_done, epoch_truncated)

        # Update Q-Table with new data
        Q.train_from_experiences(batch_size, gamma)
        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def evaluate_epoch(Q, env, my_args):
    action_list = Q.actions()

    # Reset environment, getting initial state
    state, info = env.reset()
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table policy evaluation
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q table
        action = Q.get_best_action(state)

        # Take action, get the new state and reward
        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)

        # Update reward and state
        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def Q_learn(Q, env, my_args):
    almost_one = my_args.epsilon_chance_factor
    gamma = my_args.gamma
    batch_size = my_args.batch_size
    epoch_rewards = [] # rewards per epochs
    chance_epsilon = almost_one

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = learn_epoch(Q, env, chance_epsilon, gamma, batch_size, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
            sys.stdout.flush()

        # make less likely to experiment
        # assumes positive scores for successful completion
        if epoch_total_reward > 40:
            chance_epsilon *= almost_one
            chance_epsilon = max(chance_epsilon, 0.01)
        if my_args.early_stop:
            if len(epoch_rewards) > 5:
                if sum(epoch_rewards[len(epoch_rewards)-5:]) == 5*500:
                    break
        
    return epoch_rewards

def Q_evaluate(Q, env, my_args):
    epoch_rewards = [] # rewards per epochs

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = evaluate_epoch(Q, env, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
        
    return epoch_rewards

def do_learn(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Build new Q-table structure
    # assumes that the environment has Box observation space and discrete action space
    Q = QFunction(env.observation_space.shape, env.action_space.n)

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    if os.path.exists(model_file):
        print("Model loading from {}.".format(model_file))
        Q.load(model_file)
    
    # Learn
    epoch_rewards = Q_learn(Q, env, my_args)

    print("Learn: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    Q.save(model_file)
    print("Model saved to {}.".format(model_file))

    rewards_file = get_rewards_filename(my_args.rewards_file, my_args.environment)
    df = pd.DataFrame(columns = ["epoch","reward"])
    for i in range(0, len(epoch_rewards)):
        df.loc[i] = [i, epoch_rewards[i]]
    df.to_csv(rewards_file, index=False)
    
    return

def do_score(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Load existing Q-Table
    # assumes that the environment has discrete observation and action spaces
    Q = QFunction([0], 0)
    model_file = get_model_filename(my_args.model_file, my_args.environment)
    print("Model loading from {}.".format(model_file))
    Q.load(model_file)


    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Q-Table Learning')
    parser.add_argument('action', default='learn',
                        choices=["learn", "score"],
                        nargs='?', help="desired action")
    
    parser.add_argument('--environment', '-e', default="lunar", type=str, choices=('lunar', ), help="name of the OpenAI gym environment")
    parser.add_argument('--model-file', '-m', default="", type=str, help="name of file for the model (default is constructed from environment)")
    parser.add_argument('--rewards-file', '-r', default="", type=str, help="name of file for the rewards (default is constructed from environment)")

    # Hyperparameters
    parser.add_argument('--gamma', '-g', default=0.99, type=float, help="Q-learning hyperparameter (default=0.99)")
    parser.add_argument('--epsilon-chance-factor', '-c', default=0.1, type=float, help="Scaling factor for learning policy chance of choosing random action (default=0.1)")
    parser.add_argument('--batch-size', '-b', default=32, type=int, help="number of experiences to learn from (default=32)")

    parser.add_argument('--n-epochs', '-n', default=1, type=int, help="number of episodes to run (default=1)")

    # Debugging/Observations
    parser.add_argument('--track-epochs', '-t', default=0, type=int, help="0 = don't display per-epoch information, 1 = do display per-epoch information (default=0)")
    parser.add_argument('--track-steps', '-s', default=0, type=int, help="0 = don't display per-step information, 1 = do display per-step information (default=0)")

    # Early Stopping
    parser.add_argument('--early-stop', action='store_true', help="Stop learning if perfect score is found.")
    parser.add_argument('--no-early-stop', dest="early_stop", action='store_false', help="Do not stop learning if perfect score is found.")
    parser.set_defaults(early_stop=False)

    my_args = parser.parse_args(argv[1:])

    # Do any special fixes/checks here
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'learn':
        do_learn(my_args)
    elif my_args.action == 'score':
        do_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)