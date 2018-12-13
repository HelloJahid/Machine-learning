"""
        Author :: Jahid Hasan
        Facebook :: https://www.facebook.com/profile.php?id=100004472716558

"""

# OpenAI
import numpy as np
import gym

# keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

# RL
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# set the relavant variable
ENV_NAME = 'CartPole-v0'   # game enviornemnt

# get the environment and extract the number of actions available in CartPole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
nb_actions = env.action_space.n



# ===========================================  Keras Model ===========================

class NeuralNetwork():

    # constructor
    def __init__ (self):
         self.relu = 'relu'   # activation function
         self.linear = 'linear'  # activation function

    def kerasModel(self):
        # Now build a neural network model
        model = Sequential()
        model.add(Flatten(input_shape = (1, ) + env.observation_space.shape ))
        model.add(Dense(16))
        model.add(Activation(self.relu))
        model.add(Dense(nb_actions))
        model.add(Activation(self.linear))

        return model  # return the Sequential model



# ================================  Deep Q Network  ===========================

class DQN():

    # constructor
    def __init__ (self, model, nb_actions,  epsilon, learning_rate):
        self.model = model
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate


    def DQNmodel(self):
        policy = EpsGreedyQPolicy(self.epsilon)
        memory = SequentialMemory(limit=50000, window_length = 1) # memory buffer
        dqn =  DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=memory, nb_steps_warmup =10,
                        target_model_update = 1e-2, policy=policy )
        dqn.compile(Adam(lr=self.learning_rate), metrics=['mae'])

        return dqn

# NeuralNetwork class instance
nn = NeuralNetwork()
model = nn.kerasModel()   # get keras Sequential model

# Deep Q Network class instance
dq_network = DQN(model=model, nb_actions=nb_actions, epsilon=.1, learning_rate=1e-3)
dqn = dq_network.DQNmodel()  # final network


# okay!! train our network
dqn.fit(env, nb_steps=10000, visualize=True, verbose=2)
print("Done!! training")


# Okay!! test our network
dqn.test(env, nb_episodes=10, visualize=True)
print("Okay!! testing")
