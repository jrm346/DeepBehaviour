import numpy
import tensorflow as tf


class SimpleAgent:
    def __init__(self, name, inputs, outputs, first_layer=5, second_layer=5):
        # type: (str, int, int, int, int) -> None

        self.name = name
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(inputs,)))
        self.model.add(tf.keras.layers.Dense(first_layer, activation='relu'))
        self.model.add(tf.keras.layers.Dense(second_layer, activation='relu'))
        self.model.add(tf.keras.layers.Dense(outputs, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    def predict(self, environment):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Takes an environment and returns the agents predictions

        :param environment: The environment in which the agent must choose an action
        :return: The predictions for which action to choose
        """
        return self.model.predict(environment)

    def train(self, environment, true_rewards):
        # type: (numpy.ndarray, numpy.ndarray) -> None
        self.model.fit(environment, true_rewards, verbose=0)
