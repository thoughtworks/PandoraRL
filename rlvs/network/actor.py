import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate, Input, GaussianNoise, Conv3D, MaxPooling3D
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import keras.backend as keras_backend
import numpy as np

class Actor:
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
        self._learning_rate = learning_rate
        self._tau = tau
        self._state_shape = input_shape
        self._action_shape = action_shape
        
        self.actor, self._input_layer = self._create_network()
        self.actor_target, _ = self._create_network()
        self.optimizer = Adam(learning_rate=learning_rate)

    def optimize(self, action_gradient):
        self.optimizer.apply_gradients(zip(action_gradient, self.actor.trainable_variables))

    def predict(self, input_state):
        return self.actor.predict(input_state)

    def save(self, path):
        self.actor.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.actor.load_weights(path)

    def _create_network(self):
        conv_model = Input(shape=self._state_shape)

        conv_model_1 = Conv2D(64, (5, 5),  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, (4, 4),  activation='relu')(conv_model_1)
        conv_model_1 = Conv2D(64, (3, 3),  activation='relu')(conv_model_1)
        conv_model_1 = MaxPooling2D(pool_size=(2, 2))(conv_model_1)
        
        conv_model_1 = Flatten()(conv_model_1)
        conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        conv_model_1 = Dense(128, activation='relu')(conv_model_1)
        conv_model_1 = GaussianNoise(1.0)(conv_model_1)

        action_layer = Dense(
            self._action_shape,
            activation='tanh',
            kernel_initializer=RandomUniform()
        )(conv_model_1)

        model = Model(conv_model, action_layer)
        return model, conv_model

    def update_target_network(self):
        actor_weights = np.array(self.actor_target.get_weights())
        target_weights = np.array(self.actor.get_weights())
        
        target_weights = self._tau * actor_weights + (1 - self._tau) * target_weights
        self.actor_target.set_weights(target_weights)

        
class Actor3D(Actor):
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
        super(Actor3D, self).__init__(input_shape, action_shape, learning_rate, tau)

    def _create_network(self):
        conv_model = Input(shape=self._state_shape)

        conv_model_1 = Conv3D(64, (5, 5, 5), activation='relu')(conv_model)
        conv_model_1 = Conv3D(64, (4, 4, 4), activation='relu')(conv_model_1)
        conv_model_1 = Conv3D(64, (3, 3, 3), activation='relu')(conv_model_1)
        conv_model_1 = MaxPooling3D(pool_size=(3, 3, 3))(conv_model_1)
        
        conv_model_1 = Flatten()(conv_model_1)
        conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        conv_model_1 = Dense(128, activation='relu')(conv_model_1)
        conv_model_1 = GaussianNoise(1.0)(conv_model_1)

        action_layer = Dense(
            self._action_shape,
            activation='tanh',
            kernel_initializer=RandomUniform()
        )(conv_model_1)

        model = Model(conv_model, action_layer)
        return model, conv_model

