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
        self.optimizer = self._create_optimizer()

    def train(self, states, actions, action_gradients):
        return self.optimizer([states, action_gradients])
        
    def predict(self, input_state):
        return self.actor.predict(input_state)

    def predict_target(self, input_state):
        return self.actor_target.predict(input_state)

    def save(self, path):
        self.actor.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.actor.load_weights(path)

    def _create_optimizer(self):
        action_gdts = keras_backend.placeholder(shape=(None, self._action_shape))
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        return keras_backend.function(
            inputs=[self._input_layer, action_gdts],
            outputs=[keras_backend.constant(1)],
            updates=[tf.train.AdamOptimizer(self._learning_rate).apply_gradients(grads)]
        )

    def _create_network(self):
        conv_model = Input(shape=self._state_shape)

        conv_model_1 = Conv2D(64, 5, 5,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 4, 4,  activation='relu')(conv_model_1)
        conv_model_1 = Conv2D(64, 3, 3,  activation='relu')(conv_model_1)
        conv_model_1 = MaxPooling2D(pool_size=(3, 3))(conv_model_1)
        
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

