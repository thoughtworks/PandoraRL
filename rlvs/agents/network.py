import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Convolution2D, Conv2D, MaxPooling2D, Concatenate, Input, GaussianNoise
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import keras.backend as keras_backend
import numpy as np

class Critic:
    def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
        self._tau = tau
        self._learning_rate = learning_rate
        self._state_shape = input_shape
        self._action_shape = action_shape
        
        self.critic, self._input_layer, self._action_layer = self._create_network()
        self.critic_target, _, __ = self._create_network()

        self._action_gradients = keras_backend.function(
            [self.critic.input[0], self.critic.input[1]], keras_backend.gradients(
                self.critic.output, [self.critic.input[1]]
            )
        )

    def predict_target(self, inp):
        return self.critic_target.predict(inp)
    
    def update_target_network(self):
        model_weights = np.array(self.critic.get_weights())
        target_weights = np.array(self.critic_target.get_weights())
        target_weights = self._tau * model_weights + (1 - self._tau) * target_weights 
        self.critic_target.set_weights(target_weights)

    def train_on_batch(self, states, actions, critic_target):
        return self.critic.train_on_batch([states, actions], critic_target)


    def action_gradients(self, states, actions):
        return self._action_gradients([states, actions])

    def _create_network(self):
        conv_model = Input(shape=self._state_shape)

        conv_model_1 = Conv2D(64, 5, 5,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 4, 4,  activation='relu')(conv_model_1)
        conv_model_1 = Conv2D(64, 3, 3,  activation='relu')(conv_model_1)
        conv_model_1 = MaxPooling2D(pool_size=(3, 3))(conv_model_1)
        
        
        conv_model_1 = Flatten()(conv_model_1)
        conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        conv_model_1 = Model(inputs=conv_model, outputs=conv_model_1)

        action_model = Input(shape=[self._action_shape])
        action_model_1 = Dense(256, activation='linear')(action_model)
        action_model_1 = Model(inputs=action_model, outputs=action_model_1)

        intermediate = Concatenate()([conv_model_1.output, action_model_1.output])
        intermediate = Dense(256, activation='relu')(intermediate)
        intermediate = Dropout(0.5)(intermediate)

        value_layer = Dense(1, activation='linear', kernel_initializer=RandomUniform())(intermediate)

        model = Model([conv_model, action_model], value_layer)
        model.compile(optimizer=Adam(lr=self._learning_rate), loss='mse')

        return model, conv_model, action_model


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
        self.optimizer([states, action_gradients])
        
    def predict(self, input_state):
        return self.actor.predict(input_state)

    def predict_target(self, input_state):
        return self.actor_target.predict(input_state)

    def _create_optimizer(self):
        action_gdts = keras_backend.placeholder(shape=(None, self._action_shape))
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        return keras_backend.function(
            inputs=[self.actor.input, action_gdts],
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
