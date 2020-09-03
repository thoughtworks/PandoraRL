import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Convolution2D, Conv2D, MaxPooling2D, Concatenate, Input
from tensorflow.keras.optimizers import Adam
import numpy as np


class Critic:
    def __init__(self, input_shape, action_shape, learning_rate):
        self.critic = self._create_network(input_shape, action_shape, learning_rate)
        self.critic_target = self._create_network(input_shape, action_shape, learning_rate)

    def predict(self, states, actions):
        return self.critic.predict((actions, states))

    def predict_target(self, states, actions):
        return self.critic_target.predict((actions, states))

    def fit(self, states, actions, q_values):
        return self.critic.fit((actions, states), q_values, epochs = 20, batch_size = 1)

    def action_gradients(self, states, actions):
        return tf.gradients(self.critic((actions, states)),  (actions, states))


    def _create_network(self, input_shape, action_shape, learning_rate):
        conv_model = Input(shape=input_shape)

        conv_model_1 = Conv2D(64, 5, 5,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 4, 4,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 3, 3,  activation='relu')(conv_model)

        conv_model_1 = Flatten()(conv_model_1)
        conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        conv_model_1 = Model(inputs=conv_model, outputs=conv_model_1)

        action_model = Input(shape=[action_shape])
        action_model_1 = Dense(256, activation='relu')(action_model)
        action_model_1 = Model(inputs=action_model, outputs=action_model_1)

        intermediate = Concatenate()([action_model_1.output, conv_model_1.output])
        intermediate = Dense(256, activation='relu')(intermediate)
        intermediate = Dropout(0.5)(intermediate)

        value_layer = Dense(1, activation='linear')(intermediate)

        model = Model([action_model, conv_model], value_layer)

        model.compile(optimizer=Adam(lr=learning_rate), 
              loss='binary_crossentropy',
              metrics=['accuracy']
        )

        return model

    def update_target_network(self):
        decay=0.99
        temp1 = np.array(self.critic_target.get_weights())
        temp2 = np.array(self.critic.get_weights())
        
        temp3 = decay * temp1 + (1 - decay) * temp2
        self.critic_target.set_weights(temp3)



class Actor:
    def __init__(self, input_shape, action_shape, learning_rate):                
        self.action_grads = None
        
        self.actor = self._create_network(input_shape, action_shape, learning_rate)
        self.actor_target = self._create_network(input_shape, action_shape, learning_rate)

    def predict(self, input_state):
        return self.actor.predict(input_state)

    def tensors(self, input_state):
        return self.actor(input_state)

    def fit(self, states, actions):
        return self.actor.fit(states, actions, epochs = 20, batch_size = 1, steps_per_epoch=1)

    def predict_target(self, input_state):
        return self.actor_target.predict(input_state)


    def _create_network(self, input_shape, action_shape, learning_rate):
        conv_model = Input(shape=input_shape)

        conv_model_1 = Conv2D(64, 5, 5,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 4, 4,  activation='relu')(conv_model)
        conv_model_1 = Conv2D(64, 3, 3,  activation='relu')(conv_model)

        conv_model_1 = Flatten()(conv_model_1)
        conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        conv_model_1 = Dense(128, activation='relu')(conv_model_1)

        action_layer = Dense(action_shape, activation='linear')(conv_model_1)

        model = Model(conv_model, action_layer)

        model.compile(optimizer=Adam(lr=learning_rate), 
              loss='binary_crossentropy',
              metrics=['accuracy']
        )

        return model        

    def update_target_network(self):
        decay=0.99
        temp1 = np.array(self.actor_target.get_weights())
        temp2 = np.array(self.actor.get_weights())
        
        temp3 = decay * temp1 + (1 - decay) * temp2
        self.actor_target.set_weights(temp3)
