import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling3D, Convolution3D, Conv3D, MaxPooling3D, Concatenate, Input, GaussianNoise, add
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as keras_backend
import numpy as np

from rlvs.network.graph_layer import GraphConv, GraphPool, GraphGather

class Critic:
    # def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
    def __init__(self, action_shape, learning_rate, tau=0.001):    
        self._tau = tau
        self._learning_rate = learning_rate
        # self._state_shape = input_shape
        self._action_shape = action_shape
        self.critic, self._input_layer, self._action_layer = self._create_network()
        self.critic_target, _, __ = self._create_network()

        self._action_gradients = keras_backend.function(
            [self._input_layer, self._action_layer], keras_backend.gradients(
                self.critic.output, [self._action_layer]
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

    def save(self, path):
        self.critic.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.critic.load_weights(path)

    def _create_network(self):
        # conv_model = Input(shape=self._state_shape)

        # conv_model_1 = Conv3D(64, 5, 5, 5,  activation='relu')(conv_model)
        # conv_model_1 = Conv3D(64, 4, 4, 4,  activation='relu')(conv_model_1)
        # conv_model_1 = Conv3D(64, 3, 3, 3,  activation='relu')(conv_model_1)
        # conv_model_1 = MaxPooling3D(pool_size=(3, 3, 3))(conv_model_1)
        
        
        # conv_model_1 = Flatten()(conv_model_1)
        # conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        # conv_model_1 = Model(inputs=conv_model, outputs=conv_model_1)

        features_input_1 = Input(shape=(None,18,), batch_size=1)
        degree_slice_input_1 = Input(shape=(11,2), dtype=tf.int32, batch_size=1)
        deg_adjs_input_1 = []
        for i in range(10):
            deg_adjs_input_1.append(Input(shape=(None,None,), dtype=tf.int32, batch_size=1))
        ip_1 = [features_input_1, degree_slice_input_1] + deg_adjs_input_1

        graph_layer_1 = GraphConv(out_channel=64, activation_fn=tf.nn.relu)(ip_1)
        gp_in_1 = [graph_layer_1, degree_slice_input_1] + deg_adjs_input_1
        graph_pool_1 = GraphPool()(gp_in_1)
        dense_layer_1 = Dense(128, activation="relu")(graph_pool_1)
        graph_gather_layer_1 = GraphGather(activation_fn=tf.nn.relu)(dense_layer_1)

        mol1_model = Model(inputs=ip_1, outputs=graph_gather_layer_1)

        features_input_2 = Input(shape=(None,18,), batch_size=1)
        degree_slice_input_2 = Input(shape=(11,2), dtype=tf.int32, batch_size=1)
        deg_adjs_input_2 = []
        for i in range(10):
            deg_adjs_input_2.append(Input(shape=(None,None,), dtype=tf.int32, batch_size=1))
        ip_2 = [features_input_2, degree_slice_input_2] + deg_adjs_input_2

        graph_layer_2 = GraphConv(out_channel=64, activation_fn=tf.nn.relu)(ip_2)
        gp_in_2 = [graph_layer_2, degree_slice_input_2] + deg_adjs_input_2
        graph_pool_2 = GraphPool()(gp_in_2)
        dense_layer_2 = Dense(128, activation="relu")(graph_pool_2)
        graph_gather_layer_2 = GraphGather(activation_fn=tf.nn.relu)(dense_layer_2)

        mol2_model = Model(inputs=ip_2, outputs=graph_gather_layer_2)

        combination_layer = add([mol1_model.output, mol2_model.output])
        combined_dense_layer = Dense(64, activation="relu")(combination_layer)
        conv_model_1 = Model([ip_1, ip_2], combined_dense_layer)

        action_model = Input(shape=[self._action_shape])
        action_model_1 = Dense(64, activation='linear')(action_model)
        action_model_1 = Model(inputs=action_model, outputs=action_model_1)

        intermediate = Concatenate()([conv_model_1.output, action_model_1.output])
        intermediate = Dense(64, activation='relu')(intermediate)
        intermediate = Dropout(0.5)(intermediate)

        value_layer = Dense(1, activation='linear', kernel_initializer=RandomUniform())(intermediate)

        # model = Model([conv_model, action_model], value_layer)
        model = Model([[ip_1, ip_2], action_model], value_layer)
        model.compile(optimizer=Adam(lr=self._learning_rate), loss='mse')

        # return model, conv_model, action_model
        return model, tf.concat([ip_1[0], ip_2[0]], axis=1), action_model

class Actor:
    # def __init__(self, input_shape, action_shape, learning_rate, tau=0.001):
    def __init__(self, action_shape, learning_rate, tau=0.001):
        self._learning_rate = learning_rate
        self._tau = tau
        # self._state_shape = input_shape
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

    def update_target_network(self):
        actor_weights = np.array(self.actor_target.get_weights())
        target_weights = np.array(self.actor.get_weights())
        
        target_weights = self._tau * actor_weights + (1 - self._tau) * target_weights
        self.actor_target.set_weights(target_weights)

    def save(self, path):
        self.actor.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.actor.load_weights(path)

    def _create_optimizer(self):
        action_gdts = keras_backend.placeholder(shape=(None, self._action_shape))
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_weights)

        params_grad = tape.gradient(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        return keras_backend.function(
            inputs=[self._input_layer, action_gdts],
            outputs=[keras_backend.constant(1)],
            updates=[tf.optimizers.Adam(self._learning_rate).apply_gradients(grads)]
        )

    def _create_network(self):
        # conv_model = Input(shape=self._state_shape)

        # conv_model_1 = Conv3D(64, 5, 5, 5, activation='relu')(conv_model)
        # conv_model_1 = Conv3D(64, 4, 4, 4, activation='relu')(conv_model_1)
        # conv_model_1 = Conv3D(64, 3, 3, 3, activation='relu')(conv_model_1)
        # conv_model_1 = MaxPooling3D(pool_size=(3, 3, 3))(conv_model_1)
        
        # conv_model_1 = Flatten()(conv_model_1)
        # conv_model_1 = Dense(256, activation='relu')(conv_model_1)
        # conv_model_1 = Dense(128, activation='relu')(conv_model_1)

        features_input_1 = Input(shape=(None,18,), batch_size=1)
        degree_slice_input_1 = Input(shape=(11,2), dtype=tf.int32, batch_size=1)
        deg_adjs_input_1 = []
        for i in range(10):
            deg_adjs_input_1.append(Input(shape=(None,None,), dtype=tf.int32, batch_size=1))
        ip_1 = [features_input_1, degree_slice_input_1] + deg_adjs_input_1

        graph_layer_1 = GraphConv(out_channel=64, activation_fn=tf.nn.relu)(ip_1)
        gp_in_1 = [graph_layer_1, degree_slice_input_1] + deg_adjs_input_1
        graph_pool_1 = GraphPool()(gp_in_1)
        dense_layer_1 = Dense(128, activation="relu")(graph_pool_1)
        graph_gather_layer_1 = GraphGather(activation_fn=tf.nn.relu)(dense_layer_1)

        mol1_model = Model(inputs=ip_1, outputs=graph_gather_layer_1)

        features_input_2 = Input(shape=(None,18,), batch_size=1)
        degree_slice_input_2 = Input(shape=(11,2), dtype=tf.int32, batch_size=1)
        deg_adjs_input_2 = []
        for i in range(10):
            deg_adjs_input_2.append(Input(shape=(None,None,), dtype=tf.int32, batch_size=1))
        ip_2 = [features_input_2, degree_slice_input_2] + deg_adjs_input_2

        graph_layer_2 = GraphConv(out_channel=64, activation_fn=tf.nn.relu)(ip_2)
        gp_in_2 = [graph_layer_2, degree_slice_input_2] + deg_adjs_input_2
        graph_pool_2 = GraphPool()(gp_in_2)
        dense_layer_2 = Dense(128, activation="relu")(graph_pool_2)
        graph_gather_layer_2 = GraphGather(activation_fn=tf.nn.relu)(dense_layer_2)

        mol2_model = Model(inputs=ip_2, outputs=graph_gather_layer_2)

        combination_layer = add([mol1_model.output, mol2_model.output])
        conv_model_1 = Dense(64, activation="relu")(combination_layer)

        conv_model_1 = GaussianNoise(1.0)(conv_model_1)

        action_layer = Dense(
            self._action_shape,
            activation='tanh',
            kernel_initializer=RandomUniform()
        )(conv_model_1)

        # model = Model(conv_model, action_layer)
        model = Model([ip_1, ip_2], action_layer)

        # return model, conv_model
        return model, tf.concat([ip_1[0], ip_2[0]], axis=1)