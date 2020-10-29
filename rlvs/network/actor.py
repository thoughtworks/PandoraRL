import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate, Input, GaussianNoise, Conv3D, MaxPooling3D, add
from .graph_layer import GraphConv, GraphPool, GraphGather
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
        self.actor_target.set_weights(self.actor.get_weights())
        self.optimizer = Adam(learning_rate=learning_rate)

    def optimize(self, action_gradient):
        self.optimizer.apply_gradients(zip(action_gradient, self.actor.trainable_variables))

    def predict(self, input_state):
        return self.actor.predict(input_state)

    def save(self, path):
        self.actor.save_weights(filepath=path+'_actor.h5', save_format="h5")

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
        # actor_weights = np.array(self.actor_target.get_weights())
        # target_weights = np.array(self.actor.get_weights())
        
        # target_weights = self._tau * actor_weights + (1 - self._tau) * target_weights
        # self.actor_target.set_weights(target_weights)

        W, target_W = self.actor.get_weights(), self.actor_target.get_weights()
        for i in range(len(W)):
            target_W[i] = self._tau * W[i] + (1 - self._tau)* target_W[i]
        self.actor_target.set_weights(target_W)
        
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


class ActorGNN(Actor):
    def __init__(self, input_shape, action_shape, action_bounds, learning_rate, tau=0.001):
        self.adjecency_rank = 10
        self.action_bounds = action_bounds
        super(ActorGNN, self).__init__(input_shape, action_shape, learning_rate, tau)

    def _create_molecule_network(self, jj=0):
        features_input = Input(shape=(None, self._state_shape,), batch_size=1, name=f"actor_Feature_{jj}") 
        degree_slice_input = Input(shape=(11,2), dtype=tf.int32, batch_size=1, name=f"actor_Degree_slice_{jj}")
        membership = Input(shape=(None,), dtype=tf.int32, name=f'actor_membership_{jj}', batch_size=1)
        n_samples = Input(shape=(1,), dtype=tf.int32, name=f'actor_n_samples_{jj}', batch_size=1)
        deg_adjs_input = [Input(shape=(None,None,), dtype=tf.int32, batch_size=1, name=f"actor_deg_adjs_{jj}_{i}") for i in  range(self.adjecency_rank)]
        
        input_states = [features_input, degree_slice_input, membership, n_samples] + deg_adjs_input
        graph_layer = GraphConv(layer_id=jj, out_channel=64, activation_fn=tf.nn.relu)(input_states)

        graph_pool_in = [graph_layer, degree_slice_input, membership, n_samples] + deg_adjs_input
        graph_pool = GraphPool()(graph_pool_in)
        dense_layer = Dense(128, activation=tf.nn.relu)(graph_pool)
        graph_gather_layer = GraphGather(activation_fn=tf.nn.tanh)([dense_layer, membership, n_samples])
        return input_states, graph_gather_layer


    def _create_network(self):

        ip_1, graph_gather_layer_1 = self._create_molecule_network(0)
        ip_2, graph_gather_layer_2 = self._create_molecule_network(1)

        mol1_model = Model(inputs=ip_1, outputs=graph_gather_layer_1)
        mol2_model = Model(inputs=ip_2, outputs=graph_gather_layer_2)

        combination_layer = add([mol1_model.output, mol2_model.output])
        dense_layer_1 = Dense(64, activation="relu")(combination_layer)
        dense_layer_1 = GaussianNoise(1.0)(dense_layer_1)

        action_layer = Dense(
            self._action_shape,
            activation='tanh',
            kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)
        )(dense_layer_1)
        # action_layer *= self.action_bounds[1]
        
        model = Model([ip_1, ip_2], action_layer)
        return model, model.inputs
