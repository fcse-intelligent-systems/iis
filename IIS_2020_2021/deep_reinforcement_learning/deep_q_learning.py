import numpy as np
import random
from collections import deque
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape


class DQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param model: Keras model
        :param target_model: Keras model
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        if probability < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.model.predict(state)[0])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(f'dqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'dqn_{model_name}_{episode}.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                max_future_q = (reward + self.discount_factor *
                                np.amax(self.target_model.predict(next_state)[0]))
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)


class DDQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Double Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param model: Keras model
        :param target_model: Keras model
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        if probability < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.model.predict(state)[0])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(f'ddqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'ddqn_{model_name}_{episode}.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                max_action = np.argmax(self.model.predict(next_state)[0])
                max_future_q = (reward + self.discount_factor *
                                self.target_model.predict(next_state)[0][max_action])
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)


class DuelingDQN:
    def __init__(self, state_space_shape, num_actions, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Dueling Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    def _build_model(self, layers):
        """
        Builds a model with the given layers.
        :param layers: layers for the model
        """
        input_layer = Input(shape=self.state_space_shape)

        x = input_layer
        for layer in layers:
            x = layer(x)

        v = Dense(1)(x)
        a = Dense(self.num_actions)(x)

        q = (v + (a - reduce_mean(a, axis=1, keepdims=True)))

        model = Model(inputs=input_layer, outputs=q)
        model.compile(Adam(lr=self.learning_rate), loss=MeanSquaredError())
        return model

    def build_model(self, layers):
        """
        Builds the main and target network with the given layers.
        :param layers: layers for the models
        """
        self.model = self._build_model(layers)
        self.target_model = self._build_model(layers)
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        if probability < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.model.predict(state)[0])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(f'duelingdqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'duelingdqn_{model_name}_{episode}.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                max_future_q = (reward + self.discount_factor *
                                np.amax(self.target_model.predict(next_state)[0]))
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)


class DDPG:
    def __init__(self, state_space_shape, action_space_shape, learning_rate_actor=0.1, learning_rate_critic=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Actor Critic (Deep Deterministic Policy Gradient) agent.
        :param state_space_shape: shape of the observation space
        :param action_space_shape: shape of the action space
        :param learning_rate_actor: learning rate for the actor
        :param learning_rate_critic: learning rate for the critic
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.action_space_shape = action_space_shape
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    def _build_actor(self):
        """
        Builds the actor model.
        """
        input_layer = Input(shape=self.state_space_shape)

        x = Dense(16, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)

        actor_output = Dense(self.action_space_shape, activation='sigmoid')(x)

        actor = Model(inputs=input_layer, outputs=actor_output)
        actor.compile(optimizer=Adam(lr=self.learning_rate_actor))
        return actor

    def _build_critic(self):
        """
        Builds the critic model.
        """
        input_layer = Input(shape=self.state_space_shape)

        xs = Dense(16, activation='relu')(input_layer)
        xs = Dense(32, activation='relu')(xs)
        xs = Dense(16, activation='relu')(xs)

        input_action = Input(shape=self.action_space_shape)

        xa = Dense(16, activation='relu')(input_action)
        xa = Dense(32, activation='relu')(xa)
        xa = Dense(16, activation='relu')(xa)

        x = Concatenate()([xs, xa])
        x = Dense(16, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)

        critic_output = Dense(1, activation='relu')(x)

        critic = Model(inputs=[input_layer, input_action], outputs=critic_output)
        critic.compile(optimizer=Adam(lr=self.learning_rate_critic))
        return critic

    def build_model(self):
        """
        Builds the model.
        """
        self.actor = self._build_actor()
        self.target_actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_critic = self._build_critic()
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def _get_discrete_action(self, state, epsilon=0):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random()
        if probability < epsilon:
            return np.random.randint(0, self.action_space_shape)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.actor.predict(state)[0])

    def _get_continuous_action(self, state, epsilon=0):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random()
        if probability < epsilon:
            return np.random.uniform(low=0.0, high=1.0, size=self.action_space_shape)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return self.actor.predict(state)[0]

    def get_action(self, state, epsilon=0, discrete=True):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :param discrete: whether the action space is discrete or continuous
        :return:
        """
        if discrete:
            return self._get_discrete_action(state, epsilon)
        else:
            return self._get_continuous_action(state, epsilon)

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.actor.load_weights(f'actor_{model_name}_{episode}.h5')
        self.critic.load_weights(f'critic_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.actor.save_weights(f'actor_{model_name}_{episode}.h5')
        self.critic.save_weights(f'critic_{model_name}_{episode}.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state = [mb[0] for mb in minibatch]
        action = [mb[1] for mb in minibatch]
        reward = [mb[2] for mb in minibatch]
        next_state = [mb[3] for mb in minibatch]
        done = [mb[4] for mb in minibatch]

        states = convert_to_tensor(state, dtype=float32)
        actions = convert_to_tensor(action, dtype=float32)
        rewards = convert_to_tensor(reward, dtype=float32)
        next_states = convert_to_tensor(next_state, dtype=float32)
        dones = convert_to_tensor(done, dtype=float32)

        with GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            critic_value_ = squeeze(self.target_critic([next_states, target_actions]), 1)
            critic_value = squeeze(self.critic([states, actions]), 1)
            target = rewards + self.discount_factor * critic_value_ * (1 - dones)
            critic_loss = MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic([states, new_policy_actions])
            actor_loss = reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_space_shape, theta=.2, sigma=0.15, dt=1e-2, x0=None):
        """
        Initializes Ornstein Uhlenbeck action noise process.
        :param action_space_shape: shape of the action space
        :param theta: the rate of mean reversion
        :param sigma: scale of the noise
        :param dt: the timestep for the noise
        :param x0: the initial value for noise
        """
        self.mu = np.zeros(action_space_shape)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Returns action noise for one timestep.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the Ornstein Uhlenbeck action noise to the initial position.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
