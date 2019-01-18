from keras import backend
import gym
import numpy as np
from keras.layers import Dense, Input, Dropout
from keras.models import Model, load_model
from keras import optimizers


class DqnBase:

    def __init__(self, epsilon_0, gamma, alpha, memory_d, batch_size):

        self.epsilon_0 = epsilon_0
        self.gamma = gamma
        self.alpha = alpha

        self.q_network = self.get_dnn_model()
        self.target_network = self.get_dnn_model()

        self.env = gym.make('CartPole-v0')

        # initiate Replay Memories
        self.memory_d = memory_d
        self.memory_st = np.zeros(shape=(self.memory_d, 4)).astype(float)
        self.memory_at = np.zeros(shape=(self.memory_d, 1)).astype(int)
        self.memory_rt = np.zeros(shape=(self.memory_d, 1)).astype(float)
        self.memory_st1 = np.zeros(shape=(self.memory_d, 4)).astype(float)
        self.memory_done = np.zeros(shape=(self.memory_d, 1)).astype(bool)

        self.batch_size = batch_size

    # control functions =====================================================
    def get_action(self, state, episode=0):
        """
        Function to calculate next action
        ε-greedy algorithem
        """

        epsilon = self.epsilon_0 * (1 / (episode + 1))
        if epsilon < 0.1:
            epsilon = 0.1
        if epsilon <= np.random.uniform(0, 1):
            q_vals = self.predict_network(net_type="Q", state=state.reshape(-1, 4))
            next_action = np.argmax(q_vals)
        else:
            next_action = np.random.choice([0, 1])
        return next_action

    def save_states(self, save_index, state, next_action, reward, next_state, done):
        """
        self.memory_st = np.zeros(shape=(self.memory_d, 4)).astype(float)
        self.memory_at = np.zeros(shape=(self.memory_d, 1)).astype(int)
        self.memory_rt = np.zeros(shape=(self.memory_d, 1)).astype(float)
        self.memory_st1 = np.zeros(shape=(self.memory_d, 4)).astype(float)
        self.memory_done = np.zeros(shape=(self.memory_d, 1)).astype(bool)
        """
        index = save_index % self.memory_d

        self.memory_st[index] = state
        self.memory_at[index] = next_action
        self.memory_rt[index] = reward
        self.memory_st1[index] = next_state
        self.memory_done[index] = done

    def sample_records(self, save_index):
        sample_index = np.random.randint(0, save_index, size=self.batch_size)
        sample_st = self.memory_st[sample_index]
        sample_at = self.memory_at[sample_index]
        sample_rt = self.memory_rt[sample_index]
        sample_st1 = self.memory_st1[sample_index]
        sample_done = self.memory_done[sample_index]
        return [sample_st, sample_at, sample_rt, sample_st1, sample_done]

    def get_target_value(self, sample_data):
        """
        recieve list of samples data
                sample_st [batch, 4]
                sample_at [batch, 1]
                sample_rt [batch, 1]
                sample_st1 [batch, 4]
                sample_done [batch, 1]
        returns target value [batch]
        """
        predicted_q1 = self.predict_network(net_type="target", state=sample_data[3])
        max_q1 = np.max(predicted_q1, axis=1, keepdims=True)
        target_value = np.where(sample_data[4], sample_data[2],
                                sample_data[2] + self.gamma * max_q1)
        return target_value

    # gym functions ===========================================================
    def reset_env(self):
        observation = self.env.reset()
        return observation

    def run_step(self, next_action):
        return self.env.step(next_action)

    # DNN functions ===========================================================
    def get_dnn_model(self, input_length=4, dropout_rate=0.5, seed=1):
        """
        Making DNN prediction model.

        Input: input_length
        """

        # build model
        input_data = Input(shape=(input_length,), name="input_data", dtype=np.float32)
        x = Dense(units=24, activation="relu")(input_data)
        x = Dense(units=24, activation="relu")(x)
        output_val = Dense(units=2, activation=None)(x)  # [batch, 2]

        # compile model
        dnn_model = Model(inputs=input_data,
                          outputs=output_val)
        adam = optimizers.Adam(lr=self.alpha)
        dnn_model.compile(optimizer=adam, loss=self.clip_loss)

        return dnn_model

    def predict_network(self, net_type, state):
        """
        Get Q value of next actions.

        net_type: "Q" or "target"
        state: np.ndarray of states. shape=[batch_size,4].
        """

        if net_type == "Q":
            model = self.q_network
        if net_type == "target":
            model = self.target_network

        return model.predict(state)

    def train_network(self, state, action, target_value):
        """
        """
        current_q = self.q_network.predict(state)

        train_target = np.array([np.where(action == 0, target_value,
                                          current_q[:, 0].reshape([self.batch_size, 1])),
                                 np.where(action == 1, target_value,
                                          current_q[:, 1].reshape([self.batch_size, 1]))]
                                ).T.reshape([self.batch_size, 2])

        self.q_network.train_on_batch(state, train_target)

    def clip_loss(self, y_true, y_pred):
        """
        calculate loss function

        y_true, y_pred: Q values after each action. [batch, 2]
        """

        error = y_true - y_pred
        loss = backend.cast(error <= 1, "float32") * backend.cast(error >= -1, "float32") \
            * error**2 * 0.5 + \
            backend.cast(error > 1, "float32") * (error - 0.5) + \
            backend.cast(error < -1, "float32") * ((-1) * error - 0.5)
        return backend.mean(backend.mean(loss, axis=0), axis=0)

    def copy_q_to_target(self):
        self.target_network.set_weights(self.q_network.get_weights())
