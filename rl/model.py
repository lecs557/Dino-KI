"""
Reinforcement learning model for the dino game.
Deep Q-Learning
"""

import random
import numpy as np
from collections import deque

class DQN(object):
    def __init__(
        self, model, observation_size, action_size, memory_size,
        gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995
    ):
        assert isinstance(observation_size, list)
        assert isinstance(action_size, int)
        assert isinstance(memory_size, int)
        assert isinstance(gamma, float)
        assert isinstance(epsilon, float)
        assert isinstance(epsilon_min, float)
        assert isinstance(epsilon_decay, float)
        # Shape der Observationen, die das Neuronale Netz bekommt.
        self.obs_size = observation_size
        # Wieviele Aktionen ausgeführt werden können.
        self.act_size = action_size
        # Neuronales Netz mit keras-api.
        self.model = model
        self.assert_model()
        # Während des Trainings werden immmer Transitions zu dem Replay Buffer 
        # hinzugefügt. 
        # Das Neuronale Netz wird ausschließlich aus zufälligen Transitions aus 
        # dem Replay Buffer trainiert.
        # Das hilft um das Neuronale Netz Daten Effizient zu machen und das 
        # Neuronale Netzwerk an alte Transitions zu erinnern.
        self.memory = deque(maxlen=memory_size)
        # Die discount rate besimmt wieviel vorherige Erfahrungen gewichtet 
        # werden sollen.
        # 1 = max
        # 0 = min
        self.gamma = gamma
        # Die exploration rate gibt an wie random der Agent sich verhalten soll.
        # 1 = max
        # 0 = min
        # Die exploration rate wird im Laufe des Trainings verringert, damit der 
        # Agent mehr versucht seine momentane Technik zu verbessern, anstatt 
        # neue Techniken zu suchen.
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def assert_model(self):
        dummy_input = np.ones([1, *self.obs_size])
        assert self.model.predict(dummy_input).shape == (1, self.act_size)

    def add_transition(self, state, action, reward, next_state, done):
        # TODO: Make memory efficient
        assert isinstance(state, np.ndarray)
        assert isinstance(action, (int, float))
        assert isinstance(reward, (int, float))
        assert isinstance(next_state, np.ndarray)
        assert isinstance(done, bool)
        # Speichert eine Transition in dem Replay Buffer
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) > 1
        assert state.shape[0] == 1
        # Epsilon-Greedy
        # Mit einer Wahrscheinlichkeit von self.epsilon, wähle eine zufällige 
        # Aktion aus.
        if np.random.rand() <= self.epsilon:
            return np.random.choice(list(range(self.act_size)))
        # Mit einer Wahrscheinlichkeit von 1 - self.epsilon, wähle eine Aktion 
        # basierend der Policy.
        # Bestimme alle q-values der Aktionen im momentanen state.
        act_values = self.model.predict(state)
        # Wähle die Aktion mit dem höchsten q-value
        return np.argmax(act_values[0])  # returns action

    def train_on_batch(self, batch_size):
        assert isinstance(batch_size, int)
        # Sample ein batch aus dem Replay Buffer
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = np.atleast_2d(state)
            next_state = np.atleast_2d(next_state)
            # Berechne das eigentliche q-value der Aktion mithilfe der 
            # Bellman Gleichung.
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            # kleiner Trick um die keras fit api zu nutzen
            target_q = self.model.predict(state)
            target_q[0][action] = target
            # Trainiere das Neuronale Netz mit Hilfe des target q-values.
            self.model.fit(state, target_q, epochs=1, verbose=0)
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        assert isinstance(path, str)
        self.model.load_weights(path)

    def save(self, path):
        assert isinstance(path, str)
        self.model.save_weights(path)
        
if __name__ == "__main__":
    # Beispiel
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    
    observation_size = 7
    action_size = 5
    memory_size = 1000
    learning_rate = 0.01
    num_games = 10
    batch_size = 5
    
    model = Sequential()
    model.add(Dense(24, input_shape=[observation_size,], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    
    dqn = DQN(model, [observation_size], action_size, memory_size)
    
    print(dqn.get_action(np.ones((1, observation_size))))
    
    for _ in range(num_games):
        dqn.add_transition(
            np.ones(observation_size), 
            np.random.randint(action_size), 
            np.random.randint(-100, 100),
            np.ones(observation_size),
            bool(np.random.randint(2))
        )
        
    dqn.train_on_batch(batch_size)
