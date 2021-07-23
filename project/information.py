import numpy as np
from numpy.linalg import inv as inv

# This is mostly derived from wikipedia: https://en.wikipedia.org/wiki/Kalman_filter#Information_filter

class InformationFilter:
    __slots__ = 'information_vector', 'information_matrix', 'time'

    def __init__(self):
        self.init(0)

    def F(self, delta):
        # v = a * t + v0
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def Q(self, delta):
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        # Choose sigma from [4.5, 9]
        return 4.5 * error

    def init(self, time):
        self.information_vector = np.zeros((4, 1))
        self.information_matrix = np.eye(4)
        self.time = time

    def filter(self, time, i, I):
        # Compute prior
        prior_y, prior_Y = self.predict(time)

        # Update internal state with new information
        self.information_matrix = prior_Y + I
        self.information_vector = prior_y + i
        self.time = time
        return self.information_vector, self.information_matrix

    def __call__(self, time, measurement, R):
        return self.filter(time, measurement, R)

    def predict(self, time):
        delta = time - self.time
        if delta == 0:
            return self.information_vector, self.information_matrix
        F = self.F(delta)
        Q = self.Q(delta)

        # Apply dynamics model in state space
        P = inv(self.information_matrix)
        x = F @ P @ self.information_vector
        P = F @ inv(self.information_matrix) @ F.T + Q

        # Convert back to information space
        Y = inv(P)
        y = inv(P) @ x
        return y, Y
