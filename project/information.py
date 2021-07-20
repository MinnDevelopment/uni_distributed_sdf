import numpy as np
from numpy.lib.function_base import cov
from numpy.linalg import pinv as inv


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
        self.information_matrix = 0 * np.eye(4)
        self.time = time

    def filter(self, time, i, I):
        i = i.reshape((4, 1)) # Just to make sure the sum works properly
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
        F = inv(self.F(delta))
        Q = inv(self.Q(delta))

        M = F.T @ self.information_matrix @ F
        C = M @ inv(M + Q)
        L = np.eye(4) - C
        
        mat = L @ M @ L.T + C @ Q @ C.T
        vec = L @ F.T @ self.information_vector
        return vec, mat