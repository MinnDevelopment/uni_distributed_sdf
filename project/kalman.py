import numpy as np
from numpy.linalg import pinv as inv

class KalmanFilter:
    __slots__ = 'state', 'covariance', 'time', 'H'

    def __init__(self, H):
        self.state = np.zeros((4, 1))
        self.covariance = 500 * np.eye(4)
        self.time = 0
        self.H = H

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
        self.state = np.zeros((4, 1))
        self.covariance = 500 * np.eye(4)
        self.time = time

    def filter(self, time, measurement, R):
        delta = time - self.time
        F = self.F(delta)
        H = self.H
        # Predict location
        estimated_state = F @ self.state
        estimate_covariance = F @ self.covariance @ F.T + self.Q(delta)
        # Calculate innovation from measurement
        innovation = measurement - H @ estimated_state
        innovation_error = H @ estimate_covariance @ H.T + R
        gain = estimate_covariance @ H.T @ inv(innovation_error)
        # Update internal state with new information
        self.state = estimated_state + gain @ innovation
        self.covariance = estimate_covariance - gain @ innovation_error @ gain.T
        self.time = time
        return self.state, self.covariance

    def __call__(self, time, measurement, R):
        return self.filter(time, measurement, R)

    def predict(self, time):
        delta = time - self.time
        if delta == 0:
            return self.state, self.covariance
        F = self.F(delta)
        state = F @ self.state
        covariance = F @ self.covariance @ F.T + self.Q(delta)
        return state, covariance