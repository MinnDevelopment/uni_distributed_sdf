from typing import List
import numpy as np

from project import KalmanFilter

class SensorModel:
    def __init__(self, R, H=np.hstack([np.eye(2), np.zeros((2, 2))])):
        self.R = R
        self.H = H

class LinearSensor:
    def __init__(self, gen, cov=100):
        self.model = SensorModel(np.diag([cov, cov]))
        self.gen = gen # Generator for the measurements

    @property
    def H(self):
        return self.model.H

    @property
    def R(self):
        return self.model.R
    
    def noise(self):
        return np.random.multivariate_normal([0, 0], self.R).reshape((2, 1))
    
    def measure(self, t):
        return self.noise() + self.gen(t), self.R

class ProcessingNodeSensor:
    def __init__(self, sensor: LinearSensor):
        self.sensor = sensor
        self.measurement = np.zeros((1, 2))
        self.filter = KalmanFilter(sensor.model.H)

    def process(self, t):
        # Take a sensor measurement
        z, R = self.sensor.measure(t)
        # Store unfused measurement for presentation
        self.measurement = np.array(z)
        # Do local filtering on the sensor measurement
        x, P = self.filter(t, z, R)

        # Return the filtered state and covariance
        return x, P
    
    def predict(self, t):
        return self.filter.predict(t)
