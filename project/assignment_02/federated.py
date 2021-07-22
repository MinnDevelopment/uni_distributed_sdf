from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project import KalmanFilter, LinearSensor, ProcessingNodeSensor
from project.assignment_01 import NaiveFusion


class RelaxedEvolutionKalmanFilter(KalmanFilter):
    def __init__(self, H, S):
        super().__init__(H)
        self.S = S # Used for relaxed evolution model (Lecture 7 Slide 33)
    
    def Q(self, delta):
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        # Choose sigma from [4.5, 9]
        return self.S * 4.5 * error

    def filter(self, time, measurement, R):
        measurement = measurement.reshape((2, 1))
        H = self.H
        # Predict location
        estimated_state, estimated_covariance = self.predict(time)
        # Calculate innovation from measurement
        innovation_error = H @ estimated_covariance @ H.T + R
        gain = estimated_covariance @ H.T @ inv(innovation_error)
        # Update internal state with new information
        self.state = estimated_state + gain @ (measurement - H @ estimated_state)
        self.covariance = estimated_covariance - gain @ innovation_error @ gain.T
        self.time = time
        return self.state, self.covariance

class FederatedFusion(NaiveFusion):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ProcessingNodeSensor(sensor) for sensor in sensors]

        # Use the relaxed evolution model based filter implementation instead of the default
        for node in self.nodes:
            node.filter = RelaxedEvolutionKalmanFilter(node.sensor.H, len(self.nodes))
