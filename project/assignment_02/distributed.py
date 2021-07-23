from project.kalman import KalmanFilter
from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project.assignment_01 import NaiveFusion
from project.assignment_02.federated import RelaxedEvolutionKalmanFilter
from project.sensor import ProcessingNodeSensor, SensorModel

# Lecture 8 Slide 36

class DistributedKalmanFilter(RelaxedEvolutionKalmanFilter):
    def __init__(self, model, sensor_models: List[SensorModel]):
        super().__init__(model, len(sensor_models))
        self.Rs = [m.R for m in sensor_models] # Assuming H is the same everywhere for simplicity
        self.previous_covariances = [R.trace() * np.eye(4) for R in self.Rs]
        self.globalization(0) # This is technically not needed but 0|0 is a posterior we can use in prediction for k=1

    def compute_covariance(self, delta, P, R):
        F = self.F(delta)
        Q = self.Q(delta)
        H = self.H

        # Same predict/filter cycle as usual, except we don't care about the state for each local track
        prior = F @ P @ F.T + Q
        innovation_error = H @ prior @ H.T + R
        gain = prior @ H.T @ inv(innovation_error)
        return prior - gain @ innovation_error @ gain.T

    def globalization(self, delta):
        # Compute covariances for each sensor with the known dynamics and sensor models
        self.previous_covariances = covariances = [self.compute_covariance(delta, P, R) for P, R in zip(self.previous_covariances, self.Rs)]
        # Compute the GLOBAL covariance through a convex combination
        global_P = self.S * inv(np.sum([inv(P_s) for P_s in covariances], axis=0))
        x, P = self.state, self.covariance
        # Update the local state and covariance by using the global covariance
        self.state = global_P @ inv(P) @ x
        self.covariance = global_P
    
    def filter(self, time, measurement):
        delta = time - self.time
        # Standard relaxed evolution filtering like federated
        x, P = RelaxedEvolutionKalmanFilter.filter(self, time, measurement)
        # Globalization step to get global covariance
        self.globalization(delta)
        return self.state, self.covariance

class DistributedSensorNode(ProcessingNodeSensor):
    def __init__(self, sensor):
        super().__init__(sensor)

    def init(self, nodes):
        self.filter = DistributedKalmanFilter(self.model, [n.model for n in nodes])

class DistributedFusion(NaiveFusion):
    def __init__(self, sensors):
        super().__init__(sensors)
        self.nodes = [DistributedSensorNode(sensor) for sensor in self.sensors]
        for node in self.nodes:
            node.init(self.nodes)
            