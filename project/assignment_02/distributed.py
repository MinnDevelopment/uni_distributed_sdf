from numpy.lib.function_base import cov
from project.assignment_02.federated import RelaxedEvolutionKalmanFilter
from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project.assignment_01 import NaiveFusion
from project.sensor import LinearSensor, ProcessingNodeSensor, SensorModel

# Lecture 8 Slide 36

class DistributedKalmanFilter(RelaxedEvolutionKalmanFilter):
    def __init__(self, H, S):
        super().__init__(H, S)

    def globalization(self, covariances):
        global_P = self.S * inv(np.sum([inv(P_s) for P_s in covariances], axis=0))
        x, P = self.state, self.covariance
        self.state = global_P @ inv(P) @ x
        self.covariance = global_P

class DistributedSensorNode(ProcessingNodeSensor):
    def __init__(self, sensor):
        super().__init__(sensor)
        self.H = sensor.H
        self.covariances = []

    def init(self, nodes):
        self.nodes = nodes
        self.filter = DistributedKalmanFilter(self.H, len(nodes))
    
    def receive(self, P):
        # Receives the covariance from another sensor
        self.covariances.append(P)
        # Check if we collected all covariances now
        if len(self.covariances) == len(self.nodes):
            # Prepare the filter for the next timestep by globalizing the state
            self.filter.globalization(self.covariances)
            # Clear covariances since we used them up, collect new ones in next timestep
            self.covariances.clear()
    
    def share(self, P):
        # Shares the current covariance with other sensors in the network
        for node in self.nodes:
            node.receive(P)

    def process(self, t):
        z, R = self.sensor.measure(t)
        x, P = self.filter(t, z, R)
        self.measurement = z
        self.share(P)
        return x, P

class DistributedFusion(NaiveFusion):
    def __init__(self, sensors):
        super().__init__(sensors)
        self.nodes = [DistributedSensorNode(sensor) for sensor in self.sensors]
        for node in self.nodes:
            node.init(self.nodes)
