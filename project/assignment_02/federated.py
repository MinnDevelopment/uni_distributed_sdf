from typing import List

import numpy as np
from numpy.linalg import inv as inv
from project import KalmanFilter, LinearSensor, ProcessingNodeSensor
from project.assignment_01 import NaiveFusion


class RelaxedEvolutionKalmanFilter(KalmanFilter):
    def __init__(self, model, S):
        super().__init__(model)
        self.S = S # Used for relaxed evolution model (Lecture 7 Slide 33)
    
    def Q(self, delta):
        return self.S * KalmanFilter.Q(self, delta)


class FederatedFusion(NaiveFusion):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ProcessingNodeSensor(sensor) for sensor in sensors]

        # Use the relaxed evolution model based filter implementation instead of the default
        for node in self.nodes:
            node.filter = RelaxedEvolutionKalmanFilter(node.model, len(self.nodes))
