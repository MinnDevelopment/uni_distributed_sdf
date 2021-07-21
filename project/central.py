from project.kalman import KalmanFilter
import numpy as np
from numpy.linalg import pinv as inv
from typing import List

from project.sensor import ProcessingNodeSensor, LinearSensor
from project.fusion import CentralProcessingNode

# This is only used for evaluation

class ForwardingNode(ProcessingNodeSensor):
    def __init__(self, sensor: LinearSensor):
        super().__init__(sensor)

    def process(self, t):
        z, R = self.sensor.measure(t)
        self.measurement = np.array(z)
        return z, R

class CentralFusion(CentralProcessingNode):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ForwardingNode(sensor) for sensor in sensors]
        self.filter = KalmanFilter(sensors[0].H)
    
    def process(self, t):
        x = np.zeros((2, 1))
        P = np.zeros((2, 2))

        for z, R in [s.process(t) for s in self.nodes]:
            R = inv(R)
            x += R @ z
            P += R
        P = inv(P)
        x = P @ x
        return self.filter(t, x, P)
    
    def predict(self, t):
        return self.filter.predict(t)