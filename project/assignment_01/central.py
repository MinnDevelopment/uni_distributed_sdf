from typing import List

import numpy as np
from numpy.linalg import inv as inv
from project.fusion import CentralProcessingNode
from project.kalman import KalmanFilter
from project.sensor import LinearSensor, ProcessingNodeSensor, SensorModel

# This is only used for evaluation

class ForwardingNode(ProcessingNodeSensor):
    def __init__(self, sensor: LinearSensor):
        super().__init__(sensor)

    def process(self, t):
        z = self.sensor.measure(t)
        self.measurement = np.array(z)
        return z

class CentralFusion(CentralProcessingNode):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ForwardingNode(sensor) for sensor in sensors]

        R = inv(np.sum([inv(sensor.R) for sensor in sensors], axis=0))
        model = SensorModel(R)
        self.filter = KalmanFilter(model)
    
    def process(self, t):
        x = np.zeros((2, 1))

        for z, R in [(s.process(t), s.model.R) for s in self.nodes]:
            R = inv(R)
            x += R @ z
        x = self.filter.R @ x
        return self.filter(t, x)
    
    def predict(self, t):
        return self.filter.predict(t)
