from project.sensor import LinearSensor
from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project import CentralProcessingNode, LinearSensor, ProcessingNodeSensor


class NaiveFusion(CentralProcessingNode):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ProcessingNodeSensor(sensor) for sensor in sensors]

    def combine(self, outputs):
        x = np.zeros((4, 1))
        P = np.zeros((4, 4))

        # Perform convex combination
        for x_i, P_i in outputs:
            # Compute the sum for the sensors with weights
            P_i = inv(P_i)
            P += P_i
            x += P_i @ x_i

        P = inv(P)
        x = P @ x

        return x, P


    def process(self, t):
        x = np.zeros((4, 1))
        P = np.zeros((4, 4))

        # Take measurements from each sensor
        outputs = [s.process(t) for s in self.nodes]
        self.measurements = [o[0][:2].flatten() for o in outputs]
        return self.combine(outputs)

    def predict(self, t):
        x = np.zeros((4, 1))
        P = np.zeros((4, 4))

        # Predict position from each sensor
        outputs = [s.predict(t) for s in self.nodes]
        return self.combine(outputs)