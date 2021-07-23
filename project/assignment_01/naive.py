from typing import List

import numpy as np
from numpy.linalg import inv as inv
from project import CentralProcessingNode, ProcessingNodeSensor
from project.sensor import LinearSensor


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
        # Take measurements from each sensor
        outputs = [s.process(t) for s in self.nodes]
        return self.combine(outputs)

    def predict(self, t):
        # Predict position from each sensor
        outputs = [s.predict(t) for s in self.nodes]
        return self.combine(outputs)
