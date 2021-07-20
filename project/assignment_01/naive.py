from project.sensor import LinearSensor
from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project import CentralProcessingNode, LinearSensor, ProcessingNodeSensor


class NaiveFusion(CentralProcessingNode):
    def __init__(self, sensors: List[LinearSensor]):
        super().__init__(sensors)
        self.nodes = [ProcessingNodeSensor(sensor) for sensor in sensors]

    def process(self, t):
        x = np.zeros((4, 1))
        P = np.zeros((4, 4))

        # Take measurements from each sensor
        outputs = [s.process(t) for s in self.nodes]
        self.measurements = [o[0][:2].flatten() for o in outputs]

        # Perform convex combination
        for x_i, P_i in outputs:
            # Compute the sum for the sensors with weights
            P_i = inv(P_i)
            # print(P_i.shape, x_i.shape, x.shape)
            P += P_i
            x += P_i @ x_i

        P = inv(P)
        x = P @ x

        return x, P
