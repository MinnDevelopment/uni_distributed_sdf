from project.sensor import LinearSensor
from typing import List

import numpy as np

from project import ProcessingNodeSensor

class CentralProcessingNode:
    def __init__(self, sensors: List[LinearSensor]):
        self.sensors = sensors
        self.nodes = [] # Initialized by implementation
        self.measurements = [] # Store last set of measurements for presentation reasons

    def process(self, t):
        pass # Implemented by Federated and Naive Fusion centers
