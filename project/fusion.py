from typing import List

from project.sensor import LinearSensor


class CentralProcessingNode:
    def __init__(self, sensors: List[LinearSensor]):
        self.sensors = sensors
        self.nodes = [] # Initialized by implementation

    def process(self, t):
        pass # Implemented by Federated and Naive Fusion centers

    def predict(self, t):
        pass
