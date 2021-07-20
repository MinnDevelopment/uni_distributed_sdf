from typing import List

import numpy as np
from numpy.linalg import pinv as inv
from project.assignment_01 import NaiveFusion
from project.sensor import ProcessingNodeSensor, SensorModel

# Lecture 8 Slide 36

class DistributedSensorNode(ProcessingNodeSensor):
    def __init__(self, sensor, models: List[SensorModel]):
        super().__init__(sensor)
        self.H = sensor.H
        self.R = [model.R for model in models]
    
    def process(self, t):
        # TODO: Globalization step!
        pass

class DistributedFusion(NaiveFusion):
    def __init__(self, sensors):
        super().__init__(sensors)
