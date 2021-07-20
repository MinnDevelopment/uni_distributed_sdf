from numpy.lib.function_base import cov
from project.information import InformationFilter
from project import ProcessingNodeSensor, CentralProcessingNode
import numpy as np
from numpy.linalg import pinv as inv

class TrackletSensorUnit(ProcessingNodeSensor):
    def __init__(self, sensor):
        super().__init__(sensor)

    def process(self, t):
        prior_x, prior_P = self.filter.predict(t)
        post_x, post_P = ProcessingNodeSensor.process(self, t)

        prior_P, post_P = inv(prior_P), inv(post_P)
        I = post_P - prior_P
        i = post_P @ post_x - prior_P @ prior_x
        return i, I

        # z, R = self.sensor.measure(t)
        # # Store measurement for presentation
        # self.measurement = np.array(z)
        # # Convert measurement to information space
        # H = self.sensor.H
        # R = inv(self.sensor.R)
        # I = H.T @ R @ H
        # i = H.T @ R @ z
        # return i, I

class TrackletFusion(CentralProcessingNode):
    def __init__(self, sensors):
        super().__init__(sensors)
        self.nodes = [TrackletSensorUnit(sensor) for sensor in sensors]
        self.filter = InformationFilter()
    
    def tostate(self, y, Y):
        # Convert from information space to state space
        covariance = inv(Y)
        state = covariance @ y
        return state, covariance
    
    def process(self, t):
        # Take measurements from each sensor
        outputs = [s.process(t) for s in self.nodes]
        # Fusion is simply adding up all the information
        i = np.sum([o[0] for o in outputs], axis=0)
        I = np.sum([o[1] for o in outputs], axis=0)
        # This filters in information space (inverse covariance)
        y, Y = self.filter(t, i, I)
        # Convert to state space
        return self.tostate(y, Y)
    
    def predict(self, t): # This doesn't work well at all
        y, Y = self.filter.predict(t)
        return self.tostate(y, Y)
