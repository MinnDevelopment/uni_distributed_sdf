from numpy.lib.function_base import cov
from project.information import InformationFilter
from project import ProcessingNodeSensor, CentralProcessingNode
import numpy as np
from numpy.linalg import pinv as inv

class TrackletSensorUnit(ProcessingNodeSensor):
    def __init__(self, sensor):
        super().__init__(sensor)
    
    # Used to gain the prior by applying the transition model to the posterior
    def F(self, delta):
        return self.filter.F(delta)
    
    def Q(self, delta):
        return self.filter.Q(delta)

class TrackletFusion(CentralProcessingNode):
    def __init__(self, sensors):
        super().__init__(sensors)
        self.nodes = [TrackletSensorUnit(sensor) for sensor in sensors]
        self.previous = [(node.filter.state, node.filter.covariance) for node in self.nodes] # Used to get prior
        self.filter = InformationFilter()
    
    def tostate(self, y, Y):
        # Convert from information space to state space
        covariance = inv(Y)
        state = covariance @ y
        return state, covariance
    
    def process(self, t):
        # Take measurements from each sensor
        i = np.zeros((4, 1))
        I = np.zeros((4, 4))
        posterior = []
        for node, previous in zip(self.nodes, self.previous):
            delta = t - node.filter.time
            F, Q = node.F(delta), node.Q(delta)

            # Receive posterior for current time step
            post_x, post_P = node.process(t)
            posterior.append((post_x, post_P))

            # Obtain prior through transition model (F, Q)
            prior_P = F @ previous[1] @ F.T + Q
            prior_x = F @ previous[0]
            
            # Change to information space
            post_P = inv(post_P)
            prior_P = inv(prior_P)

            I += post_P - prior_P
            i += post_P @ post_x - prior_P @ prior_x

        # Save our new posteriors for the next step
        self.previous = posterior
        # This filters in information space (inverse covariance)
        y, Y = self.filter(t, i, I)
        # Convert to state space
        return self.tostate(y, Y)
    
    def predict(self, t): # This doesn't work well at all
        y, Y = self.filter.predict(t)
        return self.tostate(y, Y)
