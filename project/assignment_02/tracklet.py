import numpy as np
from numpy.linalg import inv as inv
from project import CentralProcessingNode, ProcessingNodeSensor
from project.information import InformationFilter
from project.kalman import KalmanFilter


class TrackletFusion(CentralProcessingNode):
    def __init__(self, sensors):
        super().__init__(sensors)
        self.nodes = [ProcessingNodeSensor(sensor) for sensor in sensors]
        self.previous = [(node.filter.state, node.filter.covariance) for node in self.nodes] # Used to get prior
        self.filter = InformationFilter()
    
    def F(self, delta):
        return KalmanFilter(self.nodes[0].model).F(delta)

    def Q(self, delta):
        return KalmanFilter(self.nodes[0].model).Q(delta)
    
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
        delta = t - self.filter.time
        F, Q = self.F(delta), self.Q(delta)
        for node, previous in zip(self.nodes, self.previous):
            # Receive posterior for current time step
            post_x, post_P = node.process(t)
            posterior.append((post_x, post_P))

            # Obtain prior through transition model (F, Q)
            prior_P = F @ previous[1] @ F.T + Q # P^s(k-1|k-1) -> P^s(k|k-1)
            prior_x = F @ previous[0] # x^s(k-1|k-1) -> x^s(k|k-1)
            
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
