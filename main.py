import matplotlib.pyplot as plt
import numpy as np
from numpy import sin
from numpy.linalg import pinv as inv

import project


# Simulation framework is simply a generator which gives cartesian ground truth positions in second intervals
def cross_loop(v=25, r=1000):
    """
    We fly an infinity shaped track with the cross being the start position
    """
    a = v/r
    t = 0
    while True:
        x = r * sin(t * a / 2)
        y = r * sin(t * a) / 2
        yield x, y
        t += 1

# The 4 fusion methods we should implement
methods = {
    # Convex combination / Least-Squares / Naive Fusion
    'naive': project.assignment_01.NaiveFusion,
    # 'tracklet': project.assignment_02.TrackletFusion,
    # 'federated': project.assignment_02.FederatedFusion,
    # 'distributed': project.assignment_02.DistributedFusion
}

method = 'naive'

S = 10 # The number of sensors
T = 495 # The number of timesteps (seconds)

sigma = 5000 # The simulated error covariance value for the sensors

H = np.block([np.eye(2), np.zeros((2, 2))]) # The sensor model

def simulate(method, S=10, sigma=5000, T=495):
    # Generate S many linear sensors with the same covariance
    sensors = [project.LinearSensor(cross_loop(), sigma) for s in range(S)]
    # Create a processing node for local and distributed fusion
    nodes = [project.ProcessingNodeSensor(sensor, [sensor.model]) for sensor in sensors]
    # Initialize the fusion center with the sensor nodes
    fusion = methods[method](nodes)

    errors = []
    ground_truth = []
    predicted = []
    measurements = []
    std = []

    show_filtered_measurements = False # Whether to show filtered measurements from the processing nodes, or the raw sensor data

    # Simulate the entire sequence using 1-second timestamps
    for t, gt in zip(range(T), cross_loop()):
        # Perform fusion at timestep t
        # This does the measurements from each sensor and fuses them with the respective method
        x, P = fusion.process(t)

        std.append(np.sqrt(np.trace(inv(P))))

        # Store measurements for presentation

        if show_filtered_measurements:
            # Note that these measurements are already locally filtered by the processing nodes!
            Z = fusion.measurements
        else:
            # These are the raw measurements from the sensor (with noise included)
            # Unfiltered measurements will be a lot less accurate than the measurements provided by the local filtering
            Z = [node.measurement for node in nodes]
        measurements.extend(Z)

        # Use the sensor model to get the predicted position after fusion
        x = (H @ x).flatten()
        print("Step", t)
        print("Truth:", gt)
        print("Predicted:", x)
        print("Covariance:", P) # You can observe this converges to a fixed covariance matrix over time
        error = np.sqrt(np.mean((x - np.array(gt))**2)) # RMSE
        errors.append(error)
        ground_truth.append(gt)
        predicted.append(x)
        print("Error:", error)
        print("=======")
    
    return errors, measurements, ground_truth, predicted, std


errors, measurements, ground_truth, predicted, std = simulate('naive')

# Plot the results for this trajectory simulation

ground_truth = np.array(ground_truth)
predicted = np.array(predicted)
measurements = np.array(measurements)

plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'k', label='Ground Truth', alpha=0.5)
plt.plot(predicted[:, 0], predicted[:, 1], 'r', label='Prediction')
plt.scatter(measurements[:, 0], measurements[:, 1], 0.1, c='g', label='Measurements')
plt.legend()
plt.title(f'{method.capitalize()} Fusion with {S} Sensors and Covariance {sigma}')
plt.savefig(f'{method}-trajectory-S{S}-COV{sigma}.png')
plt.close()

# Joined plot for all methods

plt.plot(range(T), errors)
plt.title('Root Mean Squared Error')
plt.savefig(f'{method}-error.png')
plt.close()