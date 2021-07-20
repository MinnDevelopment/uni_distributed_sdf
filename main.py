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
    def wrapper(t):
        a = v/r
        while True:
            x = r * sin(t * a / 2)
            y = r * sin(t * a) / 2
            return x, y
    return wrapper

    
# Simulation framework is simply a generator which gives cartesian ground truth positions in second intervals
def sine(v=25, r=1000):
    """
    We fly an infinity shaped track with the cross being the start position
    """
    def wrapper(t):
        a = v/r
        while True:
            y = r * sin(t * a)
            return t, y
    return wrapper

# The 4 fusion methods we should implement
methods = {
    # Convex combination / Least-Squares / Naive Fusion
    'naive': project.assignment_01.NaiveFusion,
    'tracklet': project.assignment_02.TrackletFusion,
    'federated': project.assignment_02.FederatedFusion,
    # 'distributed': project.assignment_02.DistributedFusion
}

# Define our simulation parameters

T = 495 # Number of steps
stepsize = 1
S = 4 # Number of sensors
sigma = 100 # Sensor covariance factor
track = sine()

print("Simulation Parameters")
print("Steps", T)
print("Step Size", stepsize)
print("Sensors", S)
print("Covariance Factor", sigma)
print("====================================")

def simulate(method):
    np.random.seed(4)
    # Generate S many linear sensors with the same covariance
    sensors = [project.LinearSensor(track, sigma) for s in range(S)]
    # Initialize the fusion center with the sensor nodes
    fusion = methods[method](sensors)
    # Retrieve processing nodes from the fusion center
    nodes = fusion.nodes
    # The sensor model, used to extract the estimated position
    H = np.block([np.eye(2), np.zeros((2, 2))])

    errors = []
    ground_truth = []
    predicted = []
    measurements = []
    std = []

    show_filtered_measurements = False # Whether to show filtered measurements from the processing nodes, or the raw sensor data

    # Simulate the entire sequence using 1-second timestamps
    for t in range(0, T, stepsize):
        # This does the measurements from each sensor and fuses them with the respective method
        x, P = fusion.process(t)
        gt = track(t)

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
        error = np.sqrt(np.mean((x - np.array(gt))**2)) # RMSE
        errors.append(error)
        ground_truth.append(gt)
        predicted.append(x)
        std.append(np.sqrt(np.trace(inv(P))))
        if __debug__: # This means you can disable logs with `python3 -O <file>`
            print("Step", t)
            print("Truth:", gt)
            print("Predicted:", x)
            print("Covariance:", P) # You can observe this converges to a fixed covariance matrix over time
            print("Error:", error)
            print("=======")
    
    return errors, measurements, ground_truth, predicted, std

stds = dict()
rmse = dict()

for method in methods.keys():
    print("Running Method", method.capitalize())
    print("====================================")
    errors, measurements, ground_truth, predicted, std = simulate(method)

    stds[method] = std
    rmse[method] = errors

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

plt.figure(figsize=(16, 4))
for method, error in rmse.items():
    plt.plot(range(len(error)), error, linewidth=0.5, label=method)
plt.title('Root Mean Squared Error')
plt.legend()
plt.savefig(f'rmse.png')
plt.close()

for method, std in stds.items():
    plt.plot(range(len(std)), std, label=method)
plt.title('Standard Deviation')
plt.legend()
plt.savefig(f'std.png')
plt.close()