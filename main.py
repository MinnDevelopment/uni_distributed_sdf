import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
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
            return np.vstack((x, y))
    def derived(t):
        a = v/r
        while True:
            x = v * cos(t * a / 2) / 2
            y = v * cos(t * a) / 2
            return np.vstack((x, y))
    return wrapper, derived

    
# Simulation framework is simply a generator which gives cartesian ground truth positions in second intervals
def sine(v=25, r=1000):
    """
    We fly an infinity shaped track with the cross being the start position
    """
    def wrapper(t):
        a = v/r
        while True:
            y = r * sin(t * a)
            return np.vstack((t, y))
    def derived(t):
        a = v/r
        while True:
            y = v * cos(t * a)
            return np.vstack((t, y))
    return wrapper, derived

# The 4 fusion methods we should implement
methods = {
    # Fusion center receives raw measurements (no T2TF)
    'central': project.CentralFusion,
    # Convex combination / Least-Squares / Naive Fusion
    'naive': project.assignment_01.NaiveFusion,
    # Information Filtering
    'tracklet': project.assignment_02.TrackletFusion,
    # Relaxed Evolution Model
    'federated': project.assignment_02.FederatedFusion,
    # Relaxed Evolution Model + Globalization Step
    'distributed': project.assignment_02.DistributedFusion
}

# Define our simulation parameters

T = 495 # Number of steps
stepsize = 2
S = 4 # Number of sensors
sigma = 10000 # Sensor covariance factor
track, velocity = cross_loop()

random_seed = (T << 16) + (stepsize << 8) + S + sigma

print("Simulation Parameters")
print("Steps", T)
print("Step Size", stepsize)
print("Sensors", S)
print("Covariance Factor", sigma)
print("Seed", random_seed)
print("====================================")

def simulate(method):
    np.random.seed(random_seed) # Makes sure the measurements are identical for all episodes

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
    nees = []

    show_filtered_measurements = False # Whether to show filtered measurements from the processing nodes, or the raw sensor data

    # Simulate the entire sequence using 1-second timestamps
    for t in range(0, T, stepsize):
        # This does the measurements from each sensor and fuses them with the respective method
        x, P = fusion.process(t)
        gt = track(t)
        vel_gt = velocity(t)

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
        gt_x = np.vstack((gt, vel_gt))
        nees.extend(((gt_x - x).T @ inv(P) @ (gt_x - x)).flatten()) # TODO: Compute actual gt velocity
        x = H @ x
        error = np.sqrt(np.mean((x - gt)**2)) # RMSE
        errors.append(error)
        ground_truth.append(gt.flatten())
        predicted.append(x.flatten())
        std.append(np.sqrt(np.trace(inv(P))))

        # Do a prediction
        for i in range(1, stepsize):
            x, P = fusion.predict(t+i)
            gt = track(t+i)
            vel_gt = velocity(t+i)
            gt_x = np.vstack((gt, vel_gt))
            nees.extend(((gt_x - x).T @ inv(P) @ (gt_x - x)).flatten()) # TODO: Compute actual gt velocity
            x = H @ x
            error = np.sqrt(np.mean((x - gt)**2)) # RMSE
            errors.append(error)
            ground_truth.append(gt.flatten())
            predicted.append(x.flatten())
            std.append(np.sqrt(np.trace(inv(P))))

        if __debug__: # This means you can disable logs with `python3 -O <file>`
            print("Step", t)
            print("Truth:", gt.T)
            print("Predicted:", x.T)
            print("Covariance:", P) # You can observe this converges to a fixed covariance matrix over time
            print("Error:", error)
            print("=======")
    
    return errors, measurements, ground_truth, predicted, std, nees

stds = dict()
rmse = dict()
nees = dict()

# np.seterr(all="raise")
fig, axes = plt.subplots(2, 3, figsize=(32, 8))
for ax, method in zip(axes.flatten(), methods.keys()):
    print("Running Method:", method.capitalize())
    print("====================================")
    errors, measurements, ground_truth, predicted, std, nee = simulate(method)

    stds[method] = std
    rmse[method] = errors
    nees[method] = nee

    # Plot the results for this trajectory simulation

    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    measurements = np.array(measurements)

    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'k', label='Ground Truth', alpha=0.5)
    ax.plot(predicted[:, 0], predicted[:, 1], 'r', label='Prediction')
    ax.scatter(measurements[:, 0], measurements[:, 1], 0.1, c='g', label='Measurements')
    ax.legend()
    ax.set_title(f'{method.capitalize()} Fusion') # with {S} Sensors and Covariance {sigma}
ax = axes.flatten()[-1]
ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'k')
ax.set_title('Ground Truth')
fig.savefig(f'trajectory-S{S}-COV{sigma}.png')
plt.close()

# Joint plot for all methods

plt.figure(figsize=(16, 4))
width = 0.75 * len(rmse.items()) + 0.75
baseline = np.array(rmse['central'])
for method, error in rmse.items():
    plt.plot(range(len(error)), np.array(error) - baseline, linewidth=width, label=method)
    width -= 0.75
plt.title('Root Mean Squared Error')
plt.legend()
plt.savefig(f'rmse.png')
plt.close()

plt.figure(figsize=(16, 4))
width = 0.75 * len(nees.items()) + 0.75
baseline = np.array(nees['central'])
for method, error in nees.items():
    plt.plot(range(len(error)), np.array(error) - baseline, linewidth=width, label=method)
    width -= 0.75
plt.title('Normalized Estimation Error Squared')
plt.legend()
plt.savefig(f'nees.png')
plt.close()

width = 0.75 * len(stds.items()) + 0.75
baseline = np.array(stds['central'])
for method, std in stds.items():
    plt.plot(range(len(std)), np.array(std) - baseline, linewidth=width, label=method)
    width -= 0.75
plt.title('Standard Deviation')
plt.legend()
plt.savefig(f'std.png')
plt.close()