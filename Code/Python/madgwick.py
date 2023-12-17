from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()  # Enable interactive mode

class ExponentialMovingAverageFilter:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.avg = None

    def update(self, new_data):
        if self.avg is None:
            self.avg = new_data
        else:
            self.avg = self.alpha * new_data + (1 - self.alpha) * self.avg
        return self.avg

class MovingAverageFilter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.data = np.zeros((window_size, 3))

    def update(self, new_data):
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1, :] = new_data
        return np.mean(self.data, axis=0)


# Function to create a rectangular prism (representing the pen)
def create_rectangular_prism(center, length, width, height):
    # Define the vertices of the rectangular prism (8 vertices)
    x = length / 2.0
    y = width / 2.0
    z = height / 2.0
    vertices = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                         [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]) + center
    # Define the sides (faces) of the prism using the vertices
    faces = [[vertices[i] for i in [0, 1, 2, 3]], [vertices[i] for i in [4, 5, 6, 7]], 
             [vertices[i] for i in [0, 3, 7, 4]], [vertices[i] for i in [1, 2, 6, 5]], 
             [vertices[i] for i in [0, 1, 5, 4]], [vertices[i] for i in [2, 3, 7, 6]]]
    return faces

# Set up the matplotlib figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a Madgwick filter instance
madgwick = Madgwick()
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion

# # Initialize exponential moving average filters for each sensor
# acc_filter = ExponentialMovingAverageFilter()
# gyro_filter = ExponentialMovingAverageFilter()
# mag_filter = ExponentialMovingAverageFilter()

# Initialize moving average filters for each sensor
acc_filter = MovingAverageFilter()
gyro_filter = MovingAverageFilter()
mag_filter = MovingAverageFilter()

# Initial prism (pen) configuration
pen_length, pen_width, pen_height = 0.2, 0.02, 0.02  # Adjust these dimensions as needed
pen_center = [0, 0, 0]  # Center of the prism
prism = create_rectangular_prism(pen_center, pen_length, pen_width, pen_height)


# Read data from CSV file
csv_file = 'Signatures_Data/Skandan3.csv'  # Replace with your CSV file path
with open(csv_file, newline='') as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    next(data_reader)  # Skip the header row
    for row in data_reader:
        # Convert strings to float and apply the exponential moving average filter
        acc = acc_filter.update(np.array([float(row[0]), float(row[1]), float(row[2])]))
        gyro = gyro_filter.update(np.array([float(row[3]), float(row[4]), float(row[5])]))
        mag = mag_filter.update(np.array([float(row[6]), float(row[7]), float(row[8])]))

        # Update the Madgwick filter and get the current orientation as a quaternion
        q = madgwick.updateMARG(q, acc=acc, gyr=gyro, mag=mag)

        # Convert the quaternion to a rotation matrix
        R = q2R(q)

        # Apply rotation to the prism vertices
        rotated_prism = []
        for face in prism:
            rotated_face = [np.dot(R, vertex) for vertex in face]
            rotated_prism.append(rotated_face)

        # Clear the previous drawing and draw the new one
        ax.clear()
        ax.add_collection3d(Poly3DCollection(rotated_prism, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        plt.pause(0.005)

plt.show()

#    "python.dataScience.enablePlotViewer": true,
#    "python.dataScience.runByLine": false,
#   "python.dataScience.stopOnFirstLineWhileDebugging": false
# sudo apt-get install python3-tk

