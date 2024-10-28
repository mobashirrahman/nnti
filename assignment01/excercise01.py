# Exercise 1

"""
We will use the NumPy python library to generate some data, and in the process get used to
the library functions. Do the following exercise using vectorised numpy operations instead of
python loops.
"""

# Task01 Description:
"""
1. Write a function to generate two sets of points that can be linearly separated in 2D space:
• Use np.random.randn() to create two clusters of 100 points each, each representing
a different class. (This function can only generate the standard normal distribution).
• Move the mean of each of these clusters to [-2, -2] and [2, 2] respectively.
• Assign classes to these clusters as -1 and 1.
• Combine and shuffle the points from different clusters. Make sure to combine and
shuffle the assigned classes accordingly.
• Return the points and their classes as separate numpy arrays of shape (N, 2) and
(N, 1) respectively.
"""

# Import library
import numpy as np

def generate_linearly_separable_data(num_points_per_cluster=100):
    """
    Generate two linearly separable clusters of 2D points.

    Parameters:
    num_points_per_cluster (int): Number of points to generate for each cluster.

    Returns:
    X (numpy.ndarray): Array of 2D points, shape (2 * num_points_per_cluster, 2)
    y (numpy.ndarray): Array of class labels (-1 and 1), shape (2 * num_points_per_cluster, 1)
    """
    # Generate two clusters of points from a normal distribution
    cluster1 = np.random.randn(num_points_per_cluster, 2)
    cluster2 = np.random.randn(num_points_per_cluster, 2)

    # Shift the means of the clusters
    cluster1 += np.array([-2, -2])
    cluster2 += np.array([2, 2])

    # Assign class labels (-1 and 1)
    labels1 = -np.ones((num_points_per_cluster, 1))
    labels2 = np.ones((num_points_per_cluster, 1))

    # Combine the data and shuffle
    X = np.concatenate([cluster1, cluster2], axis=0)
    y = np.concatenate([labels1, labels2], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    
    # Return the points
    X = X[indices]
    y = y[indices]

    return X, y


# Task02 Description:

"""        
Write a function to similarly generate the XOR dataset:        
• It consists of 4 points as follows: [[0, 0], [0, 1], [1, 0], [1, 1]].
• The class for each point is decided as the XOR of its x and y coordinate.
• Instead of [0, 1], change the class labels to [-1, 1].
• Return the points and their classes as separate numpy arrays of shape (N, 2) and
(N, 1) respectively.
"""

import numpy as np

def generate_xor_data():
    # Define the four points
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])  # Shape (4, 2)
    
    # Compute the XOR of x and y coordinates to get class labels
    xor_values = np.logical_xor(points[:, 0], points[:, 1]).astype(int)  # Shape (4,)
    
    # Convert class labels from [0, 1] to [-1, 1]
    classes = np.where(xor_values == 0, -1, 1).reshape(-1, 1)  # Shape (4, 1)
    
    return points, classes

# Task03 Description
"""
Write a function to visualise these clusters and their labels as different colors using
matplotlib.pyplot.plot
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_clusters(points, classes):
    # Flatten the classes array for easier indexing
    class_labels = classes.flatten()
    
    # Get unique class labels
    unique_classes = np.unique(class_labels)
    
    # Define a color map
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Plot each class separately
    for i, cls in enumerate(unique_classes):
        # Select points belonging to the current class
        cls_points = points[class_labels == cls]
        
        # Plot the points using plt.plot
        plt.plot(cls_points[:, 0], cls_points[:, 1],
                 marker='o', linestyle='',
                 color=colors[i % len(colors)],
                 label=f'Class {int(cls)}')
    
    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Clusters Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

