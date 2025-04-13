import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Function to round time to the nearest 0.1 second
def round_to_0_1_seconds(time):
    """Round time to the nearest 0.1 second."""
    return time + timedelta(seconds=round(time.microsecond / 1_000_000, 1)) - timedelta(microseconds=time.microsecond)

# Helper function to handle None values in coordinates
def safe_negate(x):
    return -x if x is not None else None

# Function to truncate fractional seconds to six digits
def truncate_to_six_microseconds(time_str):
    date, time = time_str.split('T')
    time, fraction = time.split('.')
    fraction = fraction[:6]  # Keep only the first six digits
    return f"{date}T{time}.{fraction}Z"

# Function to truncate timestamps to the nearest 0.1 second
def truncate_to_0_1_seconds(time):
    """Truncate time to the nearest 0.1 second."""
    time_diff = timedelta(microseconds=time.microsecond)
    return time - time_diff + timedelta(milliseconds=round(time_diff.total_seconds() * 1000, -1))

# Function to aggregate data by radar and remove duplicates
def process_mmwave_data(data):
    time_block_data = defaultdict(list)  # Dictionary to hold aggregated data by time block
    
    for point in data:
        # Truncate time to the nearest 0.1 second
        time_block = truncate_to_0_1_seconds(point['time'])
        
        # Store data in corresponding time block, grouped by radar
        radar_num = point['l'][0].split(":")[0]  # Extract radar number (e.g., "2")

        # Append the point to the correct time block
        time_block_data[time_block].append((radar_num, point))

    processed_data = []
    for time_block, points in time_block_data.items():
        radar_data = {}
        for radar_num, point in points:
            if radar_num not in radar_data:
                radar_data[radar_num] = point  # Add radar point if not already present

        # Only keep time blocks with at least one unique radar point
        if len(radar_data) > 0:
            processed_data.append((time_block, list(radar_data.values())))

    return processed_data

# Function to filter points within a specified table area
def filter_points_in_table_area(points, table_area):
    return [(x, y, z, v) for x, y, z, v in points if table_area["x_min"] <= x <= table_area["x_max"]
            and table_area["y_min"] <= y <= table_area["y_max"]
            and z_min <= z <= z_max]

# Function to calculate point cloud density
def calculate_density(points):
    return len(points)

# Function to calculate the central point (centroid) of the points
def calculate_central_point(points):
    if len(points) == 0:
        return (None, None, None)  # Return None if there are no points
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    centroid_z = np.mean(z_coords)
    return (centroid_x, centroid_y, centroid_z)

# Function to remove static points by comparing all points between current and previous frames
def remove_static_points(current_frame, previous_frame, tolerance=0.03):
    """Remove points from the current frame that haven't moved compared to the previous frame."""
    if previous_frame is None or len(previous_frame) == 0:
        return current_frame  # If no previous frame, return current points
    
    # Convert to numpy arrays
    current_xyz = np.array([(x, y, z) for x, y, z, v in current_frame])
    previous_xyz = np.array([(x, y, z) for x, y, z, v in previous_frame])

    tree = cKDTree(previous_xyz)
    distances, _ = tree.query(current_xyz)
    moved_points = distances > tolerance
    return [current_frame[i] for i in range(len(current_frame)) if moved_points[i]]

# Function to apply DBSCAN with a modified distance metric
def apply_dbscan(points, eps=0.3, min_samples=5, z_weight=0.25):
    """Apply DBSCAN with a modified distance function for 3D points, with less emphasis on the z-axis."""
    if len(points) == 0:
        return [], []
    
    # points = np.array(points)
    xyz_points = np.array([(x, y, z) for x, y, z, v in points])
    
    # Custom distance metric for DBSCAN
    def modified_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + z_weight * (point1[2] - point2[2])**2)
    
    # Apply DBSCAN using the custom metric
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=modified_distance)
    # labels = clustering.fit_predict(points)
    labels = clustering.fit_predict(xyz_points)

    clustered_points = [points[i] for i in range(len(points)) if labels[i] != -1]
    return clustered_points, labels

def plot_loss(history):
    # Extract loss values from the training process
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Number of epochs (assuming you know how many epochs were run)
    epochs = range(1, len(train_loss) + 1)

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss values
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add title and labels
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set x-axis to show only integer values (epoch numbers)
    plt.xticks(epochs)  # Set x-ticks to be exactly the epoch numbers (1, 2, 3, ..., 20)

    plt.legend()
    plt.show()