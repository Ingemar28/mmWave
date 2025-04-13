import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.config import TABLES, Z_MIN, Z_MAX

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
            and Z_MIN <= z <= Z_MAX]

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

    if len(current_frame) == 0:
        return current_frame  # If no points in the current frame, return it as is

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
    labels = clustering.fit_predict(xyz_points)

    clustered_points = [points[i] for i in range(len(points)) if labels[i] != -1]
    return clustered_points, labels

# Main loop for processing files (with clutter removal and DBSCAN noise removal)
input_folder = 'data/'  # Update to use split files folder
density_data = {table: [] for table in TABLES}
centroid_data = {table: [] for table in TABLES}
point_data = {table: [] for table in TABLES}

total_points_per_file = {}
overall_total_points = 0

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.json'):
        input_file_path = os.path.join(input_folder, file_name)
        print(f'Processing file {input_file_path}')

        total_points_before_clutter = 0
        total_points_after_dbscan = 0
        total_points_after_dbscan_and_static_points = 0

        with open(input_file_path, 'r') as file:
            data = json.load(file)

        for point in data:
            point['time'] = round_to_0_1_seconds(datetime.strptime(truncate_to_six_microseconds(point['time']), '%Y-%m-%dT%H:%M:%S.%fZ'))

        action = file_name.split('_')[-1].replace('.json', '')  # Get the action (sit, stand, hybrid) from file name

        # Separate previous frame points for each radar
        previous_frame_points_per_radar = defaultdict(lambda: None)

        # Aggregate points across all radars for each time block
        aggregated_points = defaultdict(list)

        # Convert mmWave points to coordinate format and process
        for mmwave_point in data:
            # skip when data contains no point
            if len(mmwave_point['x']) == 0:
                continue

            time_block = mmwave_point['time']
            radar_num = mmwave_point['l'][0].split(":")[0]  # Extract radar number

            coordinates_with_v = [(x, y, z, v) for x, y, z, v in zip(mmwave_point['x'], mmwave_point['y'], mmwave_point['z'], mmwave_point['v']) if x is not None and y is not None and z is not None]
            coordinates_with_v = [p for p in coordinates_with_v if 0 <= p[2] <= 2.0]

            total_points_before_clutter += len(coordinates_with_v)

            # Noise remove step 1: DBscan
            clustered_points, labels = apply_dbscan(coordinates_with_v)
            total_points_after_dbscan += len(clustered_points)

            # Noise remove step 2: remove static points
            filtered_coordinates = remove_static_points(clustered_points, previous_frame_points_per_radar[radar_num])
            previous_frame_points_per_radar[radar_num] = clustered_points  # Update the previous frame points for the current radar
            total_points_after_dbscan_and_static_points += len(filtered_coordinates)

            # print(f"Time: {time_block}, Radar Nu m: {radar_num}, Original: {len(coordinates_with_v)}, After DBSCAN: {len(clustered_points)}, After static: {len(filtered_coordinates)}")

            aggregated_points[time_block].extend(filtered_coordinates)

        for time_block, points in aggregated_points.items():
            for table_name, table_area in TABLES.items():
                table_points = filter_points_in_table_area(points, table_area)
                density = calculate_density(table_points)
                central_point = calculate_central_point(table_points)

                # Store the density and centroid data for each table
                density_data[table_name].append({'time': time_block, 'density': density, 'file': file_name, 'action': action})
                centroid_data[table_name].append({'time': time_block, 'centroid_x': central_point[0], 'centroid_y': central_point[1], 'centroid_z': central_point[2], 'file': file_name, 'action': action})
                point_data[table_name].append({'time': time_block, 'table_points': table_points, 'file': file_name, 'action': action})

        total_points_per_file[file_name] = {
            'total_before_clutter': total_points_before_clutter,
            'total_after_dbscan': total_points_after_dbscan,
            'total_points_after_dbscan_and_static_points': total_points_after_dbscan_and_static_points
        }

        overall_total_points += total_points_after_dbscan

# Print the results for each file
for file_name, totals in total_points_per_file.items():
    print(f"File: {file_name}")
    print(f"  Total points before clutter removal: {totals['total_before_clutter']}")
    print(f"  Total points after DBSCAN noise removal: {totals['total_after_dbscan']}")
    print(f"  Total points after DBscan and static removal: {totals['total_points_after_dbscan_and_static_points']}\n")

# Print the overall total points across all files
print(f"Overall total points after DBSCAN noise removal across all files: {overall_total_points}")

# Convert the data to DataFrames for plotting and further analysis
density_dfs = {table: pd.DataFrame(data) for table, data in density_data.items()}
centroid_dfs = {table: pd.DataFrame(data) for table, data in centroid_data.items()}
point_dfs = {table: pd.DataFrame(data) for table, data in point_data.items()}

with open("data/density_dfs.pkl", "wb") as f:
    pickle.dump(density_dfs, f)

with open("data/centroid_dfs.pkl", "wb") as f:
    pickle.dump(centroid_dfs, f)

with open("data/point_dfs.pkl", "wb") as f:
    pickle.dump(point_dfs, f)

for table_name, df in density_dfs.items():
    plt.figure(figsize=(10, 6))
    df.boxplot(column='density', by='file', grid=False)
    plt.title(f'Point Cloud Density Distribution by File ({table_name})')
    plt.suptitle('')  # Removes the automatic boxplot title
    plt.xlabel('File')
    plt.ylabel('Density (Number of Points)')
    plt.xticks(rotation=45)
    plt.show()

