import os
import json
import util.util as util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict
from datetime import datetime, timedelta

# Table settings
tables = {
    "Table A": {"x_min": -1.2, "x_max": 0.2, "y_min": 0.6, "y_max": 2.0},
    "Table B": {"x_min": -0.2, "x_max": 1.2, "y_min": 2.0, "y_max": 3.2},
    "Table C": {"x_min": -1.5, "x_max": 0.0, "y_min": 3.2, "y_max": 4.5},
}
z_min, z_max = 0.0, 2.0  # Z-axis range is the same for all tables

# Main loop for processing files (with clutter removal and DBSCAN noise removal)
input_folder = ''  # Update to use split files folder
density_data = {table: [] for table in tables}
centroid_data = {table: [] for table in tables}
point_data = {table: [] for table in tables}

total_points_per_file = {}
overall_total_points = 0

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.json'):
        input_file_path = os.path.join(input_folder, file_name)
        print(f'Processing file {input_file_path}')

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

            # Noise removal
            clustered_points, labels = apply_dbscan(coordinates_with_v)
            filtered_coordinates = remove_static_points(clustered_points, previous_frame_points_per_radar[radar_num])
            previous_frame_points_per_radar[radar_num] = clustered_points

            aggregated_points[time_block].extend(filtered_coordinates)

        # get point cloud features 
        for time_block, points in aggregated_points.items():
            for table_name, table_area in tables.items():
                table_points = filter_points_in_table_area(points, table_area)
                density = calculate_density(table_points)
                central_point = calculate_central_point(table_points)

                # Store the density and centroid data for each table
                density_data[table_name].append({'time': time_block, 'density': density, 'file': file_name, 'action': action})
                centroid_data[table_name].append({'time': time_block, 'centroid_x': central_point[0], 'centroid_y': central_point[1], 'centroid_z': central_point[2], 'file': file_name, 'action': action})
                point_data[table_name].append({'time': time_block, 'table_points': table_points, 'file': file_name, 'action': action})

# Convert the data to DataFrames for plotting and further analysis
density_dfs = {table: pd.DataFrame(data) for table, data in density_data.items()}
centroid_dfs = {table: pd.DataFrame(data) for table, data in centroid_data.items()}
point_dfs = {table: pd.DataFrame(data) for table, data in point_data.items()}

# plot the stac for data
# for table_name, df in density_dfs.items():
#     plt.figure(figsize=(10, 6))
#     df.boxplot(column='density', by='file', grid=False)
#     plt.title(f'Point Cloud Density Distribution by File ({table_name})')
#     plt.suptitle('')  # Removes the automatic boxplot title
#     plt.xlabel('File')
#     plt.ylabel('Density (Number of Points)')
#     plt.xticks(rotation=45)
#     plt.show()


# Maximum number of points in each point cloud
MAX_POINTS = 100

# give labes to files
file_label_mapping = {}

# Main loop for processing files (with clutter removal and DBSCAN noise removal)
input_folder = 'data_1010/split'  # Update to use split files folder
density_data = {table: [] for table in tables}
centroid_data = {table: [] for table in tables}
point_data = {table: [] for table in tables}

total_points_per_file = {}
overall_total_points = 0

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.json'):
        input_file_path = os.path.join(input_folder, file_name)
        print(f'Processing file {input_file_path}')

        total_points_before_clutter = 0
        total_points_after_clutter = 0
        total_points_after_dbscan = 0

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

            # Noise removal
            clustered_points, labels = apply_dbscan(coordinates_with_v)
            filtered_coordinates = remove_static_points(clustered_points, previous_frame_points_per_radar[radar_num])
            previous_frame_points_per_radar[radar_num] = clustered_points

            # # Convert mmWave points to coordinate format
            # coordinates = [(x, y, z) for x, y, z in zip(mmwave_point['x'], mmwave_point['y'], mmwave_point['z']) if x is not None and y is not None and z is not None]

            # # remove points exceeding normal height
            # coordinates = [p for p in coordinates if 0 <= p[2] <= 2.0]

            # total_points_before_clutter += len(coordinates)

            # # Noise remove step 1: DBscan
            # clustered_points, labels = apply_dbscan(coordinates)
            # total_points_after_dbscan += len(clustered_points)

            # # Noise remove step 2: remove static points
            # filtered_coordinates = remove_static_points(clustered_points, previous_frame_points_per_radar[radar_num])
            # previous_frame_points_per_radar[radar_num] = clustered_points  # Update the previous frame points for the current radar
            # total_points_after_clutter += len(filtered_coordinates)

            # print(f"Time: {time_block}, Radar Num: {radar_num}, Original: {len(coordinates)}, After DBSCAN: {len(clustered_points)}, After static: {len(filtered_coordinates)}")

            aggregated_points[time_block].extend(filtered_coordinates)
            # aggregated_points[time_block].extend(coordinates)

        for time_block, points in aggregated_points.items():
            for table_name, table_area in tables.items():
                table_points = filter_points_in_table_area(points, table_area)
                density = calculate_density(table_points)
                central_point = calculate_central_point(table_points)

                # Store the density and centroid data for each table
                density_data[table_name].append({'time': time_block, 'density': density, 'file': file_name, 'action': action})
                centroid_data[table_name].append({'time': time_block, 'centroid_x': central_point[0], 'centroid_y': central_point[1], 'centroid_z': central_point[2], 'file': file_name, 'action': action})
                point_data[table_name].append({'time': time_block, 'table_points': table_points, 'file': file_name, 'action': action})

        total_points_per_file[file_name] = {
            'total_before_clutter': total_points_before_clutter,
            'total_after_clutter': total_points_after_clutter,
            'total_after_dbscan': total_points_after_dbscan
        }

        overall_total_points += total_points_after_dbscan

# Print the results for each file
for file_name, totals in total_points_per_file.items():
    print(f"File: {file_name}")
    print(f"  Total points before clutter removal: {totals['total_before_clutter']}")
    print(f"  Total points after clutter removal: {totals['total_after_clutter']}")
    print(f"  Total points after DBSCAN noise removal: {totals['total_after_dbscan']}")
    print()

# Print the overall total points across all files
print(f"Overall total points after DBSCAN noise removal across all files: {overall_total_points}")

# Convert the data to DataFrames for plotting and further analysis
density_dfs = {table: pd.DataFrame(data) for table, data in density_data.items()}
centroid_dfs = {table: pd.DataFrame(data) for table, data in centroid_data.items()}
point_dfs = {table: pd.DataFrame(data) for table, data in point_data.items()}

