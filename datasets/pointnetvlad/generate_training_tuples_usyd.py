# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import pandas as pd
import numpy as np
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from sklearn.neighbors import KDTree
import pickle
import random
from datasets.base_datasets import TrainingTuple


base_path = "/media/arvc/DATOS/Juanjo/Datasets/USyd/"

runs_folder = "weeks/"
pointcloud_fols = "/pointclouds_with_locations_5m/"

folders = []
filenames = []
training_weeks = [1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                  26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 45, 47, 49, 52]

folders = [f'output_week{week}' for week in training_weeks]
filenames = [f'pointcloud_locations_5m_week_{str(week).zfill(2)}.csv' for week in training_weeks]
print("Number of runs: " + str(len(folders)))
print(folders)
print(filenames)

#####For training and test data split#####
x_width = 100
y_width = 100
buffer = 10
# points in easting, northing (x, y) format
p1_bl_corner = [332_530, -3_750_950]
p2_bl_corner = [332_250, -3_751_240]
p3_bl_corner = [332_630, -3_751_450]
p4_bl_corner = [332_555, -3_751_125]
p = [p1_bl_corner, p2_bl_corner, p3_bl_corner, p4_bl_corner]


# modified, since regions are defined by bottom left corner + width and buffer is added
def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    in_buffer_set = False

    for point in points:
        # points in easting, northing (x, y) format
        if (point[0] - buffer) < easting < (point[0] + x_width + buffer) and (point[1] - buffer) < northing < (point[1] + y_width + buffer):
            # in buffer range - test or reject:
            if (point[0]) < easting < (point[0] + x_width) and (point[1]) < northing < (point[1] + y_width):
                # in test range
                in_test_set = True
            else:
                in_buffer_set = True
            break
    return in_test_set, in_buffer_set


##########################################


def construct_query_dict(df_centroids, base_path, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    # CURRENT DISTANCES: POS<10, NEG>25
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=25)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", file_path)


####Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder, filename in zip(folders, filenames):
    print(os.path.join(base_path, runs_folder, folder, filename))
    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        in_test, in_buffer = check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)
        if in_test and not in_buffer:
            df_test = df_test.append(row, ignore_index=True)
        elif not in_buffer:
            df_train = df_train.append(row, ignore_index=True)

print("Number of train clouds: " + str(len(df_train['file'])))
print("Number of test clouds: " + str(len(df_test['file'])))

construct_query_dict(df_train, base_path, "usyd_training_queries.pickle")
construct_query_dict(df_test, base_path, "usyd_test_queries.pickle")
