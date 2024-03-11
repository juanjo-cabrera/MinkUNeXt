# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import pandas as pd
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS
import tqdm

# Import test set boundaries
from datasets.pointnetvlad.generate_test_sets import P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, check_in_test_set
from datasets.pointnetvlad.generate_training_tuples_baseline import construct_query_dict

# Test set boundaries
P = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]


if __name__ == '__main__':
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    runs_folder = "inhouse_datasets/"
    filename = "pointcloud_centroids_10.csv"
    pointcloud_fols = "/pointcloud_25m_10/"

    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

    folders = []
    index_list = range(5, 15)
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)

    ####Initialize pandas DataFrame
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                continue
            else:
                df_train = df_train.append(row, ignore_index=True)

    print(len(df_train['file']))

    ##Combine with Oxford data
    runs_folder = "oxford/"
    filename = "pointcloud_locations_20m_10overlap.csv"
    pointcloud_fols = "/pointcloud_20m_10overlap/"

    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

    folders = []
    index_list = range(len(all_folders) - 1)
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)

    for folder in folders:
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                continue
            else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    # ind_nn_r is a threshold for positive elements - 12.5 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_refine_pruebas.pickle", ind_nn_r=12.5)
