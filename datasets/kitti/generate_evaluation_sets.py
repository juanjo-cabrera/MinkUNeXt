# Test set for Kitti Sequence 00 dataset.
# Following procedures in [cite papers Kitti for place reco] we use 170 seconds of drive from sequence for map generation
# and the rest is left for queries

import numpy as np
import argparse
from typing import List
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from config import PARAMS
from datasets.kitti.kitti_raw import KittiSequence
from datasets.base_datasets import EvaluationTuple, EvaluationSet
from sklearn.neighbors import KDTree
import pickle



MAP_TIMERANGE = (0, 170)

def output_to_file(output, base_path, filename):
    filepath = os.path.join(base_path, filename)
    with open(filepath, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filepath)

def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set

def get_scans(sequence: KittiSequence, min_displacement: float = 0.1, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all point clouds from the sequence (the full sequence or test split only)

    elems = []
    old_pos = None
    count_skipped = 0
    displacements = []

    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.rel_lidar_timestamps[ndx]) or (ts_range[1] < sequence.rel_lidar_timestamps[ndx]):
                continue
        pose = sequence.lidar_poses[ndx]
        # Kitti poses are in camera coordinates system where where y is upper axis dim
        position = pose[[0,2], 3]

        if old_pos is not None:
            displacements.append(np.linalg.norm(old_pos - position))

            if np.linalg.norm(old_pos - position) < min_displacement:
                # Ignore the point cloud if the vehicle didn't move
                count_skipped += 1
                continue

        item = EvaluationTuple(sequence.rel_lidar_timestamps[ndx], sequence.rel_scan_filepath[ndx], position)
        elems.append(item)
        old_pos = position

    print(f'{count_skipped} clouds skipped due to displacement smaller than {min_displacement}')
    print(f'mean displacement {np.mean(np.array(displacements))}')
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, min_displacement: float = 0.1,
                            dist_threshold: float = 5.) -> EvaluationSet:

    sequence = KittiSequence(dataset_root, map_sequence)

    map_set = get_scans(sequence, min_displacement, MAP_TIMERANGE)
    query_set = get_scans(sequence, min_displacement, (MAP_TIMERANGE[-1], sequence.rel_lidar_timestamps[-1]))
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')

    output_to_file(map_set, dataset_root, 'KITTI_00_database_samp10_custom.pickle')
    output_to_file(query_set, dataset_root, 'KITTI_00_query_samp10_custom.pickle')

    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':

    PARAMS.dataset_folder = '/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/'

    # Sequences are fixed
    sequence = '00'
    min_displacement = 10.0
    dist_threshold = 25.0
    dataset_root = PARAMS.dataset_folder
    
    print(f'Kitti sequence: {sequence}')
    print(f'Minimum displacement between consecutive anchors: {min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {dist_threshold}')

    kitti_eval_set = generate_evaluation_set(dataset_root, sequence, min_displacement=min_displacement,
                                             dist_threshold=dist_threshold)
    file_path_name = os.path.join(dataset_root, f'kitti_{sequence}_eval.pickle')
    print(f"Saving evaluation pickle: {file_path_name}")
    kitti_eval_set.save(file_path_name)