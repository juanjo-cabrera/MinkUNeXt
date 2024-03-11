# Code taken from MinkLoc3Dv2 repo: https://github.com/jac99/MinkLoc3Dv2.git

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PARAMS 
from quantization import quantizer
from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.pointnetvlad.pnv_train import PNVTrainingDataset
from datasets.pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from datasets.samplers import BatchSampler
from datasets.base_datasets import PointCloudLoader
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    return PNVPointCloudLoader()


def make_datasets(validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(PARAMS.aug_mode)

    # PoinNetVLAD datasets (RobotCar and Inhouse)
    # PNV datasets have their own transform
    train_transform = PNVTrainTransform(PARAMS.aug_mode)
    datasets['train'] = PNVTrainingDataset(PARAMS.dataset_folder, PARAMS.train_file,
                                           transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['val'] = PNVTrainingDataset(PARAMS.dataset_folder, PARAMS.val_file)

    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.set_transform(clouds)
            clouds = clouds.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Convert to polar (when polar coords are used) and quantize
        # Use the first value returned by quantizer
        coords = [quantizer(e)[0] for e in clouds]

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn

class CartesianQuantizer():
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx

def make_dataloaders(validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(validation=validation)
    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=PARAMS.batch_size,
                                 batch_size_limit=PARAMS.batch_size_limit,
                                 batch_expansion_rate=PARAMS.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, PARAMS.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=PARAMS.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, PARAMS.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=PARAMS.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=PARAMS.num_workers, pin_memory=True)

    return dataloders


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


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

