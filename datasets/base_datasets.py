# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from bitarray import bitarray
import tqdm


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        # position: x, y position in meters
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        # pc_loader must be set in the inheriting class
        self.pc_loader: PointCloudLoader = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")



# class USydDataset(TrainingDataset):
#     """
#     Dataset wrapper for USyd laser scans datasets described in MinkLoc3D-SI.
#     """

#     def __init__(self, dataset_path, query_filename, n_points, max_distance, transform=None,
#                  set_transform=None):
#         # transform: transform applied to each element
#         # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
#         super().__init__(dataset_path, query_filename, transform, set_transform)çç

class USydDataset(Dataset):
    def __init__(self, dataset_path, query_filename, n_points, max_distance, transform=None, set_transform=None):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.n_points = n_points
        self.max_distance = max_distance  # maximum point cloud range for

        self.dtype = np.float32


    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx].rel_scan_filepath
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        # Subsample (limited number of points) or apply padding to have the same number of points
        # in batched clouds - required by augmentation functions
        padlen = self.n_points - len(query_pc)
        if padlen > 0:
            query_pc = torch.nn.functional.pad(query_pc, (0, 0, 0, padlen), "constant", 0)
        elif padlen < 0:
            query_pc = query_pc[:self.n_points]
        return query_pc, ndx

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
        file_path = os.path.join(self.dataset_path, filename)

        pc = np.fromfile(file_path, dtype=self.dtype).reshape([-1, 4])
        pc = pc[np.linalg.norm(pc[:, :3], axis=1) < self.max_distance]

        # not use intensity
        pc = pc[:, :3]

        # shuffle points in case they are randomly subsampled later
        np.random.shuffle(pc)
        pc = torch.tensor(pc, dtype=torch.float)
        return pc

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class TrainingDataset_v2(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """

    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        # transform: transform applied to each element
        # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
    
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform

        cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(self.query_filepath, cached_query_filepath)
        else:
            print('Loading preprocessed query file: {}...'.format(cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        print('{} queries in the dataset'.format(len(self)))
        # pc_loader must be set in the inheriting class
        self.pc_loader: PointCloudLoader = None

    def preprocess_queries(self, query_filepath, cached_query_filepath):
        print('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle) # pickle.load(open(self.query_filepath, 'rb'))

        # Convert to bitarray
        for ndx in tqdm.tqdm(queries):
            queries[ndx].positives = set(queries[ndx].positives)
            queries[ndx].non_negatives = set(queries[ndx].non_negatives)
            pos_mask = [e_ndx in queries[ndx].positives for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx].non_negatives for e_ndx in range(len(queries))]
            queries[ndx].positives = bitarray(pos_mask)
            queries[ndx].non_negatives = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        # query_pc = self.load_pc(filename)
        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc, ndx

    def get_item_by_filename(self, filename):
        # Load point cloud and apply transform
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc

    def get_items(self, ndx_l):
        # Load multiple point clouds and stack into (batch_size, n_points, 3) tensor
        clouds = [self[ndx][0] for ndx in ndx_l]
        clouds = torch.stack(clouds, dim=0)
        return clouds

    def get_positives_ndx(self, ndx):
        # Get list of indexes of similar clouds
        return self.queries[ndx].positives.search(bitarray([True]))

    def get_non_negatives_ndx(self, ndx):
        # Get list of indexes of dissimilar clouds
        return self.queries[ndx].non_negatives.search(bitarray([True]))

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(filename)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = pc[np.linalg.norm(pc[:, :3], axis=1) < self.max_distance]
        if pc.size == 0:
            print(filename)
        pc = torch.tensor(pc, dtype=torch.float)
        return pc


class USydDataset_v2(TrainingDataset_v2):
    """
    Dataset wrapper for USyd
    """

    def __init__(self, dataset_path, query_filename, n_points, max_distance, transform=None, set_transform=None):
        # transform: transform applied to each element
        # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
        super().__init__(dataset_path, query_filename, transform, set_transform)
        self.n_points = n_points
        self.max_distance = max_distance  # maximum point cloud range     
        self.dtype = np.float32
   

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        # Subsample (limited number of points) or apply padding to have the same number of points
        # in batched clouds - required by augmentation functions
        padlen = self.n_points - len(query_pc)
        if padlen > 0:
            query_pc = torch.nn.functional.pad(query_pc, (0, 0, 0, padlen), "constant", 0)
        elif padlen < 0:
            query_pc = query_pc[:self.n_points]
        return query_pc, ndx

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
        file_path = os.path.join(self.dataset_path, filename)
        #file_path = file_path.replace("velodyne", "velodyne_no_ground")

        pc = np.fromfile(file_path, dtype=self.dtype).reshape([-1, 4])
        pc = pc[np.linalg.norm(pc[:, :3], axis=1) < self.max_distance]

        # not use intensity
        pc = pc[:, :3]
        # shuffle points in case they are randomly subsampled later
        np.random.shuffle(pc)
        pc = torch.tensor(pc, dtype=torch.float)
        return pc

 
