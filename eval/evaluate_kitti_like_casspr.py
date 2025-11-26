# Author: Jacek Komorowski, Monika Wysoczanska
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import sys
import sys
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
import argparse
import torch
import MinkowskiEngine as ME
import tqdm
import open3d as o3d



DEBUG = False


def evaluate(model, device):
    # Run evaluation on all eval datasets

    # if DEBUG:
    #     params.eval_database_files = params.eval_database_files[0:1]
    #     params.eval_query_files = params.eval_query_files[0:1]

    assert len(PARAMS.eval_database_files) == len(PARAMS.eval_query_files)

    stats = {}
    for database_file, query_file in zip(PARAMS.eval_database_files, PARAMS.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(PARAMS.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(PARAMS.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, database_sets, query_sets)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, database_sets, query_sets):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in database_sets:
        database_embeddings.append(get_latent_vectors(model, set, device))

    for set in query_sets:
        query_embeddings.append(get_latent_vectors(model, set, device))

    for i in tqdm.tqdm(range(len(query_sets))):
        for j in range(len(query_sets)):
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats


# def load_pc(filename):
#     # Load point cloud, does not apply any transform
#     # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
#     file_path = os.path.join(PARAMS.dataset_folder, filename)
#     pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])

#     # remove intensity for models which are not using it
#     pc = pc[:, :3]

#     pc = torch.tensor(pc, dtype=torch.float)

#     return pc

def pnv_preprocessing(xyz):
    vox_sz = 0.3
    while np.asarray(xyz).shape[0] > 4096:
        xyz = downsample_point_cloud(xyz, vox_sz)
        vox_sz += 0.01
    return np.asarray(xyz)

def downsample_point_cloud(xyz, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    return pcd_ds.points

def load_pc(pc_file_path):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
        
    file_path = os.path.join(PARAMS.dataset_folder, pc_file_path)
    pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
    
    pc = pc[:, :3]
    # limit distance
    pc = pc[np.linalg.norm(pc[:, :3], axis=1) < PARAMS.max_distance]
    
    pc = pnv_preprocessing(pc)

    pc = torch.tensor(pc, dtype=torch.float)
    padlen = PARAMS.num_points - len(pc)
    if padlen > 0:
        pc = torch.nn.functional.pad(pc, (0, 0, 0, padlen), "constant", 0)
    elif padlen < 0:
        pc = pc[:PARAMS.num_points]
   

   

    return pc


def get_latent_vectors(model, set, device):
    # Adapted from original PointNetVLAD code

    """
    if DEBUG:
        embeddings = torch.randn(len(set), 256)
        return embeddings
    """

    # if DEBUG:
    #     embeddings = np.random.rand(len(set), 256)
    #     return embeddings

    model.eval()
    embeddings_l = []
    for elem_ndx in tqdm.tqdm(set):
        x = load_pc(set[elem_ndx]["query_velo"])

        with torch.no_grad():
            # models without intensity
            
            coords = ME.utils.sparse_quantize(coordinates=x,
                                                  quantization_size=PARAMS.mink_quantization_size)
            bcoords = ME.utils.batched_coordinates([coords]).to(device)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)



            batch = {'coords': bcoords, 'features': feats}
            y = model(batch)      

        embedding = y['global'].detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets):
    # based on original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":

    # eval_database_files = ["/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/KITTI_00_database_samp10.pickle"]
    # eval_query_files = ["/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/KITTI_00_query_samp10.pickle"]
    # dataset_folder = "/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/sequences/00"
    # dataset_name = "KITTI"
    # mink_quantization_size = [2.5, 2.0, 0.42]
   
    print('#'*30)
    print('WARNING: Database and query files, paths and quantization from config are overwritten by KITTI specs in evaluate_kitti.py.')
    print('#'*30, '\n')


    PARAMS.cuda_device = 'cuda:1'
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    # set cuda device 
    torch.cuda.set_device(device)
    from model.minkunext import model
    PARAMS.max_distance = 100
    PARAMS.num_points = 4096
    PARAMS.weights_path =  '/home/arvc/Juanjo/develop/MinkUNeXt-clean/weights/MinkUNeXt_usyd_0.1_20250918_2327_best.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/MinkUNeXt/baseline_antonio/MinkUNeXt_baseline_0.01_20250917_0417_best.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/MinkUNeXt/baseline_antonio/MinkUNeXt_baseline_0.01_20250916_2037_best.pth'
    
    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))
    model.to(device)

    stats = evaluate(model, device)
    print_eval_stats(stats)
