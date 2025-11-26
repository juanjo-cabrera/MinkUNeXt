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

            if not PARAMS.save_visual_results:
                pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets)
            else:
                pcd_dir = query_sets[j][0]['query_velo']
                parent_dir = os.path.dirname(os.path.dirname(pcd_dir))
                csv_file = PARAMS.dataset_folder + '/VISUAL_RESULTS_MinkUNeXt_baseline_KITTI/' + parent_dir + '.csv'
                csv_dir = os.path.dirname(csv_file)
                if not os.path.exists(csv_dir):
                    os.makedirs(csv_dir)

                pair_recall, pair_opr = get_recall_csv(i, j, database_embeddings, query_embeddings, query_sets,
                                                database_sets, csv_file=csv_file)


            # pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            # for x in pair_similarity:
            #     similarity.append(x)

    ave_recall = recall / count
    average_similarity = 0
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    # stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


def load_pc(filename):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
    file_path = os.path.join(PARAMS.dataset_folder, filename)
    pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])

    # remove intensity for models which are not using it
    pc = pc[:, :3]

    pc = torch.tensor(pc, dtype=torch.float)

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


def get_recall_csv(m, n, database_vectors, query_vectors, query_sets, database_sets, csv_file='results.csv'):


    import csv
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'])
        # Original PointNetVLAD code
        database_output = database_vectors[m]
        queries_output = query_vectors[n]

        # When embeddings are normalized, using Euclidean distance gives the same
        # nearest neighbour search results as using cosine distance
        database_nbrs = KDTree(database_output)

        num_neighbors = 25
        recall = [0] * num_neighbors

        one_percent_retrieved = 0
        threshold = max(int(round(len(database_output)/100.0)), 1)

        num_evaluated = 0
        errors = []
        for i in range(len(queries_output)):
            # i is query element ndx
            query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
            true_neighbor = query_details[m]
            #database_details = database_sets[true_neighbor]
            query_position = query_details['easting'], query_details['northing']
            # numpy array of position
            query_position = np.array([query_position])
            # check if index is correct
            #distance_position, index = database_positions_tree.query(query_position, k=1)
            #groundtruth_position = database_details['x'], database_details['y']
            # numpy array of position 
            
            if len(true_neighbor) == 0:
                continue
            num_evaluated += 1

            # Find nearest neightbours
            distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
            estimated_position = database_sets[m][indices[0][0]]['easting'], database_sets[m][indices[0][0]]['northing']
            estimated_position = np.array([estimated_position])
            #compute euclidean error between current_position and true_position

            metric_error = np.linalg.norm(estimated_position - query_position)
            errors.append(metric_error)

            recall1_retrieved = 0
            recall1percent_retrieved = 0
            for j in range(len(indices[0])):
                if indices[0][j] in true_neighbor:
                    recall[j] += 1
                    if j == 0:
                        recall1_retrieved = 1
                    break

            if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbor)))) > 0:
                one_percent_retrieved += 1
                recall1percent_retrieved = 1


            # write to csv file
            writer.writerow([query_details['query'], query_details['easting'], query_details['northing'], database_sets[m][indices[0][0]]['query'], database_sets[m][indices[0][0]]['easting'], database_sets[m][indices[0][0]]['northing'], database_sets[m][true_neighbor[0]]['query'], database_sets[m][true_neighbor[0]]['easting'], database_sets[m][true_neighbor[0]]['northing'], recall1_retrieved, recall1percent_retrieved])

        one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
        recall = (np.cumsum(recall)/float(num_evaluated))*100
        mean_error = np.mean(errors)
    return recall, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])

# save dictioanary stats to a file
def save_eval_stats(stats, filename):
    with open (filename, 'wb') as f:
        pickle.dump(stats, f)



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
    # PARAMS.weights_path =  '/home/arvc/Juanjo/develop/MinkUNeXt-clean/weights/MinkUNeXt_usyd_0.1_20250919_1400_best.pth' #'/home/arvc/Juanjo/develop/MinkUNeXt-clean/weights/MinkUNeXt_usyd_0.1_20250918_2327_best.pth'
    PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/MinkUNeXt/baseline_antonio/MinkUNeXt_baseline_0.01_20250917_0417_best.pth'
    # PARAMS.weights_path = '/media/arvc/DATOS/Juanjo/weights/MinkUNeXt/baseline_antonio/MinkUNeXt_baseline_0.01_20250916_2037_best.pth'
    PARAMS.save_visual_results = False

    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))
    model.to(device)

    stats = evaluate(model, device)
    print_eval_stats(stats)
    save_eval_stats(stats, '/media/arvc/DATOS/Juanjo/Datasets/benchmark_datasets/stats/MinkUNeXt_baseline_eval_stats_kitti.pickle')
    # # load and print stats
    # with open('/media/arvc/DATOS/Juanjo/Datasets/benchmark_datasets/stats/MinkUNeXt_refined_eval_stats_kitti.pickle', 'rb') as f:
    #     stats = pickle.load(f)
    # print_eval_stats(stats)
