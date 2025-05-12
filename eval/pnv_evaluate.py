
# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import MinkowskiEngine as ME
import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PARAMS 
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.quantization import quantizer


def evaluate(model, device, log: bool = False, show_progress: bool = False):
    # Run evaluation on all eval datasets

    eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                           'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']

    eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                        'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
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

        temp = evaluate_dataset(model, device, database_sets, query_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, database_sets, query_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, set, device))

    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, set, device))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            if not PARAMS.save_visual_results:
                pair_recall, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                    database_sets, log=log)
            else:
                pcd_dir = query_sets[j][0]['query']
                parent_dir = os.path.dirname(os.path.dirname(pcd_dir))
                csv_file = PARAMS.dataset_folder + '/VISUAL_RESULTS/' + parent_dir + '.csv'
                csv_dir = os.path.dirname(csv_file)
                if not os.path.exists(csv_dir):
                    os.makedirs(csv_dir)

                pair_recall, pair_opr = get_recall_csv(i, j, database_embeddings, query_embeddings, query_sets,
                                                database_sets, csv_file=csv_file)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


def get_latent_vectors(model, set, device):
    # Adapted from original PointNetVLAD code

    if PARAMS.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader()

    model.eval()
    embeddings = None
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(PARAMS.dataset_folder, set[elem_ndx]["query"])
        pc = pc_loader(pc_file_path)
        pc = torch.tensor(pc)

        embedding = compute_embedding(model, pc, device)
        if embeddings is None:
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(model, pc, device):
    coords, _ = quantizer(pc)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
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
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element) for Oxford dataset
            # Check if there's a false positive returned as the first element
            if query_details['query'][:6] == 'oxford' and indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break

                with open("log_fp.txt", "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            if query_details['query'][:6] == 'oxford':
                # Save details of 5 best matches for later visualization for 1% of queries
                s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
                for k in range(min(len(indices[0]), 5)):
                    is_match = indices[0][k] in true_neighbors
                    e_ndx = indices[0][k]
                    e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                    e_emb_dist = distances[0][k]
                    world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                         (query_details['easting'] - e['easting']) ** 2)
                    s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
                s += '\n'
                out_file_name = "log_search_results.txt"
                with open(out_file_name, "a") as f:
                    f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall

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
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
  
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    # set cuda device 
    torch.cuda.set_device(device)

    from model.minkunext import model

    
    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))

    model.to(device)

    stats = evaluate(model, device, log=False, show_progress=True)
    print_eval_stats(stats)

    # Save results to the text file

    model_name = os.path.split(PARAMS.weights_path)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}".format(PARAMS.protocol, model_name)
    pnv_write_eval_stats("results.txt", prefix, stats)


