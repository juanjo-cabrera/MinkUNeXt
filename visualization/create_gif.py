# read csv file with pandas 
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.image as mpimg
import open3d as o3d
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config.config import PARAMS
 # Importamos PyVista solo si es necesario
import pyvista as pv

X_WIDTH = 150
Y_WIDTH = 150

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

P_DICT = {"oxford": [P1, P2, P3, P4], "university": [P5, P6, P7], "residential": [P8, P9, P10], "business": []}

def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):

    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        csv_file = PARAMS.dataset_folder + '/VISUAL_RESULTS/' + runs_folder + folder + '.csv'
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        df_results = pd.read_csv(csv_file)

        # create the destination folder to save pcd images
        dst_images_dir = PARAMS.dataset_folder + '/pcd_images/'
        results_dir = PARAMS.dataset_folder + '/VISUAL_RESULTS/' + runs_folder + folder + '/' 
        if not os.path.exists(dst_images_dir):
            os.makedirs(dst_images_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
 
        # get the pointcloud positions
        plot_pcds_and_positions(df_results, df_database, dst_images_dir, results_dir, p)

        

def get_pointcloud_positions(folder_dir, folders):
    # given a folder, return the positions of the pointclouds
    # each pointcloud file name is the timestamp, the 'x', 'y' and 'a' orientation, for example file_pathname = t1152904768.768371_x-8.640943_y2.861793_a-0.209387.ply
    timestamps = []
    x_positions = []
    y_positions = []
    orientations = []
    files_names = []
    for folder in folders:
        room_dir = os.path.join(folder_dir, folder)
        # check if the folder is a directory    
        if not os.path.isdir(room_dir):
            continue
        for file in os.listdir(room_dir):
            if file.endswith(".ply"):        
                files_names.append(room_dir + '/' +file)
                # quitar la extension del archivo
                file = file[:-4]
                timestamp_index = file.index('t')
                x_index = file.index('_x')
                y_index = file.index('_y')
                a_index = file.index('_a')
                timestamp = file[timestamp_index+1:x_index]         


                x = file[x_index+2:y_index]
                y = file[y_index+2:a_index]
                a = file[a_index+2:]
                # x, y, a are strings, parse them to float
                x = float(x)
                y = float(y)
                a = float(a)
                timestamp = float(timestamp)

                timestamps.append(timestamp)
                x_positions.append(x)
                y_positions.append(y)
                orientations.append(a)
    df_locations = pd.DataFrame({ 'file': files_names, 'timestamp': timestamps, 'x': x_positions, 'y': y_positions, 'orientation': orientations})
    return df_locations


def get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin):
    if coordX < xmin:
        xmin = coordX
    if coordX > xmax:
        xmax = coordX
    if coordY < ymin:
        ymin = coordY
    if coordY > ymax:
        ymax = coordY
    return xmax, xmin, ymax, ymin



def display_coord_map(df, df_database):
    # df header 'query_image', 'query_x', 'query_y', 'retrieved_database_image', 'retrieved_database_x', 'retrieved_database_y', 'real_database_image', 'real_database_x', 'real_database_y', 'recall@1', 'recall@1%'
    # df_database header 'file', 'timestamp', 'x', 'y', 'orientation'
    # plt tkagg
    plt.switch_backend('tkagg')    
    

    
    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 6), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True
    # get the coordinates of the visual model
    mapVM = df_database[['x', 'y']].to_numpy()
    plt.scatter(mapVM[:, 0], mapVM[:, 1], color='blue', label="Visual Model")
    xmax, xmin, ymax, ymin = get_axes_limits(mapVM[0][0], mapVM[0][1], xmax, xmin, ymax, ymin)

    # get the coordinates of the test images
    mapTest = df[['query_x', 'query_y', 'retrieved_database_x', 'retrieved_database_y', 'recall@1', 'recall@1%']].to_numpy()
    # get the coordinates of the real database images
    mapReal = df[['real_database_x', 'real_database_y']].to_numpy()
    



    for t in range(len(mapTest)):
        # si el recall@1 es 1, el color es verde
        if mapTest[t][4] == 1:
            if firstk1:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green', label='Recall@1 prediction')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='green')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='green')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 1, el color es amarillo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 1:
            if firstErrork:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange', label='Recall@1% prediction')
                firstErrork = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='orange')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='orange')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        # si el recall@1 es 0 y el recall@1% es 0, el color es rojo
        elif mapTest[t][4] == 0 and mapTest[t][5] == 0:
            if firstErrorRoom:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red', label='Predictions not among Recall@1 and Recall@1%')
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][0], mapTest[t][1], color='red')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color='red')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)', fontsize=18)
    plt.xlabel('x (m)', fontsize=18)
    plt.title('Pseudo-LiDAR PR', fontsize=24)
    plt.legend(fontsize=14)
    plt.grid()
    # save the figure in the same folder as the csv file
    plt.show()
    print('Figure saved in: {}'.format(os.path.join(os.path.dirname(df['query_image'][0]), 'map.png')))

def get_pointcloud_image(pcd_file_path, dst_file_path):

    
    # Configuramos PyVista para renderizado sin pantalla
    pv.OFF_SCREEN = True
    pv.start_xvfb(wait=0.1)  # Inicia un servidor X virtual
    
    # Cargamos la nube de puntos con Open3D
    # pcd = o3d.io.read_point_cloud(pcd_file_path)
    # read the pointcloud file in .bin format
    
     # Rotar la nube de puntos 90 grados alrededor del eje Z (sentido horario)
    points = np.fromfile(pcd_file_path, dtype=np.float64)
    points = np.float32(points)
    # coords are within -1..1 range in each dimension
    points = np.reshape(points, (points.shape[0] // 3, 3))
 
    # Convertimos la nube de Open3D a un formato que PyVista pueda usar
   
    
    # Creamos la escena PyVista
    plotter = pv.Plotter(off_screen=True)
    
    # Añadimos los puntos
    point_cloud = pv.PolyData(points)
    # colour point cloud with the elevation of the points
    point_cloud['Elevation'] = point_cloud.points[:, 2]
    plotter.add_mesh(point_cloud, scalars='Elevation', show_scalar_bar=False, render_points_as_spheres=False, point_size=5)    

    # plotter.add_points(point_cloud, render_points_as_spheres=False, point_size=5, rgb=True)
    # else:
    #     plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=3)
    
    zoom = 1.3

    plotter.camera.zoom(zoom)
    # get the parent directory of the file
    parent_dir = os.path.dirname(dst_file_path)
    # Creamos el directorio de destino
    os.makedirs(parent_dir, exist_ok=True)
    
    # Guardamos la imagen
    plotter.screenshot(dst_file_path, window_size=(1280, 820))
    
    print(f"Imagen guardada en: {dst_file_path}")
    pcd_image = mpimg.imread(dst_file_path)
    
    # Opcionalmente, recortar la imagen para centrar en la parte relevante
    # (ajustar estos valores según sea necesario)
    height, width, _ = pcd_image.shape
    crop_top = int(height * 0.1)     # Recortar 18% desde arriba
    crop_bottom = int(height * 0.05)   # Recortar 10% desde abajo
    crop_sides = int(width * 0.1)    # Recortar 5% de los lados
    
    # Aplicar recorte
    pcd_image = pcd_image[crop_top:(height-crop_bottom), crop_sides:(width-crop_sides), :]
    # Liberar recursos
    plotter.close()
    
    return pcd_image

def plot_pcds_and_positions(df, df_database, dst_images_dir, output_path, p):
    """
    Plots the query PCD, retrieved database PCD, real database PCD, and their positions on the map.
    """
    k = 2
    i = 0
    for index, row in df.iterrows():
        if i % k == 0:
            # check if the output figure already exists
            # output_file_path = os.path.join(output_path, f'{i}.jpeg')
            # if os.path.exists(output_file_path):
            #      print(f"Figure already exists for index: {index}, skipping...")
            #      continue
            # Print the index being processed
            print(f"Processing index: {i}")
            # Load PCD files
            query_pcd_path = os.path.join(PARAMS.dataset_folder, row['query_image'])
            retrieved_pcd_path = os.path.join(PARAMS.dataset_folder, row['retrieved_database_image'])
            real_pcd_path = os.path.join(PARAMS.dataset_folder, row['real_database_image'])

           

            query_dst_file_path = dst_images_dir + row['query_image'].replace('.bin', '.jpeg')
            retrieved_dst_file_path = dst_images_dir + row['retrieved_database_image'].replace('.bin', '.jpeg')
            real_dst_file_path =  dst_images_dir + row['real_database_image'].replace('.bin', '.jpeg')
            query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
            retrieved_image = get_pointcloud_image(retrieved_pcd_path, retrieved_dst_file_path)
            real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
            # Create a figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Query vs Database Point Clouds (Index: {i})", fontsize=24)

            # Plot images
            axes[0, 0].imshow(query_image)
            axes[0, 0].set_title("Query Point Cloud", fontsize=20)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(retrieved_image)
            axes[0, 1].set_title("Retrieved Database Point Cloud", fontsize=20)
            axes[0, 1].axis('off')

            axes[1, 0].imshow(real_image)
            axes[1, 0].set_title("Nearest Database Point Cloud", fontsize=20)
            axes[1, 0].axis('off')

            # Plot positions on the map
            axes[1, 1].scatter(df_database['easting']/1000, df_database['northing']/1000, color='blue', label="Database Positions", s=4)
            # dibuja con lineas la zona de test dada por p, que contiene 4 puntos
            for point in p:
                x_min = point[1] - X_WIDTH
                x_max = point[1] + X_WIDTH
                y_min = point[0] - Y_WIDTH
                y_max = point[0] + Y_WIDTH
                # dibuja un rectangulo
                axes[1, 1].add_patch(plt.Rectangle((x_min/1000, y_min/1000), (x_max-x_min)/1000, (y_max-y_min)/1000, fill=False, edgecolor='red', linewidth=2))
            
            # axes[1, 1].scatter(row['query_x'], row['query_y'], color='red', label="Query Position")
            # dibuja una cruz en la posicion de la query
            axes[1, 1].scatter(row['query_x']/1000, row['query_y']/1000, color='red', label="Query Position", marker='x', s=225, linewidths=4)
            axes[1, 1].scatter(row['retrieved_database_x']/1000, row['retrieved_database_y']/1000, color='orange', 
                  marker='o', s=150, label="Retrieved Database Position")
            axes[1, 1].scatter(row['real_database_x']/1000, row['real_database_y']/1000, color='green', 
                  marker='o', s=150, facecolors='none', edgecolors='green', linewidths=4, label="Nearest Database Position")
            axes[1, 1].set_title("Positions on Map", fontsize=20)
            axes[1, 1].set_xlabel("x (km)", fontsize=16)
            axes[1, 1].set_ylabel("y (km)", fontsize=16)
            # Colocar la leyenda fuera del área de la gráfica
            # if 'FRIBURGO_A' in dataset_path:
            # axes[1, 1].legend(fontsize=12)
            
            # elif 'FRIBURGO_B' in dataset_path:
            legend = axes[1, 1].legend(fontsize=12)                
            legend.get_frame().set_linewidth(1)
            dif_n = (df_database['northing']/1000).max() - (df_database['northing']/1000).min()
            dif_e = (df_database['easting']/1000).max() - (df_database['easting']/1000).min()
            dif = np.abs(dif_n - dif_e)
            # modificar los limites de los ejes para que la leyenda no se superponga
            axes[1, 1].set_xlim((df_database['easting']/1000).min() - 0.02, (df_database['easting']/1000).max() + 0.05)
            # calcula la diferencia entre el rango de los ejes

            axes[1, 1].set_ylim((df_database['northing']/1000).min() - 0.3, (df_database['northing']/1000).max() + 0.05)
            # elif 'SAARBRUCKEN_A' in dataset_path:
            #     legend = axes[1, 1].legend(fontsize=12)                
            #     legend.get_frame().set_linewidth(1)
            #     # modificar los limites de los ejes para que la leyenda no se superponga
            #     axes[1, 1].set_xlim(df_database['x'].min() - 5, df_database['x'].max() + 0.5)
            #     axes[1, 1].set_ylim(df_database['y'].min() - 0.5, df_database['y'].max() + 1.0)
            # elif 'SAARBRUCKEN_B' in dataset_path:
            #     legend = axes[1, 1].legend(fontsize=12)                
            #     legend.get_frame().set_linewidth(1)
            #     # modificar los limites de los ejes para que la leyenda no se superponga
            #     axes[1, 1].set_xlim(df_database['x'].min() - 0.5, df_database['x'].max() + 0.5)
            #     axes[1, 1].set_ylim(df_database['y'].min() - 0.5, df_database['y'].max() + 2.0)
    



            plt.subplots_adjust(wspace=0.05, hspace=0.1)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Guardar la figura con mayor calidad
            plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
            plt.close()
        i += 1
            


if __name__ == "__main__":
    print('Dataset root: {}'.format(PARAMS.dataset_folder))
    base_path = PARAMS.dataset_folder

    # For Oxford
    folders = []
    runs_folder = "oxford/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    
    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/",
                                      "pointcloud_locations_20m.csv", P_DICT["oxford"], "oxford")

    # For University Sector
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    uni_index = range(10, 15)
    for index in uni_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["university"], "university")

    # For Residential Area
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    res_index = range(5, 10)
    for index in res_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["residential"], "residential")

    # For Business District
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    bus_index = range(5)
    for index in bus_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["business"], "business")
    