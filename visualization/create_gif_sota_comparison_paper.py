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
        if folder != '2014-11-18-13-20-12' and folder != 'business_run1' and folder != 'university_run5' and folder != 'residential_run4':
            continue
        print(folder)

        if folder == '2014-11-18-13-20-12':
            fixed_index = 36
        elif folder == 'business_run1':
            fixed_index = 88
        elif folder == 'university_run5':
            fixed_index = 76
        elif folder == 'residential_run4':
            fixed_index = 14
        else:
            fixed_index = 0

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

        csv_file_minkunext = PARAMS.dataset_folder + '/VISUAL_RESULTS_MINKUNEXT_REFINED/' + runs_folder + folder + '.csv'
        csv_file_minkloc3dv2 = PARAMS.dataset_folder + '/VISUAL_RESULTS_MINKLOC3DV2_REFINED/' + runs_folder + folder + '.csv'
        csv_file_casspr = PARAMS.dataset_folder + '/VISUAL_RESULTS_CASSPR_REFINED/' + runs_folder + folder + '.csv'
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        df_results_minkunext = pd.read_csv(csv_file_minkunext)
        df_results_minkloc3dv2 = pd.read_csv(csv_file_minkloc3dv2)
        df_results_casspr = pd.read_csv(csv_file_casspr)

        # create the destination folder to save pcd images
        dst_images_dir_minkunext = PARAMS.dataset_folder + '/pcd_images_minkunext/'
        dst_images_dir_minkloc3dv2 = PARAMS.dataset_folder + '/pcd_images_minkloc3dv2/'
        dst_images_dir_casspr = PARAMS.dataset_folder + '/pcd_images_casspr/'
        results_dir = PARAMS.dataset_folder + '/SOTA_VISUAL_RESULTS_paper/' + runs_folder + folder + '/' 
        if not os.path.exists(dst_images_dir_minkunext):
            os.makedirs(dst_images_dir_minkunext)
        if not os.path.exists(dst_images_dir_minkloc3dv2):
            os.makedirs(dst_images_dir_minkloc3dv2)
        if not os.path.exists(dst_images_dir_casspr):
            os.makedirs(dst_images_dir_casspr)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
 
        # get the pointcloud positions
        plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, results_dir, p, fixed_index)

        

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




# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns
#             fig, axes = plt.subplots(1, 6, figsize=(24, 4))
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=20)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=14)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=14)
#             axes[1].axis('off')
            
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=14)
#             axes[2].axis('off')
            
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=14)
#             axes[3].axis('off')
            
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=14)
#             axes[4].axis('off')
            
#             # Plot positions on the map
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', label="Database Positions", s=4, alpha=0.6)
            
#             # Draw test zones
#             for point in p:
#                 x_min = point[1] - X_WIDTH
#                 x_max = point[1] + X_WIDTH
#                 y_min = point[0] - Y_WIDTH
#                 y_max = point[0] + Y_WIDTH
#                 axes[5].add_patch(plt.Rectangle((x_min/1000, y_min/1000), (x_max-x_min)/1000, (y_max-y_min)/1000, 
#                                               fill=False, edgecolor='red', linewidth=2, alpha=0.7))
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225, linewidths=4)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=100, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=100, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=100, label="CASSPR Retrieved")
            
#             axes[5].set_title("Positions on Map", fontsize=14)
#             axes[5].set_xlabel("x (km)", fontsize=12)
#             axes[5].set_ylabel("y (km)", fontsize=12)
            
#             # Configure legend and limits
#             legend = axes[5].legend(fontsize=10, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             # Set axis limits
#             axes[5].set_xlim((df_database['easting']/1000).min() - 0.02, (df_database['easting']/1000).max() + 0.05)
#             axes[5].set_ylim((df_database['northing']/1000).min() - 0.3, (df_database['northing']/1000).max() + 0.05)
            
#             # Adjust layout
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1
    


# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, making the map larger
#             fig, axes = plt.subplots(1, 6, figsize=(28, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.8]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=22)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=14)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=14)
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot positions on the map
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', label="Database Positions", s=4, alpha=0.6)
            
#             # Draw test zones
#             for point in p:
#                 x_min = point[1] - X_WIDTH
#                 x_max = point[1] + X_WIDTH
#                 y_min = point[0] - Y_WIDTH
#                 y_max = point[0] + Y_WIDTH
#                 axes[5].add_patch(plt.Rectangle((x_min/1000, y_min/1000), (x_max-x_min)/1000, (y_max-y_min)/1000, 
#                                               fill=False, edgecolor='red', linewidth=2, alpha=0.7))
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225, linewidths=4)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=100, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=100, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=100, label="CASSPR Retrieved")
            
#             axes[5].set_title("Positions on Map", fontsize=14)
#             axes[5].set_xlabel("x (km)", fontsize=12)
#             axes[5].set_ylabel("y (km)", fontsize=12)
            
#             # Configure legend and limits
#             legend = axes[5].legend(fontsize=10, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             # Set axis limits
#             axes[5].set_xlim((df_database['easting']/1000).min() - 0.02, (df_database['easting']/1000).max() + 0.05)
#             axes[5].set_ylim((df_database['northing']/1000).min() - 0.3, (df_database['northing']/1000).max() + 0.05)
            
#             # Adjust layout
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1



# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, making the map larger
#             fig, axes = plt.subplots(1, 6, figsize=(28, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.8]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=22)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=14)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=14)
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot zoomed map focusing on the relevant points
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', s=50, alpha=0.6, label="Database Positions")
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225*2, linewidths=4*2)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150*2, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=150*2, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=150*2, label="CASSPR Retrieved")
            
#             axes[5].set_title("Zoomed Map", fontsize=14)
#             axes[5].set_xlabel("x (km)", fontsize=12)
#             axes[5].set_ylabel("y (km)", fontsize=12)
            
#             # Calculate zoom limits to include all relevant points
#             x_points = np.array([
#                 row_minkunext['query_x']/1000,
#                 row_minkunext['retrieved_database_x']/1000,
#                 row_minkloc3dv2['retrieved_database_x']/1000,
#                 row_casspr['retrieved_database_x']/1000,
#                 row_minkunext['real_database_x']/1000
#             ])
#             y_points = np.array([
#                 row_minkunext['query_y']/1000,
#                 row_minkunext['retrieved_database_y']/1000,
#                 row_minkloc3dv2['retrieved_database_y']/1000,
#                 row_casspr['retrieved_database_y']/1000,
#                 row_minkunext['real_database_y']/1000
#             ])
#             margin = 0.02  # margin in km
#             x_min, x_max = x_points.min() - margin, x_points.max() + margin
#             y_min, y_max = y_points.min() - margin, y_points.max() + margin
#             axes[5].set_xlim(x_min, x_max)
#             axes[5].set_ylim(y_min, y_max)
            
#             # Configure legend
#             legend = axes[5].legend(fontsize=10, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1


# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, making the map larger
#             fig, axes = plt.subplots(1, 6, figsize=(28, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.8]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=22)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=14)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=14)
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=14)
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot zoomed map focusing on the relevant points
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', s=50, alpha=0.6, label="Database Positions")
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225*2, linewidths=4*2)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150*2, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=150*2, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=150*2, label="CASSPR Retrieved")
            
#             axes[5].set_title("Zoomed Map", fontsize=14)
            
#             # Calculate zoom limits to include all relevant points
#             x_points = np.array([
#                 row_minkunext['query_x']/1000,
#                 row_minkunext['retrieved_database_x']/1000,
#                 row_minkloc3dv2['retrieved_database_x']/1000,
#                 row_casspr['retrieved_database_x']/1000,
#                 row_minkunext['real_database_x']/1000
#             ])
#             y_points = np.array([
#                 row_minkunext['query_y']/1000,
#                 row_minkunext['retrieved_database_y']/1000,
#                 row_minkloc3dv2['retrieved_database_y']/1000,
#                 row_casspr['retrieved_database_y']/1000,
#                 row_minkunext['real_database_y']/1000
#             ])
#             margin = 0.02  # margin in km
#             x_min, x_max = x_points.min() - margin, x_points.max() + margin
#             y_min, y_max = y_points.min() - margin, y_points.max() + margin
#             axes[5].set_xlim(x_min, x_max)
#             axes[5].set_ylim(y_min, y_max)
            
#             # Configure axes to show relative distances in meters from bottom-left corner
#             # Get current tick locations
#             x_ticks = axes[5].get_xticks()
#             y_ticks = axes[5].get_yticks()
            
#             # Create relative tick labels in meters (from bottom-left origin)
#             x_labels = [f"{int((tick - x_min) * 1000)}" for tick in x_ticks if x_min <= tick <= x_max]
#             y_labels = [f"{int((tick - y_min) * 1000)}" for tick in y_ticks if y_min <= tick <= y_max]
            
#             # Filter ticks to only those within bounds
#             x_ticks_filtered = [tick for tick in x_ticks if x_min <= tick <= x_max]
#             y_ticks_filtered = [tick for tick in y_ticks if y_min <= tick <= y_max]
            
#             axes[5].set_xticks(x_ticks_filtered)
#             axes[5].set_yticks(y_ticks_filtered)
#             axes[5].set_xticklabels(x_labels)
#             axes[5].set_yticklabels(y_labels)
#             axes[5].set_xlabel("Relative Distance (m)", fontsize=12)
#             axes[5].set_ylabel("Relative Distance (m)", fontsize=12)
            
#             # Configure legend
#             legend = axes[5].legend(fontsize=10, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1



# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     # Font size parameters - adjust these to control all text sizes
#     TITLE_FONT_SIZE = 20
#     SUBTITLE_FONT_SIZE = 14
#     AXIS_LABEL_FONT_SIZE = 12
#     LEGEND_FONT_SIZE = 14
    
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, making the map more square
#             fig, axes = plt.subplots(1, 6, figsize=(30, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 2.2]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=TITLE_FONT_SIZE)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot zoomed map focusing on the relevant points
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', s=50, alpha=0.6, label="Database Positions")
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225*2, linewidths=4*2)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150*2, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=150*2, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=150*2, label="CASSPR Retrieved")
            
#             axes[5].set_title("Zoomed Map", fontsize=SUBTITLE_FONT_SIZE)
            
#             # Calculate zoom limits to include all relevant points
#             x_points = np.array([
#                 row_minkunext['query_x']/1000,
#                 row_minkunext['retrieved_database_x']/1000,
#                 row_minkloc3dv2['retrieved_database_x']/1000,
#                 row_casspr['retrieved_database_x']/1000,
#                 row_minkunext['real_database_x']/1000
#             ])
#             y_points = np.array([
#                 row_minkunext['query_y']/1000,
#                 row_minkunext['retrieved_database_y']/1000,
#                 row_minkloc3dv2['retrieved_database_y']/1000,
#                 row_casspr['retrieved_database_y']/1000,
#                 row_minkunext['real_database_y']/1000
#             ])
#             margin = 0.02  # margin in km
#             x_min, x_max = x_points.min() - margin, x_points.max() + margin
#             y_min, y_max = y_points.min() - margin, y_points.max() + margin
#             axes[5].set_xlim(x_min, x_max)
#             axes[5].set_ylim(y_min, y_max)
            
#             # Configure axes to show relative distances in meters from bottom-left corner
#             # Get current tick locations
#             x_ticks = axes[5].get_xticks()
#             y_ticks = axes[5].get_yticks()
            
#             # Create relative tick labels in meters (from bottom-left origin)
#             x_labels = [f"{int((tick - x_min) * 1000)}" for tick in x_ticks if x_min <= tick <= x_max]
#             y_labels = [f"{int((tick - y_min) * 1000)}" for tick in y_ticks if y_min <= tick <= y_max]
            
#             # Filter ticks to only those within bounds
#             x_ticks_filtered = [tick for tick in x_ticks if x_min <= tick <= x_max]
#             y_ticks_filtered = [tick for tick in y_ticks if y_min <= tick <= y_max]
            
#             axes[5].set_xticks(x_ticks_filtered)
#             axes[5].set_yticks(y_ticks_filtered)
#             axes[5].set_xticklabels(x_labels)
#             axes[5].set_yticklabels(y_labels)
#             axes[5].set_xlabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
#             axes[5].set_ylabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
            
#             # Configure legend with larger font
#             legend = axes[5].legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1


# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     # Font size parameters - adjust these to control all text sizes
#     TITLE_FONT_SIZE = 20
#     SUBTITLE_FONT_SIZE = 14
#     AXIS_LABEL_FONT_SIZE = 12
#     LEGEND_FONT_SIZE = 14
    
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, reducing map size
#             fig, axes = plt.subplots(1, 6, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.6]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=TITLE_FONT_SIZE)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("Query Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("Nearest Database\nPoint Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("CASSPR Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE)
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot zoomed map focusing on the relevant points
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', s=50, alpha=0.6, label="Database Positions")
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225*2, linewidths=4*2)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150*2, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                     color='purple', marker='s', s=150*2, label="MinkLoc3D Retrieved")
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")           
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=150*2, label="CASSPR Retrieved")
            
#             axes[5].set_title("Zoomed Map", fontsize=SUBTITLE_FONT_SIZE)
            
#             # Calculate zoom limits to include all relevant points
#             x_points = np.array([
#                 row_minkunext['query_x']/1000,
#                 row_minkunext['retrieved_database_x']/1000,
#                 row_minkloc3dv2['retrieved_database_x']/1000,
#                 row_casspr['retrieved_database_x']/1000,
#                 row_minkunext['real_database_x']/1000
#             ])
#             y_points = np.array([
#                 row_minkunext['query_y']/1000,
#                 row_minkunext['retrieved_database_y']/1000,
#                 row_minkloc3dv2['retrieved_database_y']/1000,
#                 row_casspr['retrieved_database_y']/1000,
#                 row_minkunext['real_database_y']/1000
#             ])
#             margin = 0.01  # margin in km
#             x_min, x_max = x_points.min() - margin, x_points.max() + margin
#             y_min, y_max = y_points.min() - margin, y_points.max() + margin
#             axes[5].set_xlim(x_min, x_max)
#             axes[5].set_ylim(y_min, y_max)
            
#             # Configure axes to show relative distances in meters from bottom-left corner
#             # Get current tick locations
#             x_ticks = axes[5].get_xticks()
#             y_ticks = axes[5].get_yticks()
            
#             # Create relative tick labels in meters (from bottom-left origin)
#             x_labels = [f"{int((tick - x_min) * 1000)}" for tick in x_ticks if x_min <= tick <= x_max]
#             y_labels = [f"{int((tick - y_min) * 1000)}" for tick in y_ticks if y_min <= tick <= y_max]
            
#             # Filter ticks to only those within bounds
#             x_ticks_filtered = [tick for tick in x_ticks if x_min <= tick <= x_max]
#             y_ticks_filtered = [tick for tick in y_ticks if y_min <= tick <= y_max]
            
#             axes[5].set_xticks(x_ticks_filtered)
#             axes[5].set_yticks(y_ticks_filtered)
#             axes[5].set_xticklabels(x_labels)
#             axes[5].set_yticklabels(y_labels)
#             axes[5].set_xlabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
#             axes[5].set_ylabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
            
#             # Configure legend with larger font
#             legend = axes[5].legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')                
#             legend.get_frame().set_linewidth(1)
            
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1


# def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p):
#     """
#     Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
#     """
#     # Font size parameters - adjust these to control all text sizes
#     TITLE_FONT_SIZE = 20
#     SUBTITLE_FONT_SIZE = 14
#     AXIS_LABEL_FONT_SIZE = 12
#     LEGEND_FONT_SIZE = 14
    
#     k = 2
#     i = 0
    
#     # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
#     for index, row_minkunext in df_results_minkunext.iterrows():
#         if i % k == 0:
#             # Print the index being processed
#             print(f"Processing index: {i}")
            
#             # Obtener las filas correspondientes de los otros modelos
#             row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
#             row_casspr = df_results_casspr.iloc[index]
            
#             # Load PCD files (query y nearest son comunes para todos los modelos)
#             query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
#             real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
#             # Retrieved PCDs para cada modelo
#             retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
#             retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
#             retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
#             # Generate images for all point clouds
#             query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
#             real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
#             retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
#             retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
#             # Get images
#             query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
#             real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
#             retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
#             retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
#             retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
#             # Create a figure with 1 row and 6 columns, reducing map size
#             fig, axes = plt.subplots(1, 6, figsize=(18, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1.6]})
#             fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=TITLE_FONT_SIZE)
            
#             # Plot images in a single row
#             axes[0].imshow(query_image)
#             axes[0].set_title("✕ Query Point Cloud", fontsize=SUBTITLE_FONT_SIZE, color='red')
#             axes[0].axis('off')
            
#             axes[1].imshow(real_image)
#             axes[1].set_title("○ Nearest Database\nPoint Cloud", fontsize=SUBTITLE_FONT_SIZE, color='green')
#             axes[1].axis('off')
            
#             # MinkUNeXt with colored border based on recall@1
#             axes[2].imshow(retrieved_minkunext_image)
#             axes[2].set_title("● MinkUNeXt Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE, color='orange')
#             border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
#             for spine in axes[2].spines.values():
#                 spine.set_edgecolor(border_color_minkunext)
#                 spine.set_linewidth(6)
#             axes[2].set_xticks([])
#             axes[2].set_yticks([])
            
#             # MinkLoc3D with colored border based on recall@1
#             axes[3].imshow(retrieved_minkloc3dv2_image)
#             axes[3].set_title("■ MinkLoc3D Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE, color='purple')
#             border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
#             for spine in axes[3].spines.values():
#                 spine.set_edgecolor(border_color_minkloc3d)
#                 spine.set_linewidth(6)
#             axes[3].set_xticks([])
#             axes[3].set_yticks([])
            
#             # CASSPR with colored border based on recall@1
#             axes[4].imshow(retrieved_casspr_image)
#             axes[4].set_title("▲ CASSPR Retrieved\nDatabase Point Cloud", fontsize=SUBTITLE_FONT_SIZE, color='brown')
#             border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
#             for spine in axes[4].spines.values():
#                 spine.set_edgecolor(border_color_casspr)
#                 spine.set_linewidth(6)
#             axes[4].set_xticks([])
#             axes[4].set_yticks([])
            
#             # Plot zoomed map focusing on the relevant points
#             axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='lightblue', s=50, alpha=0.6, label="Database Positions")
            
#             # Plot query position (común para todos)
#             axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
#                           color='red', label="Query Position", marker='x', s=225*2, linewidths=4*2)
            
#             # Plot nearest database position (común para todos)
#             axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
#                           color='green', marker='o', s=150*2, facecolors='none', edgecolors='green', 
#                           linewidths=4, label="Nearest Database Position")
            
#             # Plot retrieved positions for each model
#             axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
#                           color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")
            
#             axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
#                           color='purple', marker='s', s=150*2, label="MinkLoc3D Retrieved")
            
#             axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
#                           color='brown', marker='^', s=150*2, label="CASSPR Retrieved")
            
#             axes[5].set_title("Zoomed Map", fontsize=SUBTITLE_FONT_SIZE)
            
#             # Calculate zoom limits to include all relevant points
#             x_points = np.array([
#                 row_minkunext['query_x']/1000,
#                 row_minkunext['retrieved_database_x']/1000,
#                 row_minkloc3dv2['retrieved_database_x']/1000,
#                 row_casspr['retrieved_database_x']/1000,
#                 row_minkunext['real_database_x']/1000
#             ])
#             y_points = np.array([
#                 row_minkunext['query_y']/1000,
#                 row_minkunext['retrieved_database_y']/1000,
#                 row_minkloc3dv2['retrieved_database_y']/1000,
#                 row_casspr['retrieved_database_y']/1000,
#                 row_minkunext['real_database_y']/1000
#             ])
#             margin = 0.02  # margin in km
#             x_min, x_max = x_points.min() - margin, x_points.max() + margin
#             y_min, y_max = y_points.min() - margin, y_points.max() + margin
#             axes[5].set_xlim(x_min, x_max)
#             axes[5].set_ylim(y_min, y_max)
            
#             # Configure axes to show relative distances in meters from bottom-left corner
#             # Get current tick locations
#             x_ticks = axes[5].get_xticks()
#             y_ticks = axes[5].get_yticks()
            
#             # Create relative tick labels in meters (from bottom-left origin)
#             x_labels = [f"{int((tick - x_min) * 1000)}" for tick in x_ticks if x_min <= tick <= x_max]
#             y_labels = [f"{int((tick - y_min) * 1000)}" for tick in y_ticks if y_min <= tick <= y_max]
            
#             # Filter ticks to only those within bounds
#             x_ticks_filtered = [tick for tick in x_ticks if x_min <= tick <= x_max]
#             y_ticks_filtered = [tick for tick in y_ticks if y_min <= tick <= y_max]
            
#             axes[5].set_xticks(x_ticks_filtered)
#             axes[5].set_yticks(y_ticks_filtered)
#             axes[5].set_xticklabels(x_labels)
#             axes[5].set_yticklabels(y_labels)
#             axes[5].set_xlabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
#             axes[5].set_ylabel("Relative Distance (m)", fontsize=AXIS_LABEL_FONT_SIZE)
            
#             # Configure legend with larger font
#             # legend = axes[5].legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')                
#             # legend.get_frame().set_linewidth(1)
            
#             plt.subplots_adjust(wspace=0.02, hspace=0.1)
#             plt.tight_layout(rect=[0, 0, 1, 0.92])
            
#             # Save figure with high quality
#             plt.savefig(os.path.join(output_path, f'{i}.jpeg'), dpi=300, bbox_inches='tight')
#             plt.close()
#         i += 1


def plot_pcds_and_positions(df_results_minkunext, df_results_minkloc3dv2, df_results_casspr, df_database, dst_images_dir_minkunext, dst_images_dir_minkloc3dv2, dst_images_dir_casspr, output_path, p, fix_index):
    """
    Plots the query PCD, retrieved database PCD of each model, the real database PCD, and their positions on the map.
    """
    # Font size parameters - adjust these to control all text sizes
    TITLE_FONT_SIZE = 20
    SUBTITLE_FONT_SIZE = 22
    AXIS_LABEL_FONT_SIZE = 20
    LEGEND_FONT_SIZE = 14
    
    k = 2
    i = 0

    
    # Asumimos que todas las dataframes tienen el mismo número de filas y el mismo orden de queries
    for index, row_minkunext in df_results_minkunext.iterrows():
        if i % k == 0:
            if i != fix_index:
                i += 1
                continue
            # Print the index being processed
            print(f"Processing index: {i}")
            
            # Obtener las filas correspondientes de los otros modelos
            row_minkloc3dv2 = df_results_minkloc3dv2.iloc[index]
            row_casspr = df_results_casspr.iloc[index]
            
            # Load PCD files (query y nearest son comunes para todos los modelos)
            query_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['query_image'])
            real_pcd_path = os.path.join(PARAMS.dataset_folder, row_minkunext['real_database_image'])
            
            # Retrieved PCDs para cada modelo
            retrieved_minkunext_path = os.path.join(PARAMS.dataset_folder, row_minkunext['retrieved_database_image'])
            retrieved_minkloc3dv2_path = os.path.join(PARAMS.dataset_folder, row_minkloc3dv2['retrieved_database_image'])
            retrieved_casspr_path = os.path.join(PARAMS.dataset_folder, row_casspr['retrieved_database_image'])
            
            # Generate images for all point clouds
            query_dst_file_path = dst_images_dir_minkunext + row_minkunext['query_image'].replace('.bin', '.jpeg')
            real_dst_file_path = dst_images_dir_minkunext + row_minkunext['real_database_image'].replace('.bin', '.jpeg')
            
            retrieved_minkunext_dst = dst_images_dir_minkunext + row_minkunext['retrieved_database_image'].replace('.bin', '.jpeg')
            retrieved_minkloc3dv2_dst = dst_images_dir_minkloc3dv2 + row_minkloc3dv2['retrieved_database_image'].replace('.bin', '.jpeg')
            retrieved_casspr_dst = dst_images_dir_casspr + row_casspr['retrieved_database_image'].replace('.bin', '.jpeg')
            
            # Get images
            query_image = get_pointcloud_image(query_pcd_path, query_dst_file_path)
            real_image = get_pointcloud_image(real_pcd_path, real_dst_file_path)
            retrieved_minkunext_image = get_pointcloud_image(retrieved_minkunext_path, retrieved_minkunext_dst)
            retrieved_minkloc3dv2_image = get_pointcloud_image(retrieved_minkloc3dv2_path, retrieved_minkloc3dv2_dst)
            retrieved_casspr_image = get_pointcloud_image(retrieved_casspr_path, retrieved_casspr_dst)
            
            # Create a figure with 1 row and 6 columns, all with same dimensions
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            # fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=TITLE_FONT_SIZE)
            # fig.suptitle(f"SOTA Models Comparison - Query vs Database Point Clouds (Index: {i})", fontsize=TITLE_FONT_SIZE)
            
            # Plot images in a single row
            axes[0].imshow(query_image)
            axes[0].set_title("✕ Query Point Cloud\n", fontsize=SUBTITLE_FONT_SIZE, color='red')
            axes[0].axis('off')
            
            axes[1].imshow(real_image)
            axes[1].set_title("○ Nearest Database\nPoint Cloud\n", fontsize=SUBTITLE_FONT_SIZE, color='green')
            axes[1].axis('off')
            
            # MinkUNeXt with colored border based on recall@1
            axes[2].imshow(retrieved_minkunext_image)
            axes[2].set_title("● MinkUNeXt Retrieved\nDatabase Point Cloud\n", fontsize=SUBTITLE_FONT_SIZE, color='orange')
            border_color_minkunext = 'green' if row_minkunext['recall@1'] == 1 else 'red'
            for spine in axes[2].spines.values():
                spine.set_edgecolor(border_color_minkunext)
                spine.set_linewidth(6)
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            
            # MinkLoc3D with colored border based on recall@1
            axes[3].imshow(retrieved_minkloc3dv2_image)
            axes[3].set_title("■ MinkLoc3Dv2 Retrieved\nDatabase Point Cloud\n", fontsize=SUBTITLE_FONT_SIZE, color='purple')
            border_color_minkloc3d = 'green' if row_minkloc3dv2['recall@1'] == 1 else 'red'
            for spine in axes[3].spines.values():
                spine.set_edgecolor(border_color_minkloc3d)
                spine.set_linewidth(6)
            axes[3].set_xticks([])
            axes[3].set_yticks([])
            
            # CASSPR with colored border based on recall@1
            axes[4].imshow(retrieved_casspr_image)
            axes[4].set_title("▲ CASSPR Retrieved\nDatabase Point Cloud\n", fontsize=SUBTITLE_FONT_SIZE, color='cyan')
            border_color_casspr = 'green' if row_casspr['recall@1'] == 1 else 'red'
            for spine in axes[4].spines.values():
                spine.set_edgecolor(border_color_casspr)
                spine.set_linewidth(6)
            axes[4].set_xticks([])
            axes[4].set_yticks([])
            
            # Plot zoomed map focusing on the relevant points
            axes[5].scatter(df_database['easting']/1000, df_database['northing']/1000, color='black', s=50, alpha=0.6, label="Database Positions")
            
            # Plot query position (común para todos)
            axes[5].scatter(row_minkunext['query_x']/1000, row_minkunext['query_y']/1000, 
                          color='red', label="Query Position", marker='x', s=225*3, linewidths=4*2)
            
            # Plot nearest database position (común para todos)
            axes[5].scatter(row_minkunext['real_database_x']/1000, row_minkunext['real_database_y']/1000, 
                          color='green', marker='o', s=150*6, facecolors='none', edgecolors='green', 
                          linewidths=6, label="Nearest Database Position")
            
            # Plot retrieved positions for each model
            axes[5].scatter(row_minkloc3dv2['retrieved_database_x']/1000, row_minkloc3dv2['retrieved_database_y']/1000, 
                    color='purple', marker='s', s=150*4, label="MinkLoc3D Retrieved")
            
            axes[5].scatter(row_minkunext['retrieved_database_x']/1000, row_minkunext['retrieved_database_y']/1000, 
                          color='orange', marker='o', s=150*2, label="MinkUNeXt Retrieved")       
       
            
            axes[5].scatter(row_casspr['retrieved_database_x']/1000, row_casspr['retrieved_database_y']/1000, 
                          color='cyan', marker='^', s=150, label="CASSPR Retrieved")
            
            axes[5].set_title("● Database positions\n", fontsize=SUBTITLE_FONT_SIZE, color='black')
            
            # Calculate zoom limits to include all relevant points
            x_points = np.array([
                row_minkunext['query_x']/1000,
                row_minkunext['retrieved_database_x']/1000,
                row_minkloc3dv2['retrieved_database_x']/1000,
                row_casspr['retrieved_database_x']/1000,
                row_minkunext['real_database_x']/1000
            ])
            y_points = np.array([
                row_minkunext['query_y']/1000,
                row_minkunext['retrieved_database_y']/1000,
                row_minkloc3dv2['retrieved_database_y']/1000,
                row_casspr['retrieved_database_y']/1000,
                row_minkunext['real_database_y']/1000
            ])
            margin = 0.03  # margin in km
            x_min, x_max = x_points.min() - margin, x_points.max() + margin
            y_min, y_max = y_points.min() - margin, y_points.max() + margin
            axes[5].set_xlim(x_min, x_max)
            axes[5].set_ylim(y_min, y_max)
            
            # Configure axes to show relative distances in meters from bottom-left corner
            # Get current tick locations
            x_ticks = axes[5].get_xticks()
            y_ticks = axes[5].get_yticks()
            
            # Create relative tick labels in meters (from bottom-left origin)
            x_labels = [f"{int((tick - x_min) * 1000)}" for tick in x_ticks if x_min <= tick <= x_max]
            y_labels = [f"{int((tick - y_min) * 1000)}" for tick in y_ticks if y_min <= tick <= y_max]
            
            # Filter ticks to only those within bounds
            x_ticks_filtered = [tick for tick in x_ticks if x_min <= tick <= x_max]
            y_ticks_filtered = [tick for tick in y_ticks if y_min <= tick <= y_max]
            
            axes[5].set_xticks(x_ticks_filtered)
            axes[5].set_yticks(y_ticks_filtered)
            axes[5].set_xticklabels(x_labels)
            axes[5].set_yticklabels(y_labels)
            axes[5].set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONT_SIZE)
            axes[5].set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONT_SIZE)
            
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            # Save figure with high quality
            plt.savefig(os.path.join(output_path, f'{i}.png'), dpi=300, bbox_inches='tight')
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
    