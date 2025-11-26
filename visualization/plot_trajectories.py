import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants for Oxford test regions
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

P_DICT = {
    "oxford": [P1, P2, P3, P4], 
    "university": [P5, P6, P7], 
    "residential": [P8, P9, P10], 
    "business": []
}

def plot_oxford_trajectories(base_path):
    """Plot Oxford trajectories and test regions"""
    runs_folder = "oxford/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    
    # Oxford sequence indices used for evaluation
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    
    plt.figure(figsize=(10, 10))
    
    # Find global min coordinates to center all trajectories
    min_easting = float('inf')
    min_northing = float('inf')
    
    # First pass to find minimums
    for index in index_list:
        folder = all_folders[index]
        csv_path = os.path.join(base_path, runs_folder, folder, "pointcloud_locations_20m.csv")
        df = pd.read_csv(csv_path)
        min_easting = min(min_easting, df['easting'].min())
        min_northing = min(min_northing, df['northing'].min())
    
    # Second pass to plot centered trajectories
    for index in index_list:
        folder = all_folders[index]
        csv_path = os.path.join(base_path, runs_folder, folder, "pointcloud_locations_20m.csv")
        
        # Read trajectory and center it
        df = pd.read_csv(csv_path)
        df['easting'] = df['easting'] - min_easting
        df['northing'] = df['northing'] - min_northing
        plt.plot(df['easting'], df['northing'], 'b-', linewidth=0.5, alpha=0.5)

    # Plot test regions (also centered)
    for point in P_DICT["oxford"]:
        x = point[1] - min_easting
        y = point[0] - min_northing
        x_min = x - X_WIDTH
        x_max = x + X_WIDTH
        y_min = y - Y_WIDTH
        y_max = y + Y_WIDTH
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), 
                                        x_max-x_min, 
                                        y_max-y_min, 
                                        fill=False, 
                                        edgecolor='red', 
                                        linewidth=2))
    
    plt.title('Oxford Dataset Trajectories (Local Coordinates)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('oxford_trajectories_local.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_inhouse_trajectories(base_path):
    """Plot In-house trajectories and test regions"""
    runs_folder = "inhouse_datasets/"
    all_folders = os.listdir(os.path.join(base_path, runs_folder))
    
    areas = {
        "university": range(10, 15),
        "residential": range(5, 10),
        "business": range(5)
    }
    
    for area, indices in areas.items():
        plt.figure(figsize=(10, 10))
        
        # Find global min coordinates
        min_easting = float('inf')
        min_northing = float('inf')
        
        # First pass to find minimums
        for index in indices:
            folder = all_folders[index]
            csv_path = os.path.join(base_path, runs_folder, folder, "pointcloud_centroids_25.csv")
            df = pd.read_csv(csv_path)
            min_easting = min(min_easting, df['easting'].min())
            min_northing = min(min_northing, df['northing'].min())
            
        # Second pass to plot centered trajectories
        for index in indices:
            folder = all_folders[index]
            csv_path = os.path.join(base_path, runs_folder, folder, "pointcloud_centroids_25.csv")
            
            df = pd.read_csv(csv_path)
            df['easting'] = df['easting'] - min_easting
            df['northing'] = df['northing'] - min_northing
            plt.plot(df['easting'], df['northing'], 'b-', linewidth=0.5, alpha=0.5)
        
        # Plot test regions if defined (also centered)
        if area in P_DICT and P_DICT[area]:
            for point in P_DICT[area]:
                x = point[1] - min_easting
                y = point[0] - min_northing
                x_min = x - X_WIDTH
                x_max = x + X_WIDTH
                y_min = y - Y_WIDTH
                y_max = y + Y_WIDTH
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), 
                                                x_max-x_min, 
                                                y_max-y_min, 
                                                fill=False, 
                                                edgecolor='red', 
                                                linewidth=2))
        
        plt.title(f'{area.capitalize()} Dataset Trajectories (Local Coordinates)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(f'{area}_trajectories_local.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_kitti_trajectory(poses_path):
    """Plot KITTI sequence 00 trajectory"""
    poses = np.loadtxt(os.path.join(poses_path, '00.txt'))
    
    # Extract x,z coordinates and center them
    x = poses[:, 3] - poses[0, 3]  # Center with respect to first pose
    z = poses[:, 11] - poses[0, 11]
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, z, 'b-', linewidth=1)
    plt.title('KITTI Sequence 00 Trajectory (Local Coordinates)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('kitti_00_trajectory_local.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_usyd_trajectories(base_path):
    """Plot USyd trajectories"""
    runs_folder = "weeks/"
    plt.figure(figsize=(10, 10))
    
    # Find global min coordinates
    min_easting = float('inf')
    min_northing = float('inf')
    
    validation_weeks = [1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                       26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 45, 47, 49, 52]
    
    # First pass to find minimums
    for week in validation_weeks:
        csv_path = os.path.join(base_path, runs_folder, f'output_week{week}', 
                               f'pointcloud_locations_5m_week_{str(week).zfill(2)}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            min_easting = min(min_easting, df['easting'].min())
            min_northing = min(min_northing, df['northing'].min())
    
    # Second pass to plot centered trajectories
    for week in validation_weeks:
        csv_path = os.path.join(base_path, runs_folder, f'output_week{week}', 
                               f'pointcloud_locations_5m_week_{str(week).zfill(2)}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['easting'] = df['easting'] - min_easting
            df['northing'] = df['northing'] - min_northing
            plt.plot(df['easting'], df['northing'], 'b-', linewidth=0.5, alpha=0.5, color='blue')
    
    plt.title('USyd Dataset Trajectories (Local Coordinates)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('usyd_trajectories_local.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define paths
    oxford_base = "/media/arvc/DATOS/Juanjo/Datasets/benchmark_datasets/"
    kitti_poses = "/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/poses"
    usyd_base = "/media/arvc/DATOS/Juanjo/Datasets/USyd"
    
    # Plot trajectories for each dataset
    print("Plotting Oxford trajectories...")
    plot_oxford_trajectories(oxford_base)
    
    print("Plotting In-house trajectories...")
    plot_inhouse_trajectories(oxford_base)
    
    print("Plotting KITTI trajectory...")
    plot_kitti_trajectory(kitti_poses)
    
    print("Plotting USyd trajectories...")
    plot_usyd_trajectories(usyd_base)
    
    print("Done! All trajectory plots have been saved.")