import os
import cv2
import numpy as np
import pandas as pd
import pyvista as pv
from glob import glob
import tqdm

def get_pointcloud_image(pcd_file_path, dst_file_path):
    """Generate image from pointcloud file"""
    # Configure PyVista for off-screen rendering
    pv.OFF_SCREEN = True
    pv.start_xvfb(wait=0.1)
    
    # Load pointcloud (assuming binary format)
    points = np.fromfile(pcd_file_path, dtype=np.float32)
    # PC in Kitti is of size [num_points, 4] -> x,y,z,reflectance
    points = np.reshape(points, (-1, 4))[:, :3]
    
    # Create PyVista plotter
    plotter = pv.Plotter(off_screen=True)
    point_cloud = pv.PolyData(points)
    
    # Color by elevation
    point_cloud['Elevation'] = point_cloud.points[:, 2]
    plotter.add_mesh(point_cloud, scalars='Elevation', show_scalar_bar=False, 
                    render_points_as_spheres=False, point_size=5)
    
    # Set up camera to look at origin (0,0,0)
    camera = plotter.camera
    # Position camera above and behind the origin
    camera.position = [0, -160, 80]   # Colocamos la cámara detrás y arriba del origen
    camera.focal_point = [0, 0, 0]  # Miramos al origen
    camera.up = [0.0, 0.0, 1.0]     # El eje Z es "arriba"
    camera.zoom(0.8)                 # Ajustamos zoom para ver toda la escena
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
    
    # Save image
    plotter.screenshot(dst_file_path, window_size=(1280, 820))
    plotter.close()
    
    return dst_file_path

def process_kitti_sequence(sequence_path, output_dir, sequence="00"):
    """Process KITTI sequence and generate frames"""
    print(f"Processing KITTI sequence {sequence}...")
    
    # Create output directories
    frames_dir = os.path.join(output_dir, "kitti", sequence, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get all bin files
    bin_files = sorted(glob(os.path.join(sequence_path, "velodyne", "*.bin")))
    
    # Generate frames
    for i, bin_file in enumerate(tqdm.tqdm(bin_files)):
        if i % 60 != 0:
            continue  # Process every 10th frame to reduce total number
        frame_path = os.path.join(frames_dir, f"{i:06d}.jpeg")
        # check if frame already exists
        if not os.path.exists(frame_path):
            get_pointcloud_image(bin_file, frame_path)
        
    
    # Create video from frames
    video_dir = os.path.join(output_dir, "kitti", sequence)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"sequence_{sequence}.mp4")
    frames2video(frames_dir, video_path, fps=1)

def process_usyd_sequence(sequence_path, output_dir, week):
    """Process USyd sequence and generate frames"""
    print(f"Processing USyd week {week}...")
    
    # Create output directories
    frames_dir = os.path.join(output_dir, "usyd", f"week_{week:02d}", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Read locations CSV
    csv_path = os.path.join(sequence_path, f"pointcloud_locations_5m_week_{week:02d}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found for week {week}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Get all bin files and generate frames
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if i % 2 != 0:
                continue  # Process every 10th frame to reduce total number
        # USyd usa timestamps como nombres de archivo
        timestamp = str(int(row['timestamp']))  # Convertimos a int para eliminar decimales
        bin_file = os.path.join(sequence_path, "pointclouds_with_locations_5m", f"{timestamp}.bin")
        
        if os.path.exists(bin_file):
            frame_path = os.path.join(frames_dir, f"{i:06d}.jpeg")
            if not os.path.exists(frame_path):  # Solo generamos si no existe             
                get_pointcloud_image(bin_file, frame_path)
    
    # Create video from frames
    video_dir = os.path.join(output_dir, "usyd", f"week_{week:02d}")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"week_{week:02d}.mp4")
    frames2video(frames_dir, video_path, fps=1)

def frames2video(frames_dir, output_path, fps=10):
    """Convert frames to video"""
    print(f"Converting frames to video: {output_path}")
    
    files = sorted(glob(os.path.join(frames_dir, "*.jpeg")))
    if not files:
        print("No frames found!")
        return
        
    # Read first frame to get dimensions
    img = cv2.imread(files[0])
    height, width = img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for file in tqdm.tqdm(files, desc="Creating video"):
        img = cv2.imread(file)
        out.write(img)
    
    out.release()
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    base_output_dir = "/media/arvc/DATOS/Juanjo/Datasets/benchmark_datasets/pcd_videos"
    
    # Process KITTI sequence 00
    kitti_sequence_path = "/media/arvc/DATOS/Juanjo/Datasets/kitti/dataset/sequences/00"
    process_kitti_sequence(kitti_sequence_path, base_output_dir)
    
    # Process USyd sequences
    usyd_base_path = "/media/arvc/DATOS/Juanjo/Datasets/USyd/weeks"
    validation_weeks = [1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 
                       23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 
                       40, 41, 42, 46, 45, 47, 49, 52]
    
    for week in validation_weeks:
        sequence_path = os.path.join(usyd_base_path, f"output_week{week}")
        if os.path.exists(sequence_path):
            process_usyd_sequence(sequence_path, base_output_dir, week)