import os
import cv2
import numpy as np
from glob import glob


def frames2video(folder, output, fps=30):
    """
    Convierte los frames en una carpeta a un video MP4.
    
    Args:
        folder: Carpeta con los frames
        output: Ruta de salida del video
        fps: Frames por segundo del video
    """
    # get all jpeg files in folder
    files = sorted(glob(os.path.join(folder, '*.jpeg')))
    
    if not files:
        print(f"No JPEG files found in {folder}")
        return

    # get first image to get size
    img = cv2.imread(files[0])
    height, width, _ = img.shape

    # create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # write images to video
    for file in files:
        img = cv2.imread(file)
        out.write(img)

    out.release()
    print(f"Video saved to: {output}")


def process_environment(base_path, environment):
    """
    Procesa todas las secuencias de un entorno específico.
    
    Args:
        base_path: Ruta base del dataset
        environment: Nombre del entorno (oxford, university, etc)
    """
    env_path = os.path.join(base_path, environment)
    if not os.path.exists(env_path):
        print(f"Environment path does not exist: {env_path}")
        return
        
    # Create output directory for videos
    output_base = os.path.join(base_path, "videos", environment)
    os.makedirs(output_base, exist_ok=True)
    
    # Process each sequence in the environment
    sequences = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
    
    for sequence in sequences:
        # Path to pointcloud_20m folder containing the frames
        frames_path = os.path.join(env_path, sequence, "pointcloud_25m_25")
        if not os.path.exists(frames_path):
            print(f"Skipping {sequence} - no pointcloud_20m folder found")
            continue
            
        # Output video path
        video_name = f"{environment}_{sequence}.mp4"
        video_path = os.path.join(output_base, video_name)
        
        print(f"Processing sequence: {sequence}")
        frames2video(frames_path, video_path, fps=1)


if __name__ == '__main__':
    # Base path to the dataset
    base_path = "/media/arvc/DATOS/Juanjo/Datasets/benchmark_datasets/pcd_images_minkunext"
    
    # List of environments to process
    environments = ['oxford', 'inhouse_datasets']
    
    for environment in environments:
        print(f"\nProcessing environment: {environment}")
        process_environment(base_path, environment)
        
    print("\nAll videos generated successfully!")




