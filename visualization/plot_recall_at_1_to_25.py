import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
import random

# fix seed for reproducibility
# random.seed(42)
random.seed(155)

stats_path = '/media/arvc/DATOS1/Juanjo/Datasets/benchmark_datasets/stats/'


plt.figure()
# list all stats files in the directory using kitti name
stats_files = [f for f in os.listdir(stats_path) if f.endswith('.pickle') and 'kitti' in f]
print('Found stats files:', stats_files)

model_colors = {}

def darken_color(color, factor=0.7):
    """Oscurece un color (factor < 1 más oscuro, >1 más claro)."""
    rgb = mcolors.to_rgb(color)
    return tuple([c * factor for c in rgb])

for stats_file in stats_files:
    with open(os.path.join(stats_path, stats_file), 'rb') as f:
        stats = pickle.load(f)
    
    # Extract protocol name from file name
    model_protocol_name = stats_file.split('_eval')[-2] 
    model_name = model_protocol_name.split('_')[0]
    protocol_name = model_protocol_name.split('_')[1]
    print(f'Model: {model_name}')
    print(f'Protocol: {protocol_name}')

    # Asignar color al modelo si no lo tiene aún
    if model_name not in model_colors:
        # Seleccionar un color aleatorio de la paleta de Matplotlib
        model_colors[model_name] = random.choice(list(mcolors.TABLEAU_COLORS.values()))
        if model_name == 'MinkUNeXt':
            model_colors[model_name] = 'tab:orange'  # Fijar color específico para MinkUNet

    # Si es protocolo "refined", usar versión más oscura
    if protocol_name == "refined":
        color = darken_color(model_colors[model_name])
    else:
        color = model_colors[model_name]

        
    # Plot recall curves for each location
    for location, location_stats in stats.items():
        
        recall = location_stats['ave_recall']
        # plt.plot(range(1, 26), recall, marker='o', label=f'{model_name} ({protocol_name})')
        plt.plot(range(1, 26), recall, marker='o', label=f'{model_name} ({protocol_name})', color=color)


# plt.title(f'Recall@N KITTI')
plt.xlabel('N (Number of top database candidates)', fontsize=16)
plt.ylabel('Average Recall@N (%)', fontsize=16)
plt.xticks(range(1, 26))
plt.ylim(60, 100)
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))   # grid mayor cada 5
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))   # ticks menores cada 1

plt.grid(True, which="major", axis="x")  # grid solo en los mayores (cada 5)
plt.grid(True, which="major", axis="y")  # en Y lo dejas normal
plt.grid(True)

# Set font size for x-axis tick labels
plt.tick_params(axis='x', labelsize=14) 
# Set font size for y-axis tick labels
plt.tick_params(axis='y', labelsize=14) 
# plt.savefig(os.path.join(stats_path, f'recall_at_1_to_25_kitti.png'))
plt.legend(fontsize=14)
plt.savefig('recall_at_1_to_25_kitti.png')
plt.close()
        


random.seed(42)
# the same for usyd
plt.figure()
stats_files = [f for f in os.listdir(stats_path) if f.endswith('.pickle') and 'usyd' in f]
print('Found stats files:', stats_files)
for stats_file in stats_files:
    with open(os.path.join(stats_path, stats_file), 'rb') as f:
        stats = pickle.load(f)
    
    # Extract protocol name from file name
    model_protocol_name = stats_file.split('_eval')[-2] 
    model_name = model_protocol_name.split('_')[0]
    protocol_name = model_protocol_name.split('_')[1]
    print(f'Model: {model_name}')
    print(f'Protocol: {protocol_name}')

    # Asignar color al modelo si no lo tiene aún
    if model_name not in model_colors:
        # Seleccionar un color aleatorio de la paleta de Matplotlib
        model_colors[model_name] = random.choice(list(mcolors.TABLEAU_COLORS.values()))
    # Si es protocolo "refined", usar versión más oscura
    if protocol_name == "refined":
        color = darken_color(model_colors[model_name])
    else:
        color = model_colors[model_name]
    
    # Plot recall curves for each location
    for location, location_stats in stats.items():
        
        recall = location_stats['ave_recall']
        # plt.plot(range(1, 26), recall, marker='o', label=f'{model_name} ({protocol_name})')
        plt.plot(range(1, 26), recall, marker='o', label=f'{model_name} ({protocol_name})', color=color)

# plt.title(f'Recall@N USYD')
plt.xlabel('N (Number of top database candidates)', fontsize=16)
plt.ylabel('Average Recall@N (%)', fontsize=16)
plt.xticks(range(1, 26))
plt.ylim(60, 100)
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))   # grid mayor cada 5
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))   # ticks menores cada 1
plt.grid(True)
# Set font size for x-axis tick labels
plt.tick_params(axis='x', labelsize=14) 
# Set font size for y-axis tick labels
plt.tick_params(axis='y', labelsize=14) 
# plt.savefig(os.path.join(stats_path, f'recall_at_1_to_25_usyd.png'))
plt.legend(fontsize=14)
plt.savefig('recall_at_1_to_25_usyd.png')
plt.close()