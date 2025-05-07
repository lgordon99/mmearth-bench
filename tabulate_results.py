import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from monitor_sweep import monitor_sweep
from tqdm import tqdm

models = ['ResNet50', 'ResNet50-ImageNet', 'DINOv2', 'MPMAE', 'MPMAE-MMEarth']
splits = ['Random', 'Geographic']
adaptation_methods = ['ft', 'lp', 'llrd']

def create_table(task):
    results = np.full((len(splits) * len(adaptation_methods), len(models)), np.nan)

    for i, model in enumerate(tqdm(models, desc='Models', position=0)):
        for j, adaptation_method in enumerate(tqdm(adaptation_methods, desc='Adaptation methods', position=1, leave=False)):
            model_name = model.lower().replace('-', '_')
            random_test_rmse, geographic_test_rmse = monitor_sweep(name=f'{task}_{model_name}_{adaptation_method}')
            results[j, i] = random_test_rmse
            results[j + len(adaptation_methods), i] = geographic_test_rmse

    # Create index and columns that match your table
    row_idx = pd.MultiIndex.from_tuples([
        ('Random', 'FT'), ('Random', 'LP'),
        ('Random', 'LLRD'), ('Geographic', 'FT'),
        ('Geographic', 'LP'), ('Geographic', 'LLRD')
    ])

    df = pd.DataFrame(results, index=row_idx, columns=models)

    # Display the table
    print(df)
    exit()
    # Create a visually appealing table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')

    # Create the table with proper styling
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Find the minimum value in each row and highlight it
    for i, row in enumerate(data):
        min_idx = row.index(min(row))
        table[(i, min_idx)].set_facecolor('#aaffaa')  # Light green

    # Add title
    plt.title('Soil Nitrogen (Test RMSE)', fontsize=14, pad=20)

    # Save the figure
    plt.savefig('soil_nitrogen_rmse_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Return the DataFrame for any further processing
    return df

# Create the table
table_df = create_table('soil_nitrogen')
