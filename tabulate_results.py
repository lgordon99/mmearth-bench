import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from monitor_sweep import monitor_sweep
from sys import argv
from tqdm import tqdm

models = ['ResNet50', 'ResNet50-ImageNet', 'DINOv2', 'MPMAE', 'MPMAE-MMEarth']
splits = ['Random', 'Geographic']
adaptation_methods = ['lp', 'ft', 'llrd']

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
        ('Random', 'LP'), ('Random', 'FT'),
        ('Random', 'LLRD'), ('Geographic', 'LP'),
        ('Geographic', 'FT'), ('Geographic', 'LLRD')
    ])

    df = pd.DataFrame(results, index=row_idx, columns=models)

    print(df.T)

# Create the table
table_df = create_table(argv[1])
