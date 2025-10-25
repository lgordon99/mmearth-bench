import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import wandb

entity = utils.read_yaml('config-user.yml')['entity']
project = utils.read_yaml('config-user.yml')['project']
tag = 'pi'
runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})
tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
architectures = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'AnySat', 'TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']
# architectures = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'TerraMind', 'CopernicusFM']

def tabulate_results_RQ1_task(task):
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    data = {architecture: {train_percent: np.nan for train_percent in train_percents} for architecture in architectures}

    for architecture in architectures:
        for train_percent in train_percents:
            name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
            run = next((run for run in runs if run.name.startswith(name)), None)
            metric = 'R2' if task != 'species' else 'MAP'
            random_test_metric = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data[architecture][train_percent] = random_test_metric

    # Create DataFrame with architectures as rows and train_percents as columns
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Architecture'
    df.columns.name = 'Train %'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best per column (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    for col in df_disp.columns:
        best = df_disp[col].max(skipna=True)
        eq = df_disp[col].eq(best).fillna(False)
        mask[col] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}\\%}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='l' + 'r'*len(cols))
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} random test {caption_metric} by architecture and training percentage}}\n" +
            f"\\label{{tab:{task}_rq1}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_RQ2_task(task):
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    # Create data structure: {architecture: metric_value}
    data = {architecture: np.nan for architecture in architectures}

    for architecture in architectures:
        name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((run for run in runs if run.name.startswith(name)), None)
        metric = 'R2' if task != 'species' else 'MAP'
        geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan
        data[architecture] = geographic_test_metric

    # Create DataFrame with architectures as rows and single column for geographic test metric
    df = pd.DataFrame.from_dict(data, orient='index', columns=[f'Geographic test {metric}'])
    df.index.name = 'Architecture'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best value (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    best = df_disp.iloc[:, 0].max(skipna=True)
    eq = df_disp.iloc[:, 0].eq(best).fillna(False)
    mask.iloc[:, 0] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='lr')
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} geographic test {caption_metric} by architecture}}\n" +
            f"\\label{{tab:{task}_rq2}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_RQ3_task(task):
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    # Create data structure: {architecture: {metric_type: metric_value}}
    data = {architecture: {'Random test': np.nan, 'Geographic test': np.nan} for architecture in architectures}

    for architecture in architectures:
        name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((run for run in runs if run.name.startswith(name)), None)
        metric = 'R2' if task != 'species' else 'MAP'

        if run:
            random_test_metric = run.summary_metrics.get(f'Random test {metric}', np.nan)
            geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}', np.nan)
        else:
            random_test_metric = np.nan
            geographic_test_metric = np.nan

        data[architecture]['Random test'] = random_test_metric
        data[architecture]['Geographic test'] = geographic_test_metric

    # Create DataFrame with architectures as rows and test types as columns
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Architecture'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best per column (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    for col in df_disp.columns:
        best = df_disp[col].max(skipna=True)
        eq = df_disp[col].eq(best).fillna(False)
        mask[col] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='l' + 'r'*len(cols))
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {caption_metric} by architecture and test split}}\n" +
            f"\\label{{tab:{task}_rq3}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_task(task):
    adaptation_modes = ['FT', 'JT']
    splits = ['Random', 'Geographic']
    metric = 'R2' if task != 'species' else 'MAP'
    data = {split: {mode: {architecture: np.nan for architecture in architectures} for mode in adaptation_modes} for split in splits}

    for architecture in architectures:
        for adaptation_mode in adaptation_modes:
            name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
            run = next((run for run in runs if run.name.startswith(name)), None)
            data['Random'][adaptation_mode][architecture] = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data['Geographic'][adaptation_mode][architecture] = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan

    df = pd.concat({'Random': pd.DataFrame.from_dict(data['Random'], orient='index')[architectures].reindex(adaptation_modes),
                    'Geographic': pd.DataFrame.from_dict(data['Geographic'], orient='index')[architectures].reindex(adaptation_modes)}, axis=0).T

    # bolds the highest number in each column (after rounding to 2 decimals)
    formatted_df = df.copy()

    for col in df.columns:
        col_values = df[col].dropna() # excludes NaNs for max calculation
        formatted_df[col] = "--" if col_values.empty else df[col].apply(lambda x: f"\\textbf{{{x:.2f}}}" if pd.notna(x) and round(x, 2) == col_values.round(2).max() else (f"{x:.2f}" if pd.notna(x) else "--"))

    body = formatted_df.to_latex(index=True,
                                 header=False,
                                 escape=False,
                                 na_rep='--',
                                 column_format='l|cc|cc',
                                 multicolumn=False,
                                 multirow=False)
    header_block = (" & \\multicolumn{2}{c|}{Random} & \\multicolumn{2}{c}{Geographic} \\\\\n") + f"{df.index.name or 'Model'} & " + " & ".join(adaptation_modes * len(splits)) + " \\\\\n" # top header and sub header with vertical divider
    latex = ("\\begin{table}[ht]\n\\centering\n" +
             body.replace("\\toprule\n", "\\toprule\n" + header_block, 1) + # inserts header right after \toprule
             f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {metric}}}\n" +
             f"\\label{{tab:{task}_{metric}}}\n" +
             "\\end{table}\n")

    return latex

def tabulate_tta_results():
    tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    latex = '\n'.join([tabulate_results_task(task) for task in tasks])

    with open('latex.tex', 'w') as file:
        file.write(latex)

def plot_rq1_relative_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures:
            for train_percent in train_percents:
                run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
                random_test_metric = run.summary_metrics.get(metric_name) if run else np.nan

                if random_test_metric is not None and not np.isnan(random_test_metric): # checks if the metric is not None and not NaN
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': random_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate relative performance (delta) using ConvNeXtV2A as baseline
    df_delta = df.copy()

    for task in tasks:
        for train_percent in train_percents:
            baseline_data = df[(df['task'] == task) & (df['architecture'] == 'ConvNeXtV2A') & (df['train_percent'] == train_percent)] # gets the baseline performance for the task and train percentage

            if not baseline_data.empty:
                baseline_metric = baseline_data['metric'].iloc[0] # value of the test metric for the baseline architecture
                mask = (df_delta['task'] == task) & (df_delta['train_percent'] == train_percent)
                df_delta.loc[mask, 'metric'] = df_delta.loc[mask, 'metric'] - baseline_metric # subtracts the baseline metric from the test metric

    # Calculate overall y-axis range across all tasks
    all_metrics = df_delta['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, y_max + y_margin]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df_delta[df_delta['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                architecture_data_sorted = architecture_data.sort_values('train_percent') # sort by train_percent to ensure proper line connection

                # Set style based on architecture
                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture in ['MPMAE', 'Satlas']:
                    color = colors[j]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                    color = colors[j]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = colors[j]
                    linestyle = '--'
                    marker = 'o'  # circle

                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=architecture, markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'Δ {metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    fig.suptitle('Change in Random Test Metric from Randomly Initialized ConvNeXtV2A', fontsize=16, y=0.98) # adds main title

    legend_elements = []
    marker_groups = {
        'o': [], # Circle
        's': [], # Square
        '^': []  # Triangle
    }

    # 1. Group handles by marker type
    for j, architecture in enumerate(architectures):
        if architecture == 'ConvNeXtV2A':
            color = 'black'
            marker = 'o'
        elif architecture in ['MPMAE', 'Satlas']:
            color = colors[j]
            marker = 's'
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            color = colors[j]
            marker = '^'
        else:
            # Fallback for any other architecture
            color = colors[j]
            marker = 'o'

        # Create the Line2D handle
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='-',
                            markersize=8, label=architecture,
                            markerfacecolor=color)

        # Adjust markerfacecolor for the ConvNeXtV2A baseline (black circle)
        if architecture == 'ConvNeXtV2A':
             handle.set_markerfacecolor('black')
             handle.set_markeredgecolor('black')

        # Place the handle into the correct group
        marker_groups[marker].append(handle)

    # Get all handles in the desired display order (o, s, ^)
    group_o = marker_groups.get('o', [])
    group_s = marker_groups.get('s', [])
    group_t = marker_groups.get('^', []) # triangle

    # Check if all architectures were accounted for (optional)
    # total_handles = len(group_o) + len(group_s) + len(group_t)
    # print(f"Total architectures: {len(architectures)}, Total handles: {total_handles}")

    # Combine the groups for easy calculation of total size
    all_groups = [group_o, group_s, group_t]

    # 2. Plot each group as a separate legend call for custom column layout

    # Use a common Anchor point (e.g., center of the figure)
    # The handles need to be grouped logically for the three columns

    # Calculate the number of columns and items
    num_o = len(group_o)
    num_s = len(group_s)
    num_t = len(group_t)

    # The number of legends to plot is the number of non-empty groups
    valid_groups = [group for group in all_groups if group]
    num_valid_groups = len(valid_groups)

    # The total number of legend entries (for figuring out the overall width)
    total_entries = num_o + num_s + num_t

    # We need to compute the `bbox_to_anchor` for each individual legend call
    # A simple way to space them is to compute a relative x-position.

    # Total width of all groups combined in terms of columns (1 column per group)
    # The x-coordinates will be calculated based on the number of groups

    # Calculate the normalized x-position for the center of the figure (0.5)

    # Start anchor position (normalized)
    anchor_x_start = 0.5 - (num_valid_groups / 2) * 0.3 # Rough start position

    current_x = 0.5 - 0.05 * (num_o + num_s + num_t) # Start slightly left of center to allow for full width

    # 3. Plot each legend separately

    # Group 1: Marker 'o' (ConvNeXtV2A and any others)
    leg1 = fig.legend(handles=group_o,
                      loc='lower center',
                      bbox_to_anchor=(current_x + (num_o * 0.04), -0.05), # Adjusted to stack left to right
                      ncol=1,
                      fontsize=10,
                      title='Baseline',
                      title_fontsize=10,
                      frameon=False)

    fig.add_artist(leg1) # Add the first legend to the figure

    current_x += num_o * 0.1 # Move x-anchor for the next group

    # Group 2: Marker 's' (MPMAE, Satlas)
    leg2 = fig.legend(handles=group_s,
                      loc='lower center',
                      bbox_to_anchor=(current_x + (num_s * 0.02), -0.05), # Adjusted position
                      ncol=1,
                      fontsize=10,
                      title='MAE/Self-Supervised',
                      title_fontsize=10,
                      frameon=False)

    fig.add_artist(leg2)

    current_x += num_s * 0.1 # Move x-anchor for the next group

    # Group 3: Marker '^' (TerraMind, CopernicusFM, Multimodal)
    leg3 = fig.legend(handles=group_t,
                      loc='lower center',
                      bbox_to_anchor=(current_x + (num_t * 0.02), -0.05), # Adjusted position
                      ncol=1,
                      fontsize=10,
                      title='Foundation Models',
                      title_fontsize=10,
                      frameon=False)

    fig.add_artist(leg3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1) # makes room for the main title and the multiple legends below
    plt.savefig(f'RQ1_relative_plot_{tag}.png', dpi=300, bbox_inches='tight')

def plot_rq1_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures:
            for train_percent in train_percents:
                run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
                random_test_metric = run.summary_metrics.get(metric_name) if run else np.nan

                if random_test_metric is not None and not np.isnan(random_test_metric): # checks if the metric is not None and not NaN
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': random_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate overall y-axis range across all tasks
    all_metrics = df['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, 1.05]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                architecture_data_sorted = architecture_data.sort_values('train_percent') # sort by train_percent to ensure proper line connection

                # Set style based on architecture
                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture in ['MPMAE', 'Satlas']:
                    color = colors[j]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                    color = colors[j]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = colors[j]
                    linestyle = '--'
                    marker = 'o'  # circle

                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=architecture, markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)

        # Set y-axis range: 0-1 for species, consistent range for others
        if task == 'species':
            ax.set_ylim([0, 1.05])
        else:
            ax.set_ylim(y_range)  # Set consistent y-axis range across other tasks

    fig.suptitle('Random Test Metric Performance by Architecture and Training Data Percentage', fontsize=16, y=0.98) # adds main title

    # Make room for the legend before calling tight_layout
    # Set a generous bottom margin to avoid overlap with legend
    plt.subplots_adjust(top=0.85, bottom=0.4)

    # Apply tight layout to the plots and titles only
    plt.tight_layout()

    # Re-adjust margins after tight_layout to ensure suptitle has space and legend space is reserved
    fig.subplots_adjust(top=0.85, bottom=0.4)

    # ------------------------------------------------
    # Custom Legend Grouped by Marker Type
    # ------------------------------------------------

    marker_groups = {
        'o': [], # Circle: Baseline
        's': [], # Square: MAE/Self-Supervised
        '^': []  # Triangle: Foundation Models
    }

    # 1. Group handles by marker type
    for j, architecture in enumerate(architectures):
        # Determine properties
        if architecture == 'ConvNeXtV2A':
            color = 'black'
            marker = 'o'
        elif architecture in ['MPMAE', 'Satlas']:
            color = colors[j]
            marker = 's'
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            color = colors[j]
            marker = '^'
        else:
            color = colors[j]
            marker = 'o'

        # Create the Line2D handle
        # Set markerfacecolor/markeredgecolor for clarity in the legend
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='-',
                            markersize=8, label=architecture,
                            markerfacecolor=color)

        # Handle the baseline 'o' which should be solid black
        if architecture == 'ConvNeXtV2A':
            handle.set_markerfacecolor('black')
            handle.set_markeredgecolor('black')

        # Use a consistent key for grouping
        if architecture in ['MPMAE', 'Satlas']:
            marker_groups['s'].append(handle)
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            marker_groups['^'].append(handle)
        else: # ConvNeXtV2A and others fall into 'o'
            marker_groups['o'].append(handle)

    # Get all handles
    group_o = marker_groups['o']
    group_s = marker_groups['s']
    group_t = marker_groups['^']

    # Group titles for display
    titles = ['RGB', 'S2', 'Multimodal']

    # 2. Calculate positioning for three horizontally centered columns

    # Set a consistent vertical anchor point near the top of the bottom margin
    anchor_y_top = 0.30

    # Calculate precise centering for three columns with closer spacing
    # Use smaller spacing for tighter grouping
    column_spacing = 0.08

    # 3. Plot each group as a separate legend call (tightly centered)

    # Legend 1 (Marker 'o') - RGB (left)
    leg1 = fig.legend(handles=group_o,
                    loc='upper left',
                    bbox_to_anchor=(0.5 - column_spacing - 0.015, anchor_y_top),
                    ncol=1,
                    fontsize=10,
                    title=titles[0],
                    title_fontsize=10,
                    frameon=False)
    fig.add_artist(leg1)

    # Legend 2 (Marker 's') - S2 (center)
    if group_s:
        leg2 = fig.legend(handles=group_s,
                        loc='upper left',
                        bbox_to_anchor=(0.5 - 0.015, anchor_y_top),
                        ncol=1,
                        fontsize=10,
                        title=titles[1],
                        title_fontsize=10,
                        frameon=False)
        fig.add_artist(leg2)

    # Legend 3 (Marker '^') - Multimodal (right)
    if group_t:
        leg3 = fig.legend(handles=group_t,
                        loc='upper left',
                        bbox_to_anchor=(0.5 + column_spacing - 0.015, anchor_y_top),
                        ncol=1,
                        fontsize=10,
                        title=titles[2],
                        title_fontsize=10,
                        frameon=False)
        fig.add_artist(leg3)

    plt.savefig(f'RQ1_plot_{tag}.png', dpi=300, bbox_inches='tight')

def plot_rq2_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        random_metric_name = f'Random test {metric}'
        geographic_metric_name = f'Geographic test {metric}'

        for architecture in architectures:
            run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
            random_test_metric = run.summary_metrics.get(random_metric_name) if run else np.nan
            geographic_test_metric = run.summary_metrics.get(geographic_metric_name) if run else np.nan

            if random_test_metric is not None and not np.isnan(random_test_metric):
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Random', 'metric': random_test_metric})
            if geographic_test_metric is not None and not np.isnan(geographic_test_metric):
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Geographic', 'metric': geographic_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate overall y-axis range across all tasks
    all_metrics = df['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, y_max + y_margin]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                # Separate random and geographic data
                random_data = architecture_data[architecture_data['split'] == 'Random']
                geographic_data = architecture_data[architecture_data['split'] == 'Geographic']

                # Set style based on architecture
                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture in ['MPMAE', 'Satlas']:
                    color = colors[j]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                    color = colors[j]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = colors[j]
                    linestyle = '--'
                    marker = 'o'  # circle

                # Plot both random and geographic points with connecting line
                random_value = random_data['metric'].iloc[0] if not random_data.empty else None
                geographic_value = geographic_data['metric'].iloc[0] if not geographic_data.empty else None

                if random_value is not None and geographic_value is not None:
                    # Plot both points and connecting line
                    ax.plot([0, 1], [random_value, geographic_value],
                           color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
                    ax.plot(0, random_value, marker, color=color, linestyle=linestyle,
                           markersize=6, linewidth=2, alpha=0.8, label=architecture if i == 0 else "")
                    ax.plot(1, geographic_value, marker, color=color, linestyle=linestyle,
                           markersize=6, linewidth=2, alpha=0.8)
                elif random_value is not None:
                    # Only random data available
                    ax.plot(0, random_value, marker, color=color, linestyle=linestyle,
                           markersize=6, linewidth=2, alpha=0.8, label=architecture if i == 0 else "")
                elif geographic_value is not None:
                    # Only geographic data available
                    ax.plot(1, geographic_value, marker, color=color, linestyle=linestyle,
                           markersize=6, linewidth=2, alpha=0.8, label=architecture if i == 0 else "")

        if i == 2:
            ax.set_xlabel('Test Split')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Random', 'Geographic'])
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    fig.suptitle('Performance Comparison: Random vs Geographic Test Splits', fontsize=16, y=0.98) # adds main title

    # ------------------------------------------------
    # Custom Legend Grouped by Marker Type (same as RQ1)
    # ------------------------------------------------

    marker_groups = {
        'o': [], # Circle: Baseline
        's': [], # Square: MAE/Self-Supervised
        '^': []  # Triangle: Foundation Models
    }

    # 1. Group handles by marker type
    for j, architecture in enumerate(architectures):
        # Determine properties
        if architecture == 'ConvNeXtV2A':
            color = 'black'
            marker = 'o'
        elif architecture in ['MPMAE', 'Satlas']:
            color = colors[j]
            marker = 's'
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            color = colors[j]
            marker = '^'
        else:
            # Fallback for any other architecture
            color = colors[j]
            marker = 'o'

        # Create the Line2D handle
        # Set markerfacecolor/markeredgecolor for clarity in the legend
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='-',
                            markersize=8, label=architecture,
                            markerfacecolor=color)

        # Handle the baseline 'o' which should be solid black
        if architecture == 'ConvNeXtV2A':
            handle.set_markerfacecolor('black')
            handle.set_markeredgecolor('black')

        # Use a consistent key for grouping
        if architecture in ['MPMAE', 'Satlas']:
            marker_groups['s'].append(handle)
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            marker_groups['^'].append(handle)
        else: # ConvNeXtV2A and others fall into 'o'
            marker_groups['o'].append(handle)

    # Get all handles
    group_o = marker_groups['o']
    group_s = marker_groups['s']
    group_t = marker_groups['^']

    # Group titles for display
    titles = ['RGB', 'S2', 'Multimodal']

    # 2. Calculate positioning for three horizontally centered columns

    # Set a consistent vertical anchor point near the top of the bottom margin
    anchor_y_top = 0.30

    # Calculate precise centering for three columns with closer spacing
    # Use smaller spacing for tighter grouping
    column_spacing = 0.08

    # 3. Plot each group as a separate legend call (tightly centered)

    # Legend 1 (Marker 'o') - RGB (left)
    leg1 = fig.legend(handles=group_o,
                    loc='upper left',
                    bbox_to_anchor=(0.5 - column_spacing - 0.015, anchor_y_top),
                    ncol=1,
                    fontsize=10,
                    title=titles[0],
                    title_fontsize=10,
                    frameon=False)
    fig.add_artist(leg1)

    # Legend 2 (Marker 's') - S2 (center)
    if group_s:
        leg2 = fig.legend(handles=group_s,
                        loc='upper left',
                        bbox_to_anchor=(0.5 - 0.015, anchor_y_top),
                        ncol=1,
                        fontsize=10,
                        title=titles[1],
                        title_fontsize=10,
                        frameon=False)
        fig.add_artist(leg2)

    # Legend 3 (Marker '^') - Multimodal (right)
    if group_t:
        leg3 = fig.legend(handles=group_t,
                        loc='upper left',
                        bbox_to_anchor=(0.5 + column_spacing - 0.015, anchor_y_top),
                        ncol=1,
                        fontsize=10,
                        title=titles[2],
                        title_fontsize=10,
                        frameon=False)
        fig.add_artist(leg3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.4) # makes room for the main title and the multiple legends below
    plt.savefig(f'RQ2_plot_{tag}.png', dpi=300, bbox_inches='tight')

def plot_rq3_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    rq3_architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(rq3_architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        random_metric_name = f'Random test {metric}'

        for architecture in rq3_architectures:
            run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
            random_test_metric = run.summary_metrics.get(random_metric_name) if run else np.nan

            if random_test_metric is not None and not np.isnan(random_test_metric):
                # Determine if this is S2 or Multimodal version
                if architecture.endswith('S2'):
                    version = 'S2'
                    base_name = architecture[:-2]  # Remove 'S2' suffix
                else:
                    version = 'Multimodal'
                    base_name = architecture

                all_data.append({'task': task, 'architecture': base_name, 'version': version, 'metric': random_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate overall y-axis range across all tasks
    all_metrics = df['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, y_max + y_margin]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]

        # Plot each base architecture (TerraMind and CopernicusFM)
        for base_arch in ['TerraMind', 'CopernicusFM']:
            arch_data = task_data[task_data['architecture'] == base_arch]

            if not arch_data.empty:
                # Separate S2 and Multimodal data
                s2_data = arch_data[arch_data['version'] == 'S2']
                multimodal_data = arch_data[arch_data['version'] == 'Multimodal']

                # Set style based on architecture
                if base_arch == 'TerraMind':
                    color = 'blue'
                    marker = '^'  # triangle up
                else:  # CopernicusFM
                    color = 'red'
                    marker = 'D'  # diamond

                # Plot both S2 and Multimodal points with connecting line
                s2_value = s2_data['metric'].iloc[0] if not s2_data.empty else None
                multimodal_value = multimodal_data['metric'].iloc[0] if not multimodal_data.empty else None

                if s2_value is not None and multimodal_value is not None:
                    # Plot both points and connecting line
                    ax.plot([0, 1], [s2_value, multimodal_value],
                           color=color, linestyle='-', linewidth=2, alpha=0.8)
                    ax.plot(0, s2_value, marker, color=color, linestyle='-',
                           markersize=6, linewidth=2, alpha=0.8, label=base_arch if i == 0 else "")
                    ax.plot(1, multimodal_value, marker, color=color, linestyle='-',
                           markersize=6, linewidth=2, alpha=0.8)
                elif s2_value is not None:
                    # Only S2 data available
                    ax.plot(0, s2_value, marker, color=color, linestyle='-',
                           markersize=6, linewidth=2, alpha=0.8, label=base_arch if i == 0 else "")
                elif multimodal_value is not None:
                    # Only Multimodal data available
                    ax.plot(1, multimodal_value, marker, color=color, linestyle='-',
                           markersize=6, linewidth=2, alpha=0.8, label=base_arch if i == 0 else "")

        if i == 2:
            ax.set_xlabel('Model Version')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['S2', 'Multimodal'])
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    fig.suptitle('S2 vs Multimodal Performance Comparison (Random Test)', fontsize=16, y=0.98) # adds main title

    # Create simple legend for the two base architectures
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='blue', linestyle='-', markersize=8, label='TerraMind'),
        plt.Line2D([0], [0], marker='D', color='red', linestyle='-', markersize=8, label='CopernicusFM')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # makes room for the main title
    plt.savefig(f'RQ3_plot_{tag}.png', dpi=300, bbox_inches='tight')

def tabulate_results(rq_number):
    with open(f'latex_{tag}_RQ{rq_number}.tex', 'w') as file:
        file.write('\n'.join([globals()[f'tabulate_results_RQ{rq_number}_task'](task) for task in tasks]))

if __name__ == '__main__':
    tabulate_results(1)
    tabulate_results(2)
    tabulate_results(3)
    tabulate_tta_results()
    plot_rq1_performance()
    plot_rq1_relative_performance()
    plot_rq2_performance()
    plot_rq3_performance()
