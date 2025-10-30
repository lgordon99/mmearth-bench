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
# architectures = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'AnySat', 'TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']
architectures = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'TerraMind', 'CopernicusFM']

# Create a consistent color mapping for all architectures
ARCHITECTURE_COLORS = {}
colors_list = plt.cm.tab10(np.linspace(0, 1, len(architectures)))

for i, arch in enumerate(architectures):
    ARCHITECTURE_COLORS[arch] = colors_list[i]

# Font size configuration
LEGEND_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 14

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
    # Escape % in headers to avoid LaTeX comments
    header_line = ' & '.join(['\\textbf{Architecture}'] + ['\\textbf{{{}}}'.format(c.replace('%', '\\%')) for c in cols]) + r' \\'
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
    train_percents = [5, 50, 100]
    architectures = ['AnySatS2', 'AnySat', 'TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    # Build a wide table with columns per (split, train%) pair
    # Columns will be: Random 5%, Geographic 5%, Random 50%, Geographic 50%, Random 100%, Geographic 100%
    metric = 'R2' if task != 'species' else 'MAP'
    # Column order grouped by split
    random_cols = [f'Random {tp}%' for tp in train_percents]
    geographic_cols = [f'Geographic {tp}%' for tp in train_percents]
    col_names = random_cols + geographic_cols

    data = {architecture: {col: np.nan for col in col_names} for architecture in architectures}

    for architecture in architectures:
        for tp in train_percents:
            name = '_'.join([task, architecture, adaptation_mode, str(tp)]) + '_'
            run = next((run for run in runs if run.name.startswith(name)), None)
            if run:
                rand_val = run.summary_metrics.get(f'Random test {metric}', np.nan)
                geo_val = run.summary_metrics.get(f'Geographic test {metric}', np.nan)
            else:
                rand_val = np.nan
                geo_val = np.nan

            data[architecture][f'Random {tp}%'] = rand_val
            data[architecture][f'Geographic {tp}%'] = geo_val

    # DataFrame with architectures as rows and the wide columns
    df = pd.DataFrame.from_dict(data, orient='index')[col_names]
    df.index.name = 'Architecture'

    # Round for display and compute highlight mask per column
    display_decimals = 2
    df_disp = df.round(display_decimals)

    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    for col in df_disp.columns:
        best = df_disp[col].max(skipna=True)
        eq = df_disp[col].eq(best).fillna(False)
        mask[col] = eq

    # Format
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # LaTeX table with grouped headers
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='l' + 'r'*len(col_names))
    # Build group header: Random and Geographic each spanning 3 columns
    group_header = (
        '\\textbf{Architecture} & '
        + '\\multicolumn{3}{c}{\\textbf{Random}} & '
        + '\\multicolumn{3}{c}{\\textbf{Geographic}} \\\\\n'
        + '\\multicolumn{1}{c}{} & '
        + ' & '.join(['\\textbf{5\\%}', '\\textbf{50\\%}', '\\textbf{100\\%}',
                       '\\textbf{5\\%}', '\\textbf{50\\%}', '\\textbf{100\\%}'])
        + r' \\'
    )
    # Insert the grouped header without additional cmidrules to avoid extra horizontal lines
    latex = latex.replace('\\toprule', '\\toprule\n' + group_header + '\n', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
             latex +
             f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} {caption_metric} by architecture, split, and training percent}}\n" +
             f"\\label{{tab:{task}_rq3}}\n" +
             "\\end{table}\n")

    return latex

def tabulate_results_task(task):
    adaptation_modes = ['JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    metric = 'R2' if task != 'species' else 'MAP'
    data = {split: {mode: {architecture: np.nan for architecture in architectures} for mode in adaptation_modes} for split in splits}

    for architecture in architectures:
        for adaptation_mode in adaptation_modes:
            name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
            run = next((run for run in runs if run.name.startswith(name)), None)
            data['Random'][adaptation_mode][architecture] = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data['Geographic'][adaptation_mode][architecture] = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan

    def _one_split_table(split_name):
        df = pd.DataFrame.from_dict(data[split_name], orient='index')[architectures].reindex(adaptation_modes).T

        # --- bold highest per column (after rounding to 2 decimals), robust to strings ---
        formatted_df = df.copy()
        for col in df.columns:
            # numeric view for max
            col_numeric = pd.to_numeric(df[col], errors='coerce')
            if col_numeric.dropna().empty:
                formatted_df[col] = "--"
                continue
            col_max_r2 = col_numeric.round(2).max()

            def _fmt_cell(x):
                x_num = pd.to_numeric(x, errors='coerce')
                if pd.isna(x_num):
                    return "--"
                s = f"{x_num:.2f}"
                return f"\\textbf{{{s}}}" if round(float(x_num), 2) == col_max_r2 else s

            formatted_df[col] = df[col].apply(_fmt_cell)

        # --- underline row-wise max (tie-friendly), robust to strings ---
        for idx in df.index:
            row_num = pd.to_numeric(df.loc[idx], errors='coerce')
            if row_num.dropna().empty:
                continue
            row_max = row_num.round(2).max()
            for mode in adaptation_modes:
                x_num = pd.to_numeric(df.at[idx, mode], errors='coerce')
                if pd.isna(x_num):
                    continue
                if round(float(x_num), 2) == row_max:
                    formatted_df.at[idx, mode] = f"\\underline{{{formatted_df.at[idx, mode]}}}"

        num_adaptation_modes = len(adaptation_modes)
        body = formatted_df.to_latex(index=True,
                                     header=False,
                                     escape=False,
                                     na_rep='--',
                                     column_format='l|' + ('c' * num_adaptation_modes),
                                     multicolumn=False,
                                     multirow=False)

        header_block = (f"{df.index.name or 'Model'} & " + " & ".join(adaptation_modes) + " \\\\\n")
        latex = ("\\begin{table}[ht]\n\\centering\n" +
                 body.replace("\\toprule\n", "\\toprule\n" + header_block, 1) +
                 f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} {split_name} test {metric}}}\n" +
                 f"\\label{{tab:{task}_{metric}_{split_name.lower()}}}\n" +
                 "\\end{table}\n")
        return latex

    latex_random = _one_split_table('Random')
    latex_geographic = _one_split_table('Geographic')

    return latex_random + "\n" + latex_geographic

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
            ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'Δ {metric}', fontsize=AXIS_LABEL_FONTSIZE)
        # No per-subplot titles
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    # No figure title

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
                      fontsize=LEGEND_FONTSIZE,
                      title='Baseline',
                      title_fontsize=LEGEND_FONTSIZE,
                      frameon=False)

    fig.add_artist(leg1) # Add the first legend to the figure

    current_x += num_o * 0.1 # Move x-anchor for the next group

    # Group 2: Marker 's' (MPMAE, Satlas)
    leg2 = fig.legend(handles=group_s,
                      loc='lower center',
                      bbox_to_anchor=(current_x + (num_s * 0.02), -0.05), # Adjusted position
                      ncol=1,
                      fontsize=LEGEND_FONTSIZE,
                      title='MAE/Self-Supervised',
                      title_fontsize=LEGEND_FONTSIZE,
                      frameon=False)

    fig.add_artist(leg2)

    current_x += num_s * 0.1 # Move x-anchor for the next group

    # Group 3: Marker '^' (TerraMind, CopernicusFM, Multimodal)
    leg3 = fig.legend(handles=group_t,
                      loc='lower center',
                      bbox_to_anchor=(current_x + (num_t * 0.02), -0.05), # Adjusted position
                      ncol=1,
                      fontsize=LEGEND_FONTSIZE,
                      title='Foundation Models',
                      title_fontsize=LEGEND_FONTSIZE,
                      frameon=False)

    fig.add_artist(leg3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1) # makes room for the main title and the multiple legends below
    plt.savefig(f'RQ1_relative_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'RQ1_relative_plot.pdf', dpi=300, bbox_inches='tight')

def plot_rq1_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size

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

                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture in ['MPMAE', 'Satlas']:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = 'o'  # circle

                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=architecture, markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}', fontsize=AXIS_LABEL_FONTSIZE)
        # No per-subplot titles
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)

        if task == 'species':
            ax.set_ylim([0, 1.05])
        else:
            ax.set_ylim(y_range)  # Set consistent y-axis range across other tasks

    # No figure title
    plt.subplots_adjust(top=0.85, bottom=0.4)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.4)

    marker_groups = {'o': [], 's': [], '^': []}

    for j, architecture in enumerate(architectures):
        if architecture == 'ConvNeXtV2A':
            color = 'black'
            marker = 'o'
        elif architecture in ['MPMAE', 'Satlas']:
            color = ARCHITECTURE_COLORS[architecture]
            marker = 's'
        elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
            color = ARCHITECTURE_COLORS[architecture]
            marker = '^'
        else:
            color = ARCHITECTURE_COLORS[architecture]
            marker = 'o'

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

    group_o = marker_groups['o']
    group_s = marker_groups['s']
    group_t = marker_groups['^']
    titles = ['RGB', 'S2', 'Multimodal']
    anchor_y_top = 0.30
    column_spacing = 0.11

    # Legend 1 (Marker 'o') - RGB (left)
    leg1 = fig.legend(handles=group_o,
                    loc='upper center',
                    bbox_to_anchor=(0.5 - column_spacing, anchor_y_top),
                    ncol=1,
                    fontsize=LEGEND_FONTSIZE,
                    title=titles[0],
                    title_fontsize=LEGEND_FONTSIZE,
                    frameon=False)
    fig.add_artist(leg1)

    # Legend 2 (Marker 's') - S2 (center)
    if group_s:
        leg2 = fig.legend(handles=group_s,
                        loc='upper center',
                        bbox_to_anchor=(0.5, anchor_y_top),
                        ncol=1,
                        fontsize=LEGEND_FONTSIZE,
                        title=titles[1],
                        title_fontsize=LEGEND_FONTSIZE,
                        frameon=False)
        fig.add_artist(leg2)

    # Legend 3 (Marker '^') - Multimodal (right)
    if group_t:
        leg3 = fig.legend(handles=group_t,
                        loc='upper center',
                        bbox_to_anchor=(0.5 + column_spacing, anchor_y_top),
                        ncol=1,
                        fontsize=LEGEND_FONTSIZE,
                        title=titles[2],
                        title_fontsize=LEGEND_FONTSIZE,
                        frameon=False)
        fig.add_artist(leg3)

    plt.savefig(f'RQ1_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'RQ1_plot.pdf', dpi=300, bbox_inches='tight')

def plot_rq2_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    plt.figure(figsize=(10, 6)) # sets figure size

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

    # Stack plots vertically: 5 rows x 1 column
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df['task'] == task]

        # Calculate y-axis range for this specific task
        task_metrics = task_data['metric'].dropna()
        if len(task_metrics) > 0:
            y_min = task_metrics.min()
            y_max = task_metrics.max()
            y_margin = (y_max - y_min) * 0.1  # Add 10% margin
            y_range = [y_min - y_margin, y_max + y_margin]
        else:
            y_range = [0, 1]  # Default range if no data

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
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = ARCHITECTURE_COLORS[architecture]
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

        # Only label x-axis on the bottom subplot
        if i == len(tasks) - 1:
            ax.set_xlabel('Test Split', fontsize=AXIS_LABEL_FONTSIZE)

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}', fontsize=AXIS_LABEL_FONTSIZE)
        # No per-subplot titles
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Random', 'Geographic'])
        # Hide x tick labels for all but bottom
        if i < len(tasks) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    # No figure title

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07)
    plt.savefig(f'RQ2_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'RQ2_plot.pdf', dpi=300, bbox_inches='tight')

def plot_rq3_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    rq3_architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    plt.figure(figsize=(10, 6)) # sets figure size

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

    fig, axes = plt.subplots(1, 5, figsize=(16, 3)) # creates a figure with 1 row and 5 columns, one per task with reduced vertical and horizontal space

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

                # Set style based on architecture - use consistent colors from RQ1/RQ2
                color = ARCHITECTURE_COLORS[base_arch]
                marker = '^'  # triangle up

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
            ax.set_xlabel('Model input', fontsize=AXIS_LABEL_FONTSIZE)

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}', fontsize=AXIS_LABEL_FONTSIZE)
        # No per-subplot titles
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sentinel-2', 'Multimodal'])
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    # No figure title

    # Create simple legend for the two base architectures - use consistent colors from RQ1/RQ2
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color=ARCHITECTURE_COLORS['TerraMind'], linestyle='-', markersize=8, label='TerraMind'),
        plt.Line2D([0], [0], marker='^', color=ARCHITECTURE_COLORS['CopernicusFM'], linestyle='-', markersize=8, label='CopernicusFM')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=LEGEND_FONTSIZE, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    plt.savefig(f'RQ3_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'RQ3_plot.pdf', dpi=300, bbox_inches='tight')

def plot_tta_improvement(test_split='Random'):
    """Plot box plot showing improvement over JT for JT-TTT and JT-TTT-Geo across tasks

    Args:
        test_split: 'Random' or 'Geographic' test split to analyze
    """

    # Collect improvement data for each task and adaptation mode across all architectures
    all_improvements = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'{test_split} test {metric}'

        # Get JT baseline performance for each architecture
        jt_baseline = {}
        for architecture in architectures:
            run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
            run = next((run for run in runs if run.name.startswith(run_name)), None)
            if run:
                jt_baseline[architecture] = run.summary_metrics.get(metric_name)

        # Calculate improvements for JT-TTT and JT-TTT-Geo for each architecture
        for adaptation_mode in ['JT-TTT', 'JT-TTT-Geo']:
            for architecture in architectures:
                if architecture in jt_baseline and jt_baseline[architecture] is not None:
                    run_name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        performance = run.summary_metrics.get(metric_name)
                        if performance is not None and not np.isnan(performance) and not np.isnan(jt_baseline[architecture]):
                            improvement = performance - jt_baseline[architecture]
                            all_improvements.append({
                                'task': task,
                                'adaptation_mode': adaptation_mode,
                                'improvement': improvement
                            })

    df = pd.DataFrame(all_improvements)

    # Prepare data for box plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df['task'] == task]

        # Prepare data for box plot - collect all architectures for each mode
        jt_ttt_data = task_data[task_data['adaptation_mode'] == 'JT-TTT']['improvement'].dropna().tolist()
        jt_ttt_geo_data = task_data[task_data['adaptation_mode'] == 'JT-TTT-Geo']['improvement'].dropna().tolist()

        # Create box plot with two groups
        positions = [1, 2]
        data_to_plot = [jt_ttt_data, jt_ttt_geo_data]

        bp = ax.boxplot(data_to_plot,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6))

        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize the plot
        ax.set_xticks(positions)
        ax.set_xticklabels(['JT-TTT', 'JT-TTT-Geo'], rotation=0)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        task_label = task.replace('_', ' ').capitalize().replace('ph', 'pH')
        # No per-subplot titles

        if i == 0:
            metric = 'R2' if task != 'species' else 'MAP'
            ax.set_ylabel(f'Δ {metric} (vs JT)', fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)

    # No figure title
    plt.tight_layout()
    plt.savefig(f'TTA_improvement_boxplot_{test_split.lower()}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'TTA_improvement_boxplot_{test_split.lower()}.pdf', dpi=300, bbox_inches='tight')

def plot_tta_improvement_by_model():
    """Plot box plot showing improvement over JT for JT-TTT and JT-TTT-Geo across architectures"""

    # Collect improvement data for each architecture and adaptation mode across all tasks
    all_improvements = []

    for architecture in architectures:
        jt_baseline = {}

        # Get JT baseline performance for each task
        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'
            run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
            run = next((run for run in runs if run.name.startswith(run_name)), None)
            if run:
                jt_baseline[task] = run.summary_metrics.get(f'Random test {metric}')

        # Calculate improvements for JT-TTT and JT-TTT-Geo for each task
        for adaptation_mode in ['JT-TTT', 'JT-TTT-Geo']:
            for task in tasks:
                if task in jt_baseline and jt_baseline[task] is not None:
                    metric = 'R2' if task != 'species' else 'MAP'
                    run_name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        performance = run.summary_metrics.get(f'Random test {metric}')
                        if performance is not None and not np.isnan(performance) and not np.isnan(jt_baseline[task]):
                            improvement = performance - jt_baseline[task]
                            all_improvements.append({
                                'architecture': architecture,
                                'adaptation_mode': adaptation_mode,
                                'improvement': improvement
                            })

    df = pd.DataFrame(all_improvements)

    # Prepare data for box plot
    num_architectures = len(architectures)
    fig, axes = plt.subplots(1, num_architectures, figsize=(5 * num_architectures, 5))

    # Handle single subplot case
    if num_architectures == 1:
        axes = [axes]

    for i, architecture in enumerate(architectures):
        ax = axes[i]
        arch_data = df[df['architecture'] == architecture]

        # Prepare data for box plot - collect all tasks for each mode
        jt_ttt_data = arch_data[arch_data['adaptation_mode'] == 'JT-TTT']['improvement'].dropna().tolist()
        jt_ttt_geo_data = arch_data[arch_data['adaptation_mode'] == 'JT-TTT-Geo']['improvement'].dropna().tolist()

        # Create box plot with two groups
        positions = [1, 2]
        data_to_plot = [jt_ttt_data, jt_ttt_geo_data]

        bp = ax.boxplot(data_to_plot,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6))

        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize the plot
        ax.set_xticks(positions)
        ax.set_xticklabels(['JT-TTT', 'JT-TTT-Geo'], rotation=0)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        # No per-subplot titles

        if i == 0:
            ax.set_ylabel(f'Δ R² (vs JT)', fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)

    # No figure title
    plt.tight_layout()
    plt.savefig('TTA_improvement_by_model_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('TTA_improvement_by_model_boxplot.pdf', dpi=300, bbox_inches='tight')

def analyze_rq3_performance_drops():
    """Calculate percentage performance drops when going from Multimodal to S2-only models,
    averaged over models and tasks with uncertainty estimates (RQ3 analysis)."""

    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    rq3_architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    # Collect all performance data
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in rq3_architectures:
            run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
            test_metric = run.summary_metrics.get(metric_name) if run else np.nan

            if test_metric is not None and not np.isnan(test_metric):
                # Determine if this is S2 or Multimodal version
                if architecture.endswith('S2'):
                    version = 'S2'
                    base_name = architecture[:-2]  # Remove 'S2' suffix
                else:
                    version = 'Multimodal'
                    base_name = architecture

                all_data.append({
                    'task': task,
                    'architecture': base_name,
                    'version': version,
                    'metric': test_metric
                })

    df = pd.DataFrame(all_data)

    # Calculate percentage performance drops for each task-architecture combination
    pct_drops_multimodal_to_s2 = []

    for task in tasks:
        for base_arch in ['TerraMind', 'CopernicusFM']:
            task_arch_data = df[(df['task'] == task) & (df['architecture'] == base_arch)]

            if len(task_arch_data) >= 2:  # Need both S2 and Multimodal versions
                multimodal_data = task_arch_data[task_arch_data['version'] == 'Multimodal']
                s2_data = task_arch_data[task_arch_data['version'] == 'S2']

                if not multimodal_data.empty and not s2_data.empty:
                    multimodal_perf = multimodal_data['metric'].iloc[0]
                    s2_perf = s2_data['metric'].iloc[0]

                    if not (np.isnan(multimodal_perf) or np.isnan(s2_perf)) and multimodal_perf > 0:
                        # Calculate percentage drop: (Multimodal - S2) / Multimodal * 100
                        pct_drop = (multimodal_perf - s2_perf) / multimodal_perf * 100
                        pct_drops_multimodal_to_s2.append(pct_drop)

    # Calculate summary statistics with uncertainty
    pct_drops_multimodal_to_s2 = np.array(pct_drops_multimodal_to_s2)

    # Mean and standard error
    mean_pct_drop = np.mean(pct_drops_multimodal_to_s2)
    se_pct_drop = np.std(pct_drops_multimodal_to_s2, ddof=1) / np.sqrt(len(pct_drops_multimodal_to_s2))

    # 95% confidence intervals (using normal approximation for simplicity)
    n = len(pct_drops_multimodal_to_s2)
    z_critical = 1.96  # 95% confidence interval for normal distribution

    ci_lower = mean_pct_drop - z_critical * se_pct_drop
    ci_upper = mean_pct_drop + z_critical * se_pct_drop

    # Print results
    print("\nRQ3 ANALYSIS - MULTIMODAL vs S2-ONLY MODELS")
    print("=" * 60)
    print("RQ3 Performance Drop Analysis (Multimodal → S2-only Models)")
    print("=" * 80)
    print(f"Training data: 100% (full training data)")
    print(f"Test split: Random")
    print(f"Number of task-architecture combinations: {n}")
    print()
    print("Multimodal → S2-only models:")
    print(f"  Mean percentage drop: {mean_pct_drop:.2f}% ± {se_pct_drop:.2f}%")
    print(f"  95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print()
    print("Additional statistics:")
    print(f"  Standard deviation: {np.std(pct_drops_multimodal_to_s2, ddof=1):.2f}%")
    print(f"  Range: [{np.min(pct_drops_multimodal_to_s2):.2f}%, {np.max(pct_drops_multimodal_to_s2):.2f}%]")
    print()

    return {
        'pct_drop_multimodal_to_s2': {'mean': mean_pct_drop, 'se': se_pct_drop, 'ci': (ci_lower, ci_upper)},
        'n_combinations': n
    }

def analyze_rq2_performance_drops():
    """Calculate percentage performance drops when going from Random to Geographic test splits,
    averaged over models and tasks with uncertainty estimates (RQ2 analysis)."""

    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    # Collect all performance data
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        random_metric_name = f'Random test {metric}'
        geographic_metric_name = f'Geographic test {metric}'

        for architecture in architectures:
            run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)

            if run:
                random_metric = run.summary_metrics.get(random_metric_name)
                geographic_metric = run.summary_metrics.get(geographic_metric_name)

                if (random_metric is not None and not np.isnan(random_metric) and
                    geographic_metric is not None and not np.isnan(geographic_metric)):
                    all_data.append({
                        'task': task,
                        'architecture': architecture,
                        'random_metric': random_metric,
                        'geographic_metric': geographic_metric
                    })

    df = pd.DataFrame(all_data)

    # Calculate percentage performance drops for each task-architecture combination
    pct_drops_random_to_geographic = []

    for task in tasks:
        for architecture in architectures:
            task_arch_data = df[(df['task'] == task) & (df['architecture'] == architecture)]

            if not task_arch_data.empty:
                random_perf = task_arch_data['random_metric'].iloc[0]
                geographic_perf = task_arch_data['geographic_metric'].iloc[0]

                if not (np.isnan(random_perf) or np.isnan(geographic_perf)) and random_perf > 0:
                    # Calculate percentage drop: (Random - Geographic) / Random * 100
                    pct_drop = (random_perf - geographic_perf) / random_perf * 100
                    pct_drops_random_to_geographic.append(pct_drop)

    # Calculate summary statistics with uncertainty
    pct_drops_random_to_geographic = np.array(pct_drops_random_to_geographic)

    # Mean and standard error
    mean_pct_drop = np.mean(pct_drops_random_to_geographic)
    se_pct_drop = np.std(pct_drops_random_to_geographic, ddof=1) / np.sqrt(len(pct_drops_random_to_geographic))

    # 95% confidence intervals (using normal approximation for simplicity)
    n = len(pct_drops_random_to_geographic)
    z_critical = 1.96  # 95% confidence interval for normal distribution

    ci_lower = mean_pct_drop - z_critical * se_pct_drop
    ci_upper = mean_pct_drop + z_critical * se_pct_drop

    # Print results
    print("\nRQ2 ANALYSIS - TEST SPLIT COMPARISON")
    print("=" * 60)
    print("RQ2 Performance Drop Analysis (Random → Geographic Test Split)")
    print("=" * 80)
    print(f"Training data: 100% (full training data)")
    print(f"Number of task-architecture combinations: {n}")
    print()
    print("Random → Geographic test split:")
    print(f"  Mean percentage drop: {mean_pct_drop:.2f}% ± {se_pct_drop:.2f}%")
    print(f"  95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print()
    print("Additional statistics:")
    print(f"  Standard deviation: {np.std(pct_drops_random_to_geographic, ddof=1):.2f}%")
    print(f"  Range: [{np.min(pct_drops_random_to_geographic):.2f}%, {np.max(pct_drops_random_to_geographic):.2f}%]")
    print()

    return {
        'pct_drop_random_to_geographic': {'mean': mean_pct_drop, 'se': se_pct_drop, 'ci': (ci_lower, ci_upper)},
        'n_combinations': n
    }

def analyze_performance_drops(test_split='Random'):
    """Calculate percentage performance drops when reducing training data from 100% to 50% and 100% to 5%,
    averaged over models and tasks with uncertainty estimates.

    Args:
        test_split: 'Random' or 'Geographic' test split to analyze
    """

    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    # Collect all performance data
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'{test_split} test {metric}'

        for architecture in architectures:
            for train_percent in train_percents:
                run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
                test_metric = run.summary_metrics.get(metric_name) if run else np.nan

                if test_metric is not None and not np.isnan(test_metric):
                    all_data.append({
                        'task': task,
                        'architecture': architecture,
                        'train_percent': train_percent,
                        'metric': test_metric
                    })

    df = pd.DataFrame(all_data)

    # Calculate percentage performance drops for each task-architecture combination
    # Separate ConvNeXtV2A from other architectures
    pct_drops_100_to_50_convnext = []
    pct_drops_100_to_5_convnext = []
    pct_drops_100_to_50_pretrained = []
    pct_drops_100_to_5_pretrained = []

    for task in tasks:
        for architecture in architectures:
            task_arch_data = df[(df['task'] == task) & (df['architecture'] == architecture)]

            if len(task_arch_data) >= 3:  # Need all three training percentages
                perf_100 = task_arch_data[task_arch_data['train_percent'] == 100]['metric'].iloc[0]
                perf_50 = task_arch_data[task_arch_data['train_percent'] == 50]['metric'].iloc[0]
                perf_5 = task_arch_data[task_arch_data['train_percent'] == 5]['metric'].iloc[0]

                if not (np.isnan(perf_100) or np.isnan(perf_50) or np.isnan(perf_5)) and perf_100 > 0:
                    # Calculate percentage drop: (100% - reduced%) / 100% * 100
                    pct_drop_100_to_50 = (perf_100 - perf_50) / perf_100 * 100
                    pct_drop_100_to_5 = (perf_100 - perf_5) / perf_100 * 100

                    # Separate by architecture type
                    if architecture == 'ConvNeXtV2A':
                        pct_drops_100_to_50_convnext.append(pct_drop_100_to_50)
                        pct_drops_100_to_5_convnext.append(pct_drop_100_to_5)
                    else:
                        pct_drops_100_to_50_pretrained.append(pct_drop_100_to_50)
                        pct_drops_100_to_5_pretrained.append(pct_drop_100_to_5)

    # Calculate summary statistics with uncertainty
    # Helper function to calculate statistics
    def calc_stats(drops):
        drops_arr = np.array(drops)
        if len(drops_arr) == 0:
            return None, None, None, None, 0
        mean = np.mean(drops_arr)
        se = np.std(drops_arr, ddof=1) / np.sqrt(len(drops_arr)) if len(drops_arr) > 1 else 0
        z_critical = 1.96
        ci_lower = mean - z_critical * se
        ci_upper = mean + z_critical * se
        return mean, se, ci_lower, ci_upper, len(drops_arr)

    # ConvNeXtV2A statistics (averaged over tasks)
    mean_50_conv, se_50_conv, ci_50_lower_conv, ci_50_upper_conv, n_50_conv = calc_stats(pct_drops_100_to_50_convnext)
    mean_5_conv, se_5_conv, ci_5_lower_conv, ci_5_upper_conv, n_5_conv = calc_stats(pct_drops_100_to_5_convnext)

    # Pretrained architectures statistics (averaged over tasks and architectures)
    mean_50_pre, se_50_pre, ci_50_lower_pre, ci_50_upper_pre, n_50_pre = calc_stats(pct_drops_100_to_50_pretrained)
    mean_5_pre, se_5_pre, ci_5_lower_pre, ci_5_upper_pre, n_5_pre = calc_stats(pct_drops_100_to_5_pretrained)

    # Print results
    if test_split == 'Random':
        print("\nRQ1 ANALYSIS - TRAINING DATA REDUCTION")
        print("=" * 60)
    print(f"\n{test_split.upper()} TEST SPLIT ANALYSIS")
    print("=" * 50)
    print(f"Percentage Performance Drop Analysis ({test_split} Test Split)")
    print(f"Training data reduction: 100% → 50% and 100% → 5%")
    print()

    print("ConvNeXtV2A (Randomly Initialized Baseline) - Averaged over Tasks:")
    print(f"  Number of tasks: {n_50_conv}")
    print("  100% → 50% training data:")
    if mean_50_conv is not None:
        print(f"    Mean percentage drop: {mean_50_conv:.2f}% ± {se_50_conv:.2f}%")
        print(f"    95% CI: [{ci_50_lower_conv:.2f}%, {ci_50_upper_conv:.2f}%]")
    else:
        print("    No data available")
    print("  100% → 5% training data:")
    if mean_5_conv is not None:
        print(f"    Mean percentage drop: {mean_5_conv:.2f}% ± {se_5_conv:.2f}%")
        print(f"    95% CI: [{ci_5_lower_conv:.2f}%, {ci_5_upper_conv:.2f}%]")
    else:
        print("    No data available")
    print()

    print("Pretrained Models (All Other Architectures) - Averaged over Tasks and Architectures:")
    print(f"  Number of task-architecture combinations: {n_50_pre}")
    print("  100% → 50% training data:")
    if mean_50_pre is not None:
        print(f"    Mean percentage drop: {mean_50_pre:.2f}% ± {se_50_pre:.2f}%")
        print(f"    95% CI: [{ci_50_lower_pre:.2f}%, {ci_50_upper_pre:.2f}%]")
    else:
        print("    No data available")
    print("  100% → 5% training data:")
    if mean_5_pre is not None:
        print(f"    Mean percentage drop: {mean_5_pre:.2f}% ± {se_5_pre:.2f}%")
        print(f"    95% CI: [{ci_5_lower_pre:.2f}%, {ci_5_upper_pre:.2f}%]")
    else:
        print("    No data available")
    print()

    return {
        'convnext': {
            'pct_drop_100_to_50': {'mean': mean_50_conv, 'se': se_50_conv, 'ci': (ci_50_lower_conv, ci_50_upper_conv), 'n': n_50_conv},
            'pct_drop_100_to_5': {'mean': mean_5_conv, 'se': se_5_conv, 'ci': (ci_5_lower_conv, ci_5_upper_conv), 'n': n_5_conv}
        },
        'pretrained': {
            'pct_drop_100_to_50': {'mean': mean_50_pre, 'se': se_50_pre, 'ci': (ci_50_lower_pre, ci_50_upper_pre), 'n': n_50_pre},
            'pct_drop_100_to_5': {'mean': mean_5_pre, 'se': se_5_pre, 'ci': (ci_5_lower_pre, ci_5_upper_pre), 'n': n_5_pre}
        },
        'test_split': test_split
    }

def tabulate_results(rq_number):
    with open(f'latex_{tag}_RQ{rq_number}.tex', 'w') as file:
        file.write('\n'.join([globals()[f'tabulate_results_RQ{rq_number}_task'](task) for task in tasks]))

if __name__ == '__main__':
    tabulate_results(1)
    tabulate_results(2)
    tabulate_results(3)
    tabulate_tta_results()
    plot_rq1_performance()
    # plot_rq1_relative_performance()
    plot_rq2_performance()
    plot_rq3_performance()
    plot_tta_improvement('Random')
    plot_tta_improvement('Geographic')
    plot_tta_improvement_by_model()

    # Analyze performance drops for both test splits
    random_results = analyze_performance_drops('Random')
    geographic_results = analyze_performance_drops('Geographic')
    rq2_results = analyze_rq2_performance_drops()
    rq3_results = analyze_rq3_performance_drops()
