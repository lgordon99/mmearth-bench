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
architectures_plots = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'Satlas', 'MPMAE', 'TerraMind', 'CopernicusFM']
architectures_tables = architectures_plots + ['AnySat']

# Create a consistent color mapping for all architectures
ARCHITECTURE_COLORS = {}
colors_list = plt.cm.tab10(np.linspace(0, 1, len(architectures_plots)))

for i, arch in enumerate(architectures_plots):
    ARCHITECTURE_COLORS[arch] = colors_list[i]

# Display-name mapping for plots/tables (keep 'Satlas' for wandb lookups)
def display_arch_name(name: str) -> str:
    return 'SatlasNet' if name == 'Satlas' else name

# Font size configuration
LEGEND_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 16

def tabulate_results_RQ1_task(task):
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    data = {architecture: {train_percent: np.nan for train_percent in train_percents} for architecture in architectures_tables}

    for architecture in architectures_tables:
        for train_percent in train_percents:
            name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
            run = next((run for run in runs if run.name.startswith(name)), None)
            metric = 'R2' if task != 'species' else 'MAP'
            random_test_metric = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data[architecture][train_percent] = random_test_metric

    # Create DataFrame with architectures as rows and train_percents as columns
    df = pd.DataFrame.from_dict(data, orient='index')
    # Apply display name mapping for row index
    df.index = [display_arch_name(idx) for idx in df.index]
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
    data = {architecture: np.nan for architecture in architectures_tables}

    for architecture in architectures_tables:
        name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((run for run in runs if run.name.startswith(name)), None)
        metric = 'R2' if task != 'species' else 'MAP'
        geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan
        data[architecture] = geographic_test_metric

    # Create DataFrame with architectures as rows and single column for geographic test metric
    df = pd.DataFrame.from_dict(data, orient='index', columns=[f'Geographic test {metric}'])
    # Apply display name mapping for row index
    df.index = [display_arch_name(idx) for idx in df.index]
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
    # Apply display name mapping for row index
    df.index = [display_arch_name(idx) for idx in df.index]
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

def tabulate_TTT_results_task(task):
    adaptation_modes = ['JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    metric = 'R2' if task != 'species' else 'MAP'
    data = {split: {mode: {architecture: np.nan for architecture in architectures_tables} for mode in adaptation_modes} for split in splits}
    # runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['lr1e-2,modgradnorm,max5itmean']}})
    # runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['lr1e-2,5itmean']}})

    for architecture in architectures_tables:
        for adaptation_mode in adaptation_modes:
            name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
            run = next((run for run in runs if run.name.startswith(name)), None)
            data['Random'][adaptation_mode][architecture] = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data['Geographic'][adaptation_mode][architecture] = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan

    def _one_split_table(split_name):
        df = pd.DataFrame.from_dict(data[split_name], orient='index')[architectures_tables].reindex(adaptation_modes).T

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
    latex = '\n'.join([tabulate_TTT_results_task(task) for task in tasks])

    with open('latex.tex', 'w') as file:
        file.write(latex)

def plot_rq1_relative_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures_plots))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures_plots:
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

    fig, axes = plt.subplots(1, 5, figsize=(20, 3)) # slightly reduced height for a more compact figure

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df_delta[df_delta['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures_plots):
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
                        marker, color=color, linestyle=linestyle, label=display_arch_name(architecture), markersize=6, linewidth=2, alpha=0.8)

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
    for j, architecture in enumerate(architectures_plots):
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
                            markersize=8, label=display_arch_name(architecture),
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

        for architecture in architectures_plots:
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

    fig, axes = plt.subplots(1, 5, figsize=(20, 3)) # slightly reduced height for a more compact figure

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures_plots):
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
                        marker, color=color, linestyle=linestyle, label=display_arch_name(architecture), markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}', fontsize=AXIS_LABEL_FONTSIZE)
        # Pull y-axis label slightly left for interior subplots to avoid overlap
        # if i > 0:
        #     ax.yaxis.set_label_coords(-0.08, 0.5)
        # Restore per-subplot titles for RQ1
        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)

        if task == 'species':
            ax.set_ylim([0, 1.05])
        else:
            ax.set_ylim(y_range)  # Set consistent y-axis range across other tasks

    # No figure title
    plt.tight_layout()
    # Increase horizontal spacing further to avoid overlapping y-axis labels between subplots
    # and keep a modest left margin for the legend
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.95, wspace=0.45)

    # Single-column legend to the left of plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(0.02, 0.5),
               ncol=1,
               fontsize=LEGEND_FONTSIZE,
               title='Model',
               title_fontsize=LEGEND_FONTSIZE,
               frameon=False)

    plt.savefig('RQ1_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('RQ1_plot.pdf', dpi=300, bbox_inches='tight')

def plot_rq2_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    plt.figure(figsize=(10, 6))

    # Collect data for all tasks
    all_data = []
    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        random_metric_name = f'Random test {metric}'
        geographic_metric_name = f'Geographic test {metric}'
        for architecture in architectures_plots:
            run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
            random_test_metric = run.summary_metrics.get(random_metric_name) if run else np.nan
            geographic_test_metric = run.summary_metrics.get(geographic_metric_name) if run else np.nan
            if random_test_metric is not None and not np.isnan(random_test_metric):
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Random', 'metric': random_test_metric})
            if geographic_test_metric is not None and not np.isnan(geographic_test_metric):
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Geographic', 'metric': geographic_test_metric})

    df = pd.DataFrame(all_data)

    # 1 row x 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df['task'] == task]

        # y-range for this task
        task_metrics = task_data['metric'].dropna()
        if len(task_metrics) > 0:
            y_min = task_metrics.min()
            y_max = task_metrics.max()
            y_margin = (y_max - y_min) * 0.1
            y_range = [y_min - y_margin, y_max + y_margin]
        else:
            y_range = [0, 1]

        # Plot each architecture
        for j, architecture in enumerate(architectures_plots):
            architecture_data = task_data[task_data['architecture'] == architecture]
            if architecture_data.empty:
                continue
            random_data = architecture_data[architecture_data['split'] == 'Random']
            geographic_data = architecture_data[architecture_data['split'] == 'Geographic']

            if architecture == 'ConvNeXtV2A':
                color = 'black'; linestyle = '-'; marker = 'o'
            elif architecture in ['MPMAE', 'Satlas']:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = 's'
            elif architecture in ['TerraMind', 'CopernicusFM', 'ConvNeXtV2AMultimodal']:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = '^'
            else:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = 'o'

            rv = random_data['metric'].iloc[0] if not random_data.empty else None
            gv = geographic_data['metric'].iloc[0] if not geographic_data.empty else None

            ax.plot([0, 1], [rv, gv], color=color, linestyle=linestyle, linewidth=3, alpha=0.8)
            ax.plot(0, rv, marker, color=color, linestyle=linestyle, markersize=10, linewidth=3, alpha=0.8, label=display_arch_name(architecture) if i == 0 else "")
            ax.plot(1, gv, marker, color=color, linestyle=linestyle, markersize=10, linewidth=3, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Test Split', fontsize=AXIS_LABEL_FONTSIZE+10)

        if i == 0:
            ax.set_ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE+10)

        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE+10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['R', 'G'])
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE+10)
        ax.set_ylim(y_range)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.10, wspace=0.5)
    plt.savefig('RQ2_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('RQ2_plot.pdf', dpi=300, bbox_inches='tight')

def plot_rq3_performance():
    """Create a combined RQ3 plot with two rows (Random, Geographic) and five columns (tasks).
    S2 is solid with square markers; Multimodal is dashed with triangle markers.
    Single legend on the left and single x-axis label at the bottom; task names above each column;
    row titles 'Random' and 'Geographic' above each row of five plots.
    """
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    base_archs = ['TerraMind', 'CopernicusFM']
    variants = {
        'S2': {'suffix': 'S2', 'linestyle': '-', 'marker': 'o', 'label_suffix': 'S2'},
        'Multimodal': {'suffix': '', 'linestyle': '--', 'marker': '^', 'label_suffix': ''}
    }

    # Collect data for both splits
    rows = []
    for split in ['Random', 'Geographic']:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'
            metric_name = f"{split} test {metric}"
            for base in base_archs:
                for variant_name, cfg in variants.items():
                    arch_lookup = base + cfg['suffix']
                    for tp in train_percents:
                        run = next((run for run in runs if run.name.startswith(f"{task}_{arch_lookup}_{adaptation_mode}_{tp}_")), None)
                        val = run.summary_metrics.get(metric_name) if run else np.nan
                        if val is not None and not np.isnan(val):
                            rows.append({
                                'split': split,
                                'task': task,
                                'base': base,
                                'variant': variant_name,
                                'train_percent': tp,
                                'metric': val
                            })

    df = pd.DataFrame(rows)

    # Figure: 2 rows (Random, Geographic) x 5 columns (tasks)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))

    # Set column titles (task names) at top row only
    for j, task in enumerate(tasks):
        axes[0, j].set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE+10)

    # Plot for each split and task
    for row_idx, split in enumerate(['Random', 'Geographic']):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            task_df = df[(df['split'] == split) & (df['task'] == task)]

            # y-range per panel
            task_metrics = task_df['metric'].dropna()
            if len(task_metrics) > 0:
                y_min = task_metrics.min()
                y_max = task_metrics.max()
                y_margin = (y_max - y_min) * 0.1
                y_range = [y_min - y_margin, y_max + y_margin]
            else:
                y_range = [0, 1]

            for base in base_archs:
                color = ARCHITECTURE_COLORS[base]
                for variant_name, cfg in variants.items():
                    sub = task_df[(task_df['base'] == base) & (task_df['variant'] == variant_name)]
                    if sub.empty:
                        continue
                    sub = sub.sort_values('train_percent')
                    ax.plot(sub['train_percent'], sub['metric'],
                            linestyle=cfg['linestyle'], color=color, marker=cfg['marker'],
                            label=(f"{display_arch_name(base)} {cfg['label_suffix']}").strip() if (row_idx == 0 and col_idx == 0) else None,
                            linewidth=3, markersize=10, alpha=0.9)

            # Axis cosmetics - remove individual y-axis labels
            ax.set_ylabel('')  # Remove y-axis label from each subplot
            ax.grid(True, alpha=0.3)
            ax.set_xticks(train_percents)
            ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE+10)
            # Remove x tick labels for top row only
            if row_idx == 0:
                ax.set_xticklabels([])
            ax.set_ylim(y_range)

    plt.tight_layout()
    # Adjust left margin to make room for single y-axis label
    plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.25, wspace=0.50, hspace=0.3)

    # Single y-axis label "Performance" centered vertically between the two rows, to the left of first column
    first_col_top = axes[0, 0].get_position()
    first_col_bottom = axes[1, 0].get_position()
    y_center = (first_col_top.y0 + first_col_bottom.y1) / 2.0  # Vertical center between the two rows
    fig.text(first_col_top.x0 - 0.08, y_center, 'Performance', ha='center', va='center', rotation=90, fontsize=AXIS_LABEL_FONTSIZE+10)

    # Row titles centered above the middle subplot in each row (with extra space above subplots)
    mid_top_pos = axes[0, 2].get_position()
    geo_top_pos = axes[1, 2].get_position()
    fig.text(mid_top_pos.x0 + (mid_top_pos.x1 - mid_top_pos.x0) / 2.0, mid_top_pos.y1 + 0.07, 'Random', ha='center', va='bottom', fontsize=AXIS_LABEL_FONTSIZE+10)
    fig.text(geo_top_pos.x0 + (geo_top_pos.x1 - geo_top_pos.x0) / 2.0, geo_top_pos.y1 + 0.01, 'Geographic', ha='center', va='bottom', fontsize=AXIS_LABEL_FONTSIZE+10)

    # Single x-axis label centered under the middle bottom subplot
    mid_bottom_pos = axes[1, 2].get_position()
    x_label_y = mid_bottom_pos.y0 - 0.1
    fig.text(mid_bottom_pos.x0 + (mid_bottom_pos.x1 - mid_bottom_pos.x0) / 2.0, x_label_y, 'Training Data %', ha='center', va='top', fontsize=AXIS_LABEL_FONTSIZE+10)

    # Legend below the x-axis label
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, x_label_y - 0.20), ncol=len(handles), columnspacing=0.7, handletextpad=0.3, fontsize=LEGEND_FONTSIZE+10, frameon=False)
    plt.savefig('RQ3_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('RQ3_plot.pdf', dpi=300, bbox_inches='tight')

def plot_tta_improvement_by_model():
    """Plot box plot showing improvement over JT for JT-TTT and JT-TTT-Geo across architectures"""

    # Collect improvement data for each architecture and adaptation mode across all tasks
    all_improvements = []

    for architecture in architectures_plots:
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
    num_architectures = len(architectures_plots)
    fig, axes = plt.subplots(1, num_architectures, figsize=(5 * num_architectures, 5))

    # Handle single subplot case
    if num_architectures == 1:
        axes = [axes]

    for i, architecture in enumerate(architectures_plots):
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
                        showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6))

        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize the plot
        ax.set_xticks(positions)
        ax.set_xticklabels(['MM-TTT', 'MM-TTT-Geo'], rotation=0)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(architecture, fontsize=AXIS_LABEL_FONTSIZE)

        if i == 0:
            ax.set_ylabel(f'Δ R² (vs JT)', fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)

    # No figure title
    plt.tight_layout()
    plt.savefig('TTA_improvement_by_model_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('TTA_improvement_by_model_boxplot.pdf', dpi=300, bbox_inches='tight')

def plot_tta_improvement():
    """Create a combined TTA plot with Random (top row) and Geographic (bottom row), 5 columns for tasks.
    Each subplot shows boxplots of improvements over JT for JT-TTT and JT-TTT-Geo, with outliers removed.
    """
    splits = ['Random', 'Geographic']
    all_improvements = []

    for split in splits:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'
            metric_name = f'{split} test {metric}'

            # Baseline JT per architecture
            jt_baseline = {}
            for architecture in architectures_plots:
                run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                run = next((run for run in runs if run.name.startswith(run_name)), None)
                if run:
                    jt_baseline[architecture] = run.summary_metrics.get(metric_name)

            # Improvements for JT-TTT and JT-TTT-Geo
            for mode in ['JT-TTT', 'JT-TTT-Geo']:
                for architecture in architectures_plots:
                    if architecture not in jt_baseline or jt_baseline[architecture] is None:
                        continue
                    run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if not run:
                        continue
                    perf = run.summary_metrics.get(metric_name)
                    base = jt_baseline[architecture]
                    if perf is None or np.isnan(perf) or base is None or np.isnan(base):
                        continue
                    improvement = perf - base
                    all_improvements.append({
                        'split': split,
                        'task': task,
                        'mode': mode,
                        'improvement': improvement
                    })

    df = pd.DataFrame(all_improvements)

    # 2 rows (Random, Geographic) x 5 columns (tasks)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))

    # Titles atop each column (task names)
    for j, task in enumerate(tasks):
        axes[0, j].set_title(task.replace('_', ' ').capitalize().replace('ph', 'pH'), fontsize=AXIS_LABEL_FONTSIZE)

    for row_idx, split in enumerate(splits):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            task_data = df[(df['split'] == split) & (df['task'] == task)]

            jt_ttt_data = task_data[task_data['mode'] == 'JT-TTT']['improvement'].dropna().tolist()
            jt_ttt_geo_data = task_data[task_data['mode'] == 'JT-TTT-Geo']['improvement'].dropna().tolist()

            positions = [1, 2]
            data_to_plot = [jt_ttt_data, jt_ttt_geo_data]

            bp = ax.boxplot(data_to_plot,
                            positions=positions,
                            widths=0.6,
                            patch_artist=True,
                            showmeans=True,
                            showfliers=False,
                            meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6),
                            medianprops=dict(color='black', linewidth=1))

            # Colors for boxes
            colors = ['#1f77b4', '#ff7f0e']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(['MM-TTT', 'MM-TTT-Geo'], rotation=0)
            ax.axhline(y=0, color='black', linewidth=1)

            # Set per-subplot y-axis label based on task metric
            metric = 'R2' if task != 'species' else 'MAP'
            ax.set_ylabel(f'Δ {metric}', fontsize=AXIS_LABEL_FONTSIZE)

            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    # Apply spacing first, then compute positions for accurate centering
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, hspace=0.35, wspace=0.45)

    # Add row labels centered over middle subplot in each row
    top_mid = axes[0, 2].get_position()
    bot_mid = axes[1, 2].get_position()

    fig.text(top_mid.x0 + (top_mid.x1 - top_mid.x0)/2.0, top_mid.y1 + 0.05,
             'Random', ha='center', va='bottom', fontsize=AXIS_LABEL_FONTSIZE)
    fig.text(bot_mid.x0 + (bot_mid.x1 - bot_mid.x0)/2.0, bot_mid.y1 + 0.01,
             'Geographic', ha='center', va='bottom', fontsize=AXIS_LABEL_FONTSIZE)

    # Single x-axis label (centered under the middle bottom subplot)
    # mid = axes[1, 2].get_position()
    # fig.text(mid.x0 + (mid.x1 - mid.x0) / 2.0,
    #          mid.y0 - 0.08,
    #          'Adaptation Mode', ha='center', va='top', fontsize=AXIS_LABEL_FONTSIZE)
    plt.savefig('TTA_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('TTA_plot.pdf', dpi=300, bbox_inches='tight')

def calculate_rq1_stats(test_split='Random'):
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

        for architecture in architectures_plots:
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
        for architecture in architectures_plots:
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

def calculate_rq2_stats():
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

        for architecture in architectures_plots:
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
    drops_convnext = []
    drops_pretrained = []

    for task in tasks:
        for architecture in architectures_plots:
            task_arch_data = df[(df['task'] == task) & (df['architecture'] == architecture)]
            if task_arch_data.empty:
                continue
            random_perf = task_arch_data['random_metric'].iloc[0]
            geographic_perf = task_arch_data['geographic_metric'].iloc[0]
            if np.isnan(random_perf) or np.isnan(geographic_perf) or random_perf <= 0:
                continue
            pct_drop = (random_perf - geographic_perf) / random_perf * 100
            if architecture == 'ConvNeXtV2A':
                drops_convnext.append(pct_drop)
            else:
                drops_pretrained.append(pct_drop)

    def calc_stats(drops):
        arr = np.array(drops)
        if len(arr) == 0:
            return None, None, None, None, 0
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        z = 1.96
        return mean, se, mean - z * se, mean + z * se, len(arr)

    mean_conv, se_conv, ci_l_conv, ci_u_conv, n_conv = calc_stats(drops_convnext)
    mean_pre, se_pre, ci_l_pre, ci_u_pre, n_pre = calc_stats(drops_pretrained)

    # Print results (split baseline vs pretrained)
    print("\nRQ2 ANALYSIS - TEST SPLIT COMPARISON")
    print("=" * 60)
    print("RQ2 Performance Drop Analysis (Random → Geographic Test Split)")
    print("=" * 80)
    print("ConvNeXtV2A (averaged over tasks):")
    if n_conv > 0:
        print(f"  n={n_conv}  mean: {mean_conv:.2f}% ± {se_conv:.2f}%  95% CI: [{ci_l_conv:.2f}%, {ci_u_conv:.2f}%]")
    else:
        print("  No data available")
    print("Pretrained models (all others, averaged over tasks and architectures):")
    if n_pre > 0:
        print(f"  n={n_pre}  mean: {mean_pre:.2f}% ± {se_pre:.2f}%  95% CI: [{ci_l_pre:.2f}%, {ci_u_pre:.2f}%]")
    else:
        print("  No data available")

    return {
        'convnext': {'mean': mean_conv, 'se': se_conv, 'ci': (ci_l_conv, ci_u_conv), 'n': n_conv},
        'pretrained': {'mean': mean_pre, 'se': se_pre, 'ci': (ci_l_pre, ci_u_pre), 'n': n_pre}
    }

def calculate_rq3_stats(test_split='Random'):
    """Calculate percentage performance drops when going from Multimodal to S2-only models,
    averaged over models and tasks with uncertainty estimates (RQ3 analysis).
    Set test_split to 'Random' or 'Geographic'."""

    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    rq3_architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    # Collect all performance data
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'{test_split} test {metric}'

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
    print(f"Test split: {test_split}")
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
        'n_combinations': n,
        'test_split': test_split
    }

def tabulate_results(rq_number):
    with open(f'latex_{tag}_RQ{rq_number}.tex', 'w') as file:
        file.write('\n'.join([globals()[f'tabulate_results_RQ{rq_number}_task'](task) for task in tasks]))

def tabulate_tta_per_model():
    """Create a table showing average improvement (averaged over tasks) of MT-TTT and MT-TTT-Geo over JT.
    Rows are MT-TTT and MT-TTT-Geo, columns are architectures. Includes standard error.
    """
    adaptation_modes = ['JT-TTT', 'JT-TTT-Geo']

    # Collect improvement data for each architecture and adaptation mode
    improvement_data = {architecture: {mode: [] for mode in adaptation_modes}
                        for architecture in architectures_plots}

    for architecture in architectures_plots:
        # Get JT baseline performance for each task
        jt_baseline = {}
        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'
            run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
            run = next((run for run in runs if run.name.startswith(run_name)), None)
            if run:
                jt_baseline[task] = run.summary_metrics.get(f'Random test {metric}')

        # Calculate improvements for JT-TTT and JT-TTT-Geo for each task
        for adaptation_mode in adaptation_modes:
            for task in tasks:
                if task in jt_baseline and jt_baseline[task] is not None:
                    metric = 'R2' if task != 'species' else 'MAP'
                    run_name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        performance = run.summary_metrics.get(f'Random test {metric}')
                        if performance is not None and not np.isnan(performance) and not np.isnan(jt_baseline[task]):
                            improvement = performance - jt_baseline[task]
                            improvement_data[architecture][adaptation_mode].append(improvement)

    # Calculate mean and standard error for each architecture-mode combination
    data_dict = {}
    for mode in adaptation_modes:
        data_dict[mode] = {}
        for architecture in architectures_plots:
            improvements = improvement_data[architecture][mode]
            if len(improvements) > 0:
                mean = np.mean(improvements)
                # Standard error: std / sqrt(n)
                se = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0.0
                data_dict[mode][architecture] = {'mean': mean, 'se': se, 'n': len(improvements)}
            else:
                data_dict[mode][architecture] = {'mean': np.nan, 'se': np.nan, 'n': 0}

    # Create DataFrame with mean ± SE format - transposed structure
    display_decimals = 3
    formatted_data = {}

    # Determine which values should be bolded (highest in each row = highest across methods for each architecture)
    bold_flags = {}
    for architecture in architectures_plots:
        jt_ttt_mean = data_dict['JT-TTT'][architecture]['mean']
        jt_ttt_geo_mean = data_dict['JT-TTT-Geo'][architecture]['mean']

        bold_flags[architecture] = {'JT-TTT': False, 'JT-TTT-Geo': False}

        # Compare means, handling NaN values
        if not np.isnan(jt_ttt_mean) and not np.isnan(jt_ttt_geo_mean):
            if jt_ttt_mean > jt_ttt_geo_mean:
                bold_flags[architecture]['JT-TTT'] = True
            elif jt_ttt_geo_mean > jt_ttt_mean:
                bold_flags[architecture]['JT-TTT-Geo'] = True
        elif not np.isnan(jt_ttt_mean):
            bold_flags[architecture]['JT-TTT'] = True
        elif not np.isnan(jt_ttt_geo_mean):
            bold_flags[architecture]['JT-TTT-Geo'] = True

    # Format data with architecture as rows, methods as columns
    for architecture in architectures_plots:
        formatted_data[architecture] = {}
        for mode in adaptation_modes:
            stats = data_dict[mode][architecture]
            if stats['n'] > 0 and not np.isnan(stats['mean']):
                mean_str = f"{stats['mean']:.{display_decimals}f}"
                se_str = f"{stats['se']:.{display_decimals}f}"
                # Bold inside math mode using \mathbf if needed
                if bold_flags[architecture][mode]:
                    formatted_data[architecture][mode] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                else:
                    formatted_data[architecture][mode] = f"${mean_str} \\pm {se_str}$"
            else:
                formatted_data[architecture][mode] = "--"

    # Create DataFrame - architectures as rows, methods as columns
    df = pd.DataFrame(formatted_data).T  # Transpose to get architectures as rows
    df = df.reindex(architectures_plots)  # Ensure correct row order
    # Columns should be the adaptation modes (display names)
    display_name_mapping = {'JT-TTT': 'MT-TTT', 'JT-TTT-Geo': 'MT-TTT-Geo'}
    df.columns = [display_name_mapping[mode] for mode in adaptation_modes]
    # Rename index to use display names for architectures
    df.index = [display_arch_name(arch) for arch in architectures_plots]

    # Create LaTeX table
    header_line = ' & '.join(['\\textbf{Model}'] + [f'\\textbf{{{c}}}' for c in df.columns]) + r' \\'
    latex = df.to_latex(index=True,
                        header=False,  # Set to False since we're manually adding the header
                        index_names=False,
                        escape=False,
                        column_format='l' + 'r' * len(df.columns),
                        na_rep='--')

    # Insert custom header after toprule
    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    # Insert header line and midrule after toprule
    lines.insert(toprule_idx + 1, header_line)
    lines.insert(toprule_idx + 2, '\\midrule')
    latex = '\n'.join(lines)

    # Add table environment
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            "\\caption{Average improvement over JT (averaged over tasks) for MT-TTT and MT-TTT-Geo by architecture. Values shown as mean $\\pm$ standard error across all tasks.}\n" +
            "\\label{tab:tta_improvement_averages}\n" +
            "\\end{table}\n")

    with open('tta_by_model.tex', 'w') as file:
        file.write(latex)

if __name__ == '__main__':
    # tabulate_results(1)
    # tabulate_results(2)
    # tabulate_results(3)
    tabulate_tta_results()
    tabulate_tta_per_model()
    # plot_rq1_performance()
    # plot_rq1_relative_performance()
    # plot_rq2_performance()
    # plot_rq3_performance()
    plot_tta_improvement()
    # plot_tta_improvement_by_model()

    # # Analyze performance drops for both test splits
    # random_results = calculate_rq1_stats('Random')
    # geographic_results = calculate_rq1_stats('Geographic')
    # rq2_results = calculate_rq2_stats()
    # rq3_results_random = calculate_rq3_stats('Random')
    # rq3_results_geographic = calculate_rq3_stats('Geographic')
