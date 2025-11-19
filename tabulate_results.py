from matplotlib.ticker import MaxNLocator
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
# architectures_tables = architectures_plots + ['AnySat']

architectures_tables = architectures_plots

# Create a consistent color mapping for all architectures
ARCHITECTURE_COLORS = {}
colors_list = plt.cm.tab10(np.linspace(0, 1, len(architectures_plots)))

for i, arch in enumerate(architectures_plots):
    ARCHITECTURE_COLORS[arch] = colors_list[i]

# Display-name mapping for plots/tables (keep 'Satlas' for wandb lookups)
def display_arch_name(name: str) -> str:
    return 'SatlasNet' if name == 'Satlas' else name

# Font size configuration
LEGEND_FONTSIZE = 7
AXIS_LABEL_FONTSIZE = 7
COL_WIDTH = 3.25
MARKER_SIZE = 1
LINE_WIDTH = 0.5

# Set font to Times for all figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

def tabulate_results_task(task, split, adaptation_mode):
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]
    train_percents = [5, 50, 100]
    metric = 'R2' if task != 'species' else 'MAP'
    split_metric_name = f'{split} test {metric}'

    # Load runs for each tag
    all_runs = {}

    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect data for each seed
    all_data = {}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]
        data = {architecture: {train_percent: np.nan for train_percent in train_percents} for architecture in architectures_tables}

        for architecture in architectures_tables:
            for train_percent in train_percents:
                name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                run = next((run for run in runs if run.name.startswith(name)), None)
                test_metric = run.summary_metrics.get(split_metric_name) if run else np.nan
                data[architecture][train_percent] = test_metric

        all_data[seed] = data

    # Create DataFrames for each seed
    df_disps = {}
    masks = {}
    display_decimals = 2

    for seed in seeds:
        # Create DataFrame with architectures as rows and train_percents as columns
        df = pd.DataFrame.from_dict(all_data[seed], orient='index')
        # Apply display name mapping for row index
        df.index = [display_arch_name(idx) for idx in df.index]
        df.index.name = 'Architecture'
        df.columns.name = 'Train %'
        # Round for display
        df_disp = df.round(display_decimals)
        df_disps[seed] = df_disp

        # --- highlight best per column (after rounding) ---
        mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)

        for col in df_disp.columns:
            best = df_disp[col].max(skipna=True)
            eq = df_disp[col].eq(best).fillna(False)
            mask[col] = eq

        masks[seed] = mask

    # Format each DataFrame
    df_fmts = {}

    for seed in seeds:
        df_disp = df_disps[seed]
        mask = masks[seed]
        # Format from the rounded values
        df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
        bold = '\\textbf{' + df_fmt + '}'
        df_fmt = df_fmt.where(~mask, bold)
        df_fmts[seed] = df_fmt

    # Combine DataFrames horizontally (side by side)
    # First, add seed number as a prefix to column names
    combined_dfs = []

    for seed in seeds:
        df_fmt = df_fmts[seed].copy()
        # Rename columns to include seed number
        df_fmt.columns = [f'{col} (Seed {seed})' for col in df_fmt.columns]
        combined_dfs.append(df_fmt)

    # Concatenate horizontally
    combined_df = pd.concat(combined_dfs, axis=1)

    # Create LaTeX table
    cols = combined_df.columns.tolist()
    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    # Build seed header with vertical lines: all but the last seed should have a vertical line on the right
    seed_header_parts = []
    for i, seed in enumerate(seeds):
        if i < len(seeds) - 1:
            # Not the last seed: include vertical line on the right
            seed_header_parts.append(f'\\multicolumn{{{len(train_percents)}}}{{c|}}{{\\textbf{{Seed {seed}}}}}')
        else:
            # Last seed: no vertical line on the right
            seed_header_parts.append(f'\\multicolumn{{{len(train_percents)}}}{{c}}{{\\textbf{{Seed {seed}}}}}')
    seed_header = ' & '.join([''] + seed_header_parts) + r' \\'
    train_header = ' & '.join(['\\textbf{Model}'] + [f'\\textbf{{{c}\\%}}' for seed in seeds for c in train_percents]) + r' \\'
    # Column format: l (Model) | rrr (Seed 41) | rrr (Seed 42) | rrr (Seed 43)
    column_format = 'l|' + '|'.join(['r'*len(train_percents) for _ in seeds])
    latex = combined_df.to_latex(index=True, header=False, index_names=False, escape=False, column_format=column_format)
    latex = latex.replace('\\toprule', '\\toprule\n' + seed_header + '\n' + train_header + '\n\\midrule', 1)
    split_lower = split.lower()
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} {split_lower} test {caption_metric} by model and training percentage}}\n" +
            f"\\label{{tab:{task}_{split_lower}}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_RQ3_task(task, split):
    """
    Generate LaTeX table for RQ3 with specified split and multiple seeds.

    Args:
        task: Task name (e.g., 'biomass', 'soil_pH')
        split: 'Random' or 'Geographic' (default: 'Random')

    Returns:
        LaTeX table string
    """
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]
    metric = 'R2' if task != 'species' else 'MAP'
    split_metric_name = f'{split} test {metric}'

    # Load runs for each tag
    all_runs = {}

    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Column names: just training percentages for the specified split
    col_names = [f'{tp}%' for tp in train_percents]

    # Collect data for each seed
    all_data = {}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]
        data = {architecture: {col: np.nan for col in col_names} for architecture in architectures}

        for architecture in architectures:
            for tp in train_percents:
                name = '_'.join([task, architecture, adaptation_mode, str(tp)]) + '_'
                run = next((run for run in runs if run.name.startswith(name)), None)
                if run:
                    val = run.summary_metrics.get(split_metric_name, np.nan)
                else:
                    val = np.nan

                data[architecture][f'{tp}%'] = val

        all_data[seed] = data

    # Create DataFrames for each seed
    df_disps = {}
    masks = {}
    display_decimals = 2

    for seed in seeds:
        # DataFrame with architectures as rows and train_percents as columns
        df = pd.DataFrame.from_dict(all_data[seed], orient='index')
        # Apply display name mapping for row index
        df.index = [display_arch_name(idx) for idx in df.index]
        df.index.name = 'Architecture'
        # Round for display
        df_disp = df.round(display_decimals)
        df_disps[seed] = df_disp

        # --- highlight best per column (after rounding) ---
        mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
        for col in df_disp.columns:
            best = df_disp[col].max(skipna=True)
            eq = df_disp[col].eq(best).fillna(False)
            mask[col] = eq

        masks[seed] = mask

    # Format each DataFrame
    df_fmts = {}

    for seed in seeds:
        df_disp = df_disps[seed]
        mask = masks[seed]
        # Format from the rounded values
        df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
        bold = '\\textbf{' + df_fmt + '}'
        df_fmt = df_fmt.where(~mask, bold)
        df_fmts[seed] = df_fmt

    # Combine DataFrames horizontally (side by side)
    # First, add seed number as a prefix to column names
    combined_dfs = []

    for seed in seeds:
        df_fmt = df_fmts[seed].copy()
        # Rename columns to include seed number
        df_fmt.columns = [f'{col} (Seed {seed})' for col in df_fmt.columns]
        combined_dfs.append(df_fmt)

    # Concatenate horizontally
    combined_df = pd.concat(combined_dfs, axis=1)

    # Create LaTeX table
    cols = combined_df.columns.tolist()
    cols_per_seed = len(train_percents)  # 3 columns per seed (5%, 50%, 100%)

    # Build group header: Seed headers spanning 3 columns each
    seed_header = ' & '.join([''] + [f'\\multicolumn{{{cols_per_seed}}}{{c}}{{\\textbf{{Seed {seed}}}}}' for seed in seeds]) + r' \\'

    # Second header row: Training percentages for each seed
    train_header = ' & '.join(['\\textbf{Model}'] + [f'\\textbf{{{c}\\%}}' for seed in seeds for c in train_percents]) + r' \\'

    latex = combined_df.to_latex(index=True, header=False, index_names=False, escape=False, column_format='l' + 'r'*len(cols))
    latex = latex.replace('\\toprule', '\\toprule\n' + seed_header + '\n' + train_header + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    split_lower = split.lower()
    latex = ("\\begin{table}[ht]\n\\centering\n" +
             latex +
             f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} {split_lower} test {caption_metric} by model, training percent, and seed}}\n" +
             f"\\label{{tab:{task}_rq3_{split_lower}}}\n" +
             "\\end{table}\n")

    return latex

def tabulate_results(split, adaptation_mode):
    """Generate combined LaTeX table for all tasks with specified split.

    Args:
        split: 'Random' or 'Geographic' (default: 'Random')
    """
    split_lower = split.lower()

    with open(f'results_{adaptation_mode}_{split_lower}.tex', 'w') as file:
        file.write('\n'.join([tabulate_results_task(task, split, adaptation_mode) for task in tasks]))

def tabulate_results_RQ3(split):
    """Generate combined LaTeX table for all tasks with specified split for RQ3.

    Args:
        split: 'Random' or 'Geographic'
    """
    split_lower = split.lower()
    filename = f'results_RQ3_{split_lower}.tex'
    with open(filename, 'w') as file:
        file.write('\n'.join([tabulate_results_RQ3_task(task, split=split) for task in tasks]))

def tabulate_TTT_results_task(task):
    adaptation_modes = ['FT', 'JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]
    metric = 'R2' if task != 'species' else 'MAP'
    display_name_mapping = {'JT-TTT': 'MT-TTT', 'JT-TTT-Geo': 'MT-TTT-Geo'}

    # Load runs for each tag
    all_runs = {}

    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect data for each seed
    all_data = {}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]
        data = {split: {mode: {architecture: np.nan for architecture in architectures_tables} for mode in adaptation_modes} for split in splits}

        for architecture in architectures_tables:
            for adaptation_mode in adaptation_modes:
                name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
                run = next((run for run in runs if run.name.startswith(name)), None)
                if run:
                    data['Random'][adaptation_mode][architecture] = run.summary_metrics.get(f'Random test {metric}', np.nan)
                    data['Geographic'][adaptation_mode][architecture] = run.summary_metrics.get(f'Geographic test {metric}', np.nan)

        all_data[seed] = data

    def _one_seed_table(seed):
        # Create DataFrames for Random and Geographic splits
        df_random = pd.DataFrame.from_dict(all_data[seed]['Random'], orient='index')[architectures_tables].reindex(adaptation_modes).T
        df_geographic = pd.DataFrame.from_dict(all_data[seed]['Geographic'], orient='index')[architectures_tables].reindex(adaptation_modes).T

        # Apply display name mapping to index
        df_random.index = [display_arch_name(idx) for idx in df_random.index]
        df_geographic.index = [display_arch_name(idx) for idx in df_geographic.index]
        df_random.index.name = 'Model'
        df_geographic.index.name = 'Model'

        # Format each DataFrame
        formatted_dfs = {}

        for split_name, df in [('Random', df_random), ('Geographic', df_geographic)]:
            formatted_df = df.copy()

            # --- bold highest per column (after rounding to 2 decimals), robust to strings ---
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

            formatted_dfs[split_name] = formatted_df

        # Combine Random and Geographic horizontally (side by side)
        df_random_fmt = formatted_dfs['Random'].copy()
        df_geographic_fmt = formatted_dfs['Geographic'].copy()

        # Concatenate horizontally (columns are already the adaptation modes)
        combined_df = pd.concat([df_random_fmt, df_geographic_fmt], axis=1)

        # Create LaTeX table
        num_adaptation_modes = len(adaptation_modes)
        cols_per_split = num_adaptation_modes
        cols = combined_df.columns.tolist()

        # Build group header: Split headers spanning 4 columns each
        split_header = ' & '.join([''] + [f'\\multicolumn{{{cols_per_split}}}{{c}}{{\\textbf{{{split}}}}}' for split in splits]) + r' \\'

        # Second header row: Adaptation modes for each split (with display name mapping)
        mode_header_parts = []
        for split in splits:
            display_modes = [display_name_mapping.get(mode, mode) for mode in adaptation_modes]
            mode_header_parts.extend(display_modes)
        mode_header = '\\textbf{Model} & ' + ' & '.join(mode_header_parts) + r' \\'

        body = combined_df.to_latex(index=True,
                                     header=False,
                                     index_names=False,
                                     escape=False,
                                     na_rep='--',
                                     column_format='l|' + ('c' * len(cols)),
                                     multicolumn=False,
                                     multirow=False)

        caption_metric = r"R$^2$" if metric == 'R2' else "MAP"
        # Replace \toprule\n with our custom headers
        body = body.replace("\\toprule\n", "\\toprule\n" + split_header + "\n" + mode_header + "\n\\midrule", 1)

        latex = ("\\begin{table}[ht]\n\\centering\n" +
                 body +
                 f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {caption_metric} by architecture, adaptation mode, and split (Seed {seed})}}\n" +
                 f"\\label{{tab:{task}_{metric}_seed{seed}}}\n" +
                 "\\end{table}\n")
        return latex

    # Generate one table per seed
    latex_tables = []
    for seed in seeds:
        latex_tables.append(_one_seed_table(seed))

    return "\n".join(latex_tables)

def tabulate_TTT_results():
    tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    latex = '\n'.join([tabulate_TTT_results_task(task) for task in tasks])

    with open('results_TTT.tex', 'w') as file:
        file.write(latex)

def plot_rq1_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    tags = ['pi_41', 'pi_42', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect data for all tasks, aggregating over seeds
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures_plots:
            for train_percent in train_percents:
                # Collect metrics from all seeds
                seed_metrics = []
                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]
                    name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                    run = next((run for run in runs if run.name.startswith(name)), None)
                    random_test_metric = run.summary_metrics.get(metric_name) if run else None
                    if random_test_metric is not None and not np.isnan(random_test_metric):
                        seed_metrics.append(random_test_metric)

                # Average over seeds if we have any valid metrics
                if seed_metrics:
                    avg_metric = np.mean(seed_metrics)
                    # Calculate standard error across seeds
                    std_error = np.std(seed_metrics, ddof=1) / np.sqrt(len(seed_metrics)) if len(seed_metrics) > 1 else 0
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': avg_metric, 'std_error': std_error})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDTH, 1.5), gridspec_kw=dict(left=0.09, right=0.98, top=0.71, bottom=0.23, wspace=0.4))

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]
        task_metrics = task_data['metric'].dropna()
        task_std_error = task_data['std_error'].dropna()
        # Account for mean ± SE in y-range
        y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
        y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()
        y_margin = (y_max - y_min) * 0.1
        y_range = [y_min - y_margin, y_max + y_margin]

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

                # Plot shaded region for mean ± SE across seeds
                x_vals = architecture_data_sorted['train_percent'].values
                y_mean = architecture_data_sorted['metric'].values
                y_std_error = architecture_data_sorted['std_error'].values
                y_lower = y_mean - y_std_error
                y_upper = y_mean + y_std_error
                ax.fill_between(x_vals, y_lower, y_upper,
                               color=color, alpha=0.2, linewidth=0)

                # Plot the line and markers
                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=display_arch_name(architecture), markersize=MARKER_SIZE, linewidth=LINE_WIDTH, alpha=0.8)

        if i == 0:
            ax.set_ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
        elif i == 2:
            ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.tick_params(axis='y', rotation=90)
        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(y_range)

    handles, labels = axes[0].get_legend_handles_labels()

    # Create a mapping from display name to handle
    handle_map = {label: handle for handle, label in zip(handles, labels)}

    # Desired column layout (4 columns):
    # Column 1: ConvNeXtV2A, ScaleMAE
    # Column 2: DINOv3Web, DINOv3Sat
    # Column 3: SatlasNet, MPMAE
    # Column 4: TerraMind, CopernicusFM

    # Map architecture names to their display names
    arch_to_display = {arch: display_arch_name(arch) for arch in architectures_plots}

    # Define columns
    col1_archs = ['ConvNeXtV2A', 'ScaleMAE']
    col2_archs = ['DINOv3Web', 'DINOv3Sat']
    col3_archs = ['Satlas', 'MPMAE']
    col4_archs = ['TerraMind', 'CopernicusFM']

    # Build handles and labels for each column
    col1_handles = []
    col1_labels = []
    for arch in col1_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col1_handles.append(handle_map[display_name])
            col1_labels.append(display_name)

    col2_handles = []
    col2_labels = []
    for arch in col2_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col2_handles.append(handle_map[display_name])
            col2_labels.append(display_name)

    col3_handles = []
    col3_labels = []
    for arch in col3_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col3_handles.append(handle_map[display_name])
            col3_labels.append(display_name)

    col4_handles = []
    col4_labels = []
    for arch in col4_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col4_handles.append(handle_map[display_name])
            col4_labels.append(display_name)

    # Create four separate legends positioned side by side above the plot
    # Use 'upper center' to align the top rows of all columns
    # Spread them out horizontally to avoid overlap
    # Position above the plot (top is at 0.75, so position at 0.90)
    legend_y_position = 1.05
    space_between_columns = 0.246
    legend1_x_position = 0.13
    # Center the four columns horizontally with increased spacing between them
    # Positions: 0.15, 0.35, 0.65, 0.85 (spacing of 0.20, centered at 0.50)
    legend1 = fig.legend(col1_handles, col1_labels,
                        loc='upper center',
                        bbox_to_anchor=(legend1_x_position, legend_y_position),
                        ncol=1,
                        handletextpad=0.3,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend1)  # Add immediately to prevent removal

    legend2 = fig.legend(col2_handles, col2_labels,
                        loc='upper center',
                        bbox_to_anchor=(legend1_x_position+space_between_columns, legend_y_position),
                        ncol=1,
                        handletextpad=0.3,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend2)  # Add immediately to prevent removal

    legend3 = fig.legend(col3_handles, col3_labels,
                        loc='upper center',
                        bbox_to_anchor=(legend1_x_position+2*space_between_columns, legend_y_position),
                        ncol=1,
                        handletextpad=0.3,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend3)  # Add immediately to prevent removal

    legend4 = fig.legend(col4_handles, col4_labels,
                        loc='upper center',
                        bbox_to_anchor=(legend1_x_position+3*space_between_columns, legend_y_position),
                        ncol=1,
                        handletextpad=0.3,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend4)  # Add immediately to prevent removal

    plt.savefig('RQ1_plot.png', dpi=300)
    plt.savefig('RQ1_plot.pdf', dpi=300)

def plot_rq2_performance():
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2
    tags = ['pi_41', 'pi_42', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect data for all tasks, aggregating over seeds
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        random_metric_name = f'Random test {metric}'
        geographic_metric_name = f'Geographic test {metric}'

        for architecture in architectures_plots:
            # Collect metrics from all seeds for Random split
            random_seed_metrics = []
            geographic_seed_metrics = []
            for tag, seed in zip(tags, seeds):
                runs = all_runs[tag]
                name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                run = next((run for run in runs if run.name.startswith(name)), None)
                random_test_metric = run.summary_metrics.get(random_metric_name) if run else None
                geographic_test_metric = run.summary_metrics.get(geographic_metric_name) if run else None
                if random_test_metric is not None and not np.isnan(random_test_metric):
                    random_seed_metrics.append(random_test_metric)
                if geographic_test_metric is not None and not np.isnan(geographic_test_metric):
                    geographic_seed_metrics.append(geographic_test_metric)

            # Calculate mean and standard error for Random split
            if random_seed_metrics:
                random_avg = np.mean(random_seed_metrics)
                random_std_error = np.std(random_seed_metrics, ddof=1) / np.sqrt(len(random_seed_metrics)) if len(random_seed_metrics) > 1 else 0
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Random', 'metric': random_avg, 'std_error': random_std_error})

            # Calculate mean and standard error for Geographic split
            if geographic_seed_metrics:
                geographic_avg = np.mean(geographic_seed_metrics)
                geographic_std_error = np.std(geographic_seed_metrics, ddof=1) / np.sqrt(len(geographic_seed_metrics)) if len(geographic_seed_metrics) > 1 else 0
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Geographic', 'metric': geographic_avg, 'std_error': geographic_std_error})

    df = pd.DataFrame(all_data)

    # 1 row x 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDTH, 1.1), gridspec_kw=dict(left=0.09, right=0.99, top=0.83, bottom=0.32, wspace=0.38))

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df['task'] == task]

        # y-range for this task, accounting for mean ± SE
        task_metrics = task_data['metric'].dropna()
        task_std_error = task_data['std_error'].dropna()
        if len(task_metrics) > 0:
            # Account for mean ± SE in y-range
            y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
            y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()
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
            rv_std_error = random_data['std_error'].iloc[0] if not random_data.empty and 'std_error' in random_data.columns else 0
            gv_std_error = geographic_data['std_error'].iloc[0] if not geographic_data.empty and 'std_error' in geographic_data.columns else 0

            # Plot shaded region for mean ± SE across seeds
            if rv is not None and gv is not None:
                x_vals = np.array([0, 1])
                y_lower = np.array([rv - rv_std_error, gv - gv_std_error])
                y_upper = np.array([rv + rv_std_error, gv + gv_std_error])
                ax.fill_between(x_vals, y_lower, y_upper,
                               color=color, alpha=0.2, linewidth=0)

            # Plot line connecting the two points
            ax.plot([0, 1], [rv, gv], color=color, linestyle=linestyle, linewidth=LINE_WIDTH, alpha=0.8)

            # Plot points
            if rv is not None:
                ax.plot(0, rv, marker, color=color, linestyle=linestyle, markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, alpha=0.8,
                       label=display_arch_name(architecture) if i == 0 else "")
            if gv is not None:
                ax.plot(1, gv, marker, color=color, linestyle=linestyle, markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, alpha=0.8)

        if i == 0:
            ax.set_ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)

        if i == 2:
            ax.set_xlabel('Test Split', fontsize=AXIS_LABEL_FONTSIZE)

        ax.tick_params(axis='y', rotation=90)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['R', 'G'])
        ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(y_range)

    plt.savefig('RQ2_plot.png', dpi=300)
    plt.savefig('RQ2_plot.pdf', dpi=300)

def plot_rq3_performance():
    """Create an RQ3 plot with Random and Geographic splits in 2 rows x 5 columns (tasks).
    S2 is solid with circle markers; Multimodal is dashed with triangle markers.
    Legend under the plot; task names above each column.
    """
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    splits = ['Random', 'Geographic']
    base_archs = ['TerraMind', 'CopernicusFM']
    variants = {
        'S2': {'suffix': 'S2', 'linestyle': '-', 'marker': 'o', 'label_suffix': 'S2'},
        'Multimodal': {'suffix': '', 'linestyle': '--', 'marker': '^', 'label_suffix': ''}
    }
    tags = ['pi_41', 'pi_42', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect data for both splits, aggregating over seeds
    rows = []
    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        for split in splits:
            metric_name = f"{split} test {metric}"
            for base in base_archs:
                for variant_name, cfg in variants.items():
                    arch_lookup = base + cfg['suffix']
                    for tp in train_percents:
                        # Collect metrics from all seeds
                        seed_metrics = []
                        for tag, seed in zip(tags, seeds):
                            runs = all_runs[tag]
                            name = '_'.join([task, arch_lookup, adaptation_mode, str(tp)]) + '_'
                            run = next((run for run in runs if run.name.startswith(name)), None)
                            val = run.summary_metrics.get(metric_name) if run else None
                            if val is not None and not np.isnan(val):
                                seed_metrics.append(val)

                        # Calculate mean and standard error if we have any valid metrics
                        if seed_metrics:
                            avg_metric = np.mean(seed_metrics)
                            std_error = np.std(seed_metrics, ddof=1) / np.sqrt(len(seed_metrics)) if len(seed_metrics) > 1 else 0
                            rows.append({
                                'task': task,
                                'split': split,
                                'base': base,
                                'variant': variant_name,
                                'train_percent': tp,
                                'metric': avg_metric,
                                'std_error': std_error
                            })

    df = pd.DataFrame(rows)

    # Figure: 2 rows x 5 columns (splits x tasks)
    fig, axes = plt.subplots(2, 5, figsize=(COL_WIDTH, 2), gridspec_kw=dict(left=0.09, right=0.96, top=0.89, bottom=0.30, wspace=0.4, hspace=0.3))

    for i, task in enumerate(tasks):
        for j, split in enumerate(splits):
            ax = axes[j, i]
            task_split_df = df[(df['task'] == task) & (df['split'] == split)]

            # y-range per panel, accounting for mean ± SE
            task_metrics = task_split_df['metric'].dropna()
            task_std_error = task_split_df['std_error'].dropna()
            if not task_metrics.empty:
                # Account for mean ± SE in y-range
                y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
                y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()
                y_margin = (y_max - y_min) * 0.1
                y_range = [y_min - y_margin, y_max + y_margin]
            else:
                y_range = [0, 1]

            for base in base_archs:
                color = ARCHITECTURE_COLORS[base]
                for variant_name, cfg in variants.items():
                    sub = task_split_df[(task_split_df['base'] == base) & (task_split_df['variant'] == variant_name)]
                    if sub.empty:
                        continue
                    sub = sub.sort_values('train_percent')

                    # Plot shaded region for mean ± SE across seeds
                    x_vals = sub['train_percent'].values
                    y_mean = sub['metric'].values
                    y_std_error = sub['std_error'].values
                    y_lower = y_mean - y_std_error
                    y_upper = y_mean + y_std_error
                    ax.fill_between(x_vals, y_lower, y_upper,
                                   color=color, alpha=0.2, linewidth=0)

                    # Plot the line and markers
                    ax.plot(sub['train_percent'], sub['metric'],
                            linestyle=cfg['linestyle'], color=color, marker=cfg['marker'],
                            label=(f"{display_arch_name(base)} {cfg['label_suffix']}").strip() if i == 0 and j == 0 else None,
                            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9)

            # Axis cosmetics
            if i == 2 and j == 1:
                ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

            # Set title only on top row
            if j == 0:
                ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            for label in ax.get_yticklabels():
                label.set_ha('center')
                label.set_va('center')

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', rotation=90)
            ax.set_xticks(train_percents)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE)

            # Remove x-axis ticks and labels on top row
            if j == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, top=False)

            ax.set_ylim(y_range)

    # Add "Performance" label in the middle between the two rows
    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    fig.text(0.02, center_y, 'Performance', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    # Add "Random" and "Geographic" labels to the right of the subplots
    top_row_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_row_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.98, top_row_center, 'Random', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')
    fig.text(0.98, bottom_row_center, 'Geographic', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')

    # Legend under the plot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            columnspacing=0.5,
            handletextpad=0.3,
            handlelength=1,
            labelspacing=0.1,
            fontsize=LEGEND_FONTSIZE,
            frameon=False)

    plt.savefig('RQ3_plot.png', dpi=300)
    plt.savefig('RQ3_plot.pdf', dpi=300)

def plot_ttt_improvement():
    """Create a TTT improvement plot with Random and Geographic splits in 2 rows x 5 columns (tasks).
    Each subplot shows boxplots of improvements over JT for JT-TTT and JT-TTT-Geo, with outliers removed.
    Statistics are computed separately for each seed, then averaged across seeds.
    Improvement is calculated as raw delta: new - old
    Legend under the plot.
    """

    splits = ['Random', 'Geographic']
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect improvements per seed: {seed: {task: {split: {mode: [improvements]}}}}
    improvements_per_seed = {seed: {task: {split: {mode: [] for mode in ['JT-TTT', 'JT-TTT-Geo']}
                                           for split in splits} for task in tasks} for seed in seeds}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]

        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'

            for split in splits:
                metric_name = f'{split} test {metric}'

                # Baseline JT per architecture for this seed
                jt_baseline = {}
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        jt_baseline[architecture] = run.summary_metrics.get(metric_name)

                # Improvements for JT-TTT and JT-TTT-Geo for this seed
                for mode in ['JT-TTT', 'JT-TTT-Geo']:
                    for architecture in architectures_plots:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)

                        if not run:
                            continue

                        performance = run.summary_metrics.get(metric_name)
                        base = jt_baseline.get(architecture)

                        # Raw delta: new - old
                        if base is not None and performance is not None and not np.isnan(base) and not np.isnan(performance):
                            improvement = performance - base
                            if not np.isnan(improvement):
                                improvements_per_seed[seed][task][split][mode].append(improvement)

    # Compute boxplot statistics per seed, then average them
    # Structure: {task: {split: {mode: {stat_name: averaged_value}}}}
    averaged_stats = {}

    for task in tasks:
        averaged_stats[task] = {}
        for split in splits:
            averaged_stats[task][split] = {}
            for mode in ['JT-TTT', 'JT-TTT-Geo']:
                # Collect statistics for each seed
                seed_stats = []
                for seed in seeds:
                    improvements = improvements_per_seed[seed][task][split][mode]
                    if len(improvements) > 0:
                        improvements_array = np.array(improvements)
                        # Compute boxplot statistics
                        q1 = np.percentile(improvements_array, 25)
                        median = np.percentile(improvements_array, 50)
                        q3 = np.percentile(improvements_array, 75)
                        mean = np.mean(improvements_array)
                        iqr = q3 - q1
                        # Whiskers: 1.5 * IQR from Q1 and Q3
                        whislo = q1 - 1.5 * iqr
                        whishi = q3 + 1.5 * iqr
                        # Clip whiskers to actual data range
                        whislo = max(whislo, np.min(improvements_array))
                        whishi = min(whishi, np.max(improvements_array))
                        seed_stats.append({
                            'q1': q1,
                            'med': median,
                            'q3': q3,
                            'mean': mean,
                            'whislo': whislo,
                            'whishi': whishi
                        })

                # Average statistics across seeds
                if len(seed_stats) > 0:
                    averaged_stats[task][split][mode] = {
                        'q1': np.mean([s['q1'] for s in seed_stats]),
                        'med': np.mean([s['med'] for s in seed_stats]),
                        'q3': np.mean([s['q3'] for s in seed_stats]),
                        'mean': np.mean([s['mean'] for s in seed_stats]),
                        'whislo': np.mean([s['whislo'] for s in seed_stats]),
                        'whishi': np.mean([s['whishi'] for s in seed_stats])
                    }
                else:
                    averaged_stats[task][split][mode] = None

    # 2 rows x 5 columns (splits x tasks)
    fig, axes = plt.subplots(2, 5, figsize=(COL_WIDTH, 2), gridspec_kw=dict(left=0.1, right=0.96, top=0.89, bottom=0.13, wspace=0.4, hspace=0.45))

    # Colors for boxes
    colors = ['#1f77b4', '#ff7f0e']

    for i, task in enumerate(tasks):
        for j, split in enumerate(splits):
            ax = axes[j, i]

            # Get averaged statistics for this task and split
            stats_list = []
            positions = []
            mode_colors = []  # Track which color to use for each boxplot
            for pos, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors), start=1):
                stats = averaged_stats[task][split][mode]
                if stats is not None:
                    # Create statistics dict for bxp
                    stats_dict = {
                        'med': stats['med'],
                        'q1': stats['q1'],
                        'q3': stats['q3'],
                        'whislo': stats['whislo'],
                        'whishi': stats['whishi'],
                        'mean': stats['mean']
                    }
                    stats_list.append(stats_dict)
                    positions.append(pos)
                    mode_colors.append(color)

            if len(stats_list) > 0:
                # Create boxplots from averaged statistics
                bp = ax.bxp(stats_list,
                            positions=positions,
                            widths=0.6,
                            patch_artist=True,
                            showmeans=True,
                            showfliers=False,
                            meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=MARKER_SIZE),
                            medianprops=dict(color='black', linewidth=LINE_WIDTH),
                            boxprops=dict(linewidth=LINE_WIDTH),
                            whiskerprops=dict(linewidth=LINE_WIDTH),
                            capprops=dict(linewidth=LINE_WIDTH))

                # Colors for boxes - use the tracked colors
                for patch, color in zip(bp['boxes'], mode_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(0.5)  # Thinner border around boxes

            # Remove x-axis ticks
            ax.set_xticks([])
            ax.set_xticklabels([])  # Remove method names from under each subplot
            ax.tick_params(axis='x', bottom=False)  # Hide x-axis ticks
            ax.axhline(y=0, color='black', linewidth=LINE_WIDTH)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis='y', rotation=90)

            # After the plot is drawn, get the natural y-axis range and set ticks at min/max
            # Round to 0.01 (two decimal places) and ensure at most 2 ticks
            ymin, ymax = ax.get_ylim()
            rounded_ymin = np.round(ymin / 0.01) * 0.01
            rounded_ymax = np.round(ymax / 0.01) * 0.01
            ax.set_yticks([rounded_ymin, rounded_ymax])

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

            for label in ax.get_yticklabels():
                label.set_ha('center')
                label.set_va('center')

            # Set title only on top row
            if j == 0:
                ax.set_title(task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH'), fontsize=AXIS_LABEL_FONTSIZE)

    # Add "Performance" label in the middle between the two rows
    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    fig.text(0.02, center_y, 'Δ Performance', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    # Add "Random" and "Geographic" labels to the right of the subplots
    top_row_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_row_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.98, top_row_center, 'Random', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')
    fig.text(0.98, bottom_row_center, 'Geographic', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')

    # Create legend for method colors under the plot
    legend_labels = ['TTT-MMR', 'TTT-MMR-Geo']
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) for color in colors]
    fig.legend(legend_handles, legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_labels),
            fontsize=LEGEND_FONTSIZE,
            frameon=False)

    plt.savefig('TTT_plot.png', dpi=300)
    plt.savefig('TTT_plot.pdf', dpi=300)

def plot_ttt_improvement_normalized():
    """Create a TTT improvement plot with Random and Geographic splits in 2 rows x 5 columns (tasks).
    Each subplot shows boxplots of normalized improvements over JT for JT-TTT and JT-TTT-Geo, with outliers removed.
    Statistics are computed separately for each seed, then averaged across seeds.
    Improvement is calculated as normalized: (r2_new - r2_old) / (1 - r2_old)
    Legend under the plot.
    """

    splits = ['Random', 'Geographic']
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect improvements per seed: {seed: {task: {split: {mode: [improvements]}}}}
    improvements_per_seed = {seed: {task: {split: {mode: [] for mode in ['JT-TTT', 'JT-TTT-Geo']}
                                           for split in splits} for task in tasks} for seed in seeds}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]

        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'

            for split in splits:
                metric_name = f'{split} test {metric}'

                # Baseline JT per architecture for this seed
                jt_baseline = {}
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        jt_baseline[architecture] = run.summary_metrics.get(metric_name)

                # Improvements for JT-TTT and JT-TTT-Geo for this seed
                for mode in ['JT-TTT', 'JT-TTT-Geo']:
                    for architecture in architectures_plots:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)

                        if not run:
                            continue

                        performance = run.summary_metrics.get(metric_name)
                        base = jt_baseline.get(architecture)

                        # Normalized improvement: (r2_new - r2_old) / (1 - r2_old)
                        if base is not None and performance is not None and not np.isnan(base) and not np.isnan(performance):
                            # Handle edge case: if baseline is 1, avoid division by zero
                            if base != 1:
                                improvement = (performance - base) / (1 - base)
                                if not np.isnan(improvement):
                                    improvements_per_seed[seed][task][split][mode].append(improvement)

    # Compute boxplot statistics per seed, then average them
    # Structure: {task: {split: {mode: {stat_name: averaged_value}}}}
    averaged_stats = {}

    for task in tasks:
        averaged_stats[task] = {}
        for split in splits:
            averaged_stats[task][split] = {}
            for mode in ['JT-TTT', 'JT-TTT-Geo']:
                # Collect statistics for each seed
                seed_stats = []
                for seed in seeds:
                    improvements = improvements_per_seed[seed][task][split][mode]
                    if len(improvements) > 0:
                        improvements_array = np.array(improvements)
                        # Compute boxplot statistics
                        q1 = np.percentile(improvements_array, 25)
                        median = np.percentile(improvements_array, 50)
                        q3 = np.percentile(improvements_array, 75)
                        mean = np.mean(improvements_array)
                        iqr = q3 - q1
                        # Whiskers: 1.5 * IQR from Q1 and Q3
                        whislo = q1 - 1.5 * iqr
                        whishi = q3 + 1.5 * iqr
                        # Clip whiskers to actual data range
                        whislo = max(whislo, np.min(improvements_array))
                        whishi = min(whishi, np.max(improvements_array))
                        seed_stats.append({
                            'q1': q1,
                            'med': median,
                            'q3': q3,
                            'mean': mean,
                            'whislo': whislo,
                            'whishi': whishi
                        })

                # Average statistics across seeds
                if len(seed_stats) > 0:
                    averaged_stats[task][split][mode] = {
                        'q1': np.mean([s['q1'] for s in seed_stats]),
                        'med': np.mean([s['med'] for s in seed_stats]),
                        'q3': np.mean([s['q3'] for s in seed_stats]),
                        'mean': np.mean([s['mean'] for s in seed_stats]),
                        'whislo': np.mean([s['whislo'] for s in seed_stats]),
                        'whishi': np.mean([s['whishi'] for s in seed_stats])
                    }
                else:
                    averaged_stats[task][split][mode] = None

    # 2 rows x 5 columns (splits x tasks)
    fig, axes = plt.subplots(2, 5, figsize=(COL_WIDTH, 2), gridspec_kw=dict(left=0.1, right=0.96, top=0.89, bottom=0.13, wspace=0.4, hspace=0.45))

    # Colors for boxes
    colors = ['#1f77b4', '#ff7f0e']

    for i, task in enumerate(tasks):
        for j, split in enumerate(splits):
            ax = axes[j, i]

            # Get averaged statistics for this task and split
            stats_list = []
            positions = []
            mode_colors = []  # Track which color to use for each boxplot
            for pos, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors), start=1):
                stats = averaged_stats[task][split][mode]
                if stats is not None:
                    # Create statistics dict for bxp
                    stats_dict = {
                        'med': stats['med'],
                        'q1': stats['q1'],
                        'q3': stats['q3'],
                        'whislo': stats['whislo'],
                        'whishi': stats['whishi'],
                        'mean': stats['mean']
                    }
                    stats_list.append(stats_dict)
                    positions.append(pos)
                    mode_colors.append(color)

            if len(stats_list) > 0:
                # Create boxplots from averaged statistics
                bp = ax.bxp(stats_list,
                            positions=positions,
                            widths=0.6,
                            patch_artist=True,
                            showmeans=True,
                            showfliers=False,
                            meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=MARKER_SIZE),
                            medianprops=dict(color='black', linewidth=LINE_WIDTH),
                            boxprops=dict(linewidth=LINE_WIDTH),
                            whiskerprops=dict(linewidth=LINE_WIDTH),
                            capprops=dict(linewidth=LINE_WIDTH))

                # Colors for boxes - use the tracked colors
                for patch, color in zip(bp['boxes'], mode_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(0.5)  # Thinner border around boxes

            # Remove x-axis ticks
            ax.set_xticks([])
            ax.set_xticklabels([])  # Remove method names from under each subplot
            ax.tick_params(axis='x', bottom=False)  # Hide x-axis ticks
            ax.axhline(y=0, color='black', linewidth=LINE_WIDTH)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis='y', rotation=90)

            # After the plot is drawn, get the natural y-axis range and set ticks at min/max
            # Round to 0.01 (two decimal places) and ensure at most 2 ticks
            ymin, ymax = ax.get_ylim()
            rounded_ymin = np.round(ymin / 0.01) * 0.01
            rounded_ymax = np.round(ymax / 0.01) * 0.01
            ax.set_yticks([rounded_ymin, rounded_ymax])

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

            for label in ax.get_yticklabels():
                label.set_ha('center')
                label.set_va('center')

            # Set title only on top row
            if j == 0:
                ax.set_title(task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH'), fontsize=AXIS_LABEL_FONTSIZE)

    # Add "Normalized Δ Performance" label in the middle between the two rows
    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    fig.text(0.02, center_y, 'Normalized Δ Performance', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    # Add "Random" and "Geographic" labels to the right of the subplots
    top_row_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_row_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.98, top_row_center, 'Random', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')
    fig.text(0.98, bottom_row_center, 'Geographic', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')

    # Create legend for method colors under the plot
    legend_labels = ['TTT-MMR', 'TTT-MMR-Geo']
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) for color in colors]
    fig.legend(legend_handles, legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_labels),
            fontsize=LEGEND_FONTSIZE,
            frameon=False)

    plt.savefig('TTT_plot_normalized.png', dpi=300)
    plt.savefig('TTT_plot_normalized.pdf', dpi=300)

def calculate_rq1_stats(test_split='Random'):
    """Calculate performance drops (raw deltas) when reducing training data from 100% to 50% and 100% to 5%,
    averaged over models and tasks with uncertainty estimates.
    Uses raw delta: new - old

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

                if not (np.isnan(perf_100) or np.isnan(perf_50) or np.isnan(perf_5)):
                    # Calculate raw delta: new - old
                    # For drops: old = perf_100, new = perf_reduced
                    norm_drop_100_to_50 = perf_50 - perf_100
                    norm_drop_100_to_5 = perf_5 - perf_100

                    # Separate by architecture type
                    if architecture == 'ConvNeXtV2A':
                        pct_drops_100_to_50_convnext.append(norm_drop_100_to_50)
                        pct_drops_100_to_5_convnext.append(norm_drop_100_to_5)
                    else:
                        pct_drops_100_to_50_pretrained.append(norm_drop_100_to_50)
                        pct_drops_100_to_5_pretrained.append(norm_drop_100_to_5)

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
    print(f"Performance Drop Analysis ({test_split} Test Split)")
    print(f"Training data reduction: 100% → 50% and 100% → 5%")
    print(f"Raw delta: new - old")
    print()

    print("ConvNeXtV2A (Randomly Initialized Baseline) - Averaged over Tasks:")
    print(f"  Number of tasks: {n_50_conv}")
    print("  100% → 50% training data:")
    if mean_50_conv is not None:
        print(f"    Mean drop: {mean_50_conv:.4f} ± {se_50_conv:.4f}")
        print(f"    95% CI: [{ci_50_lower_conv:.4f}, {ci_50_upper_conv:.4f}]")
    else:
        print("    No data available")
    print("  100% → 5% training data:")
    if mean_5_conv is not None:
        print(f"    Mean drop: {mean_5_conv:.4f} ± {se_5_conv:.4f}")
        print(f"    95% CI: [{ci_5_lower_conv:.4f}, {ci_5_upper_conv:.4f}]")
    else:
        print("    No data available")
    print()

    print("Pretrained Models (All Other Architectures) - Averaged over Tasks and Architectures:")
    print(f"  Number of task-architecture combinations: {n_50_pre}")
    print("  100% → 50% training data:")
    if mean_50_pre is not None:
        print(f"    Mean drop: {mean_50_pre:.4f} ± {se_50_pre:.4f}")
        print(f"    95% CI: [{ci_50_lower_pre:.4f}, {ci_50_upper_pre:.4f}]")
    else:
        print("    No data available")
    print("  100% → 5% training data:")
    if mean_5_pre is not None:
        print(f"    Mean drop: {mean_5_pre:.4f} ± {se_5_pre:.4f}")
        print(f"    95% CI: [{ci_5_lower_pre:.4f}, {ci_5_upper_pre:.4f}]")
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
    """Calculate performance drops (raw deltas) when going from Random to Geographic test splits,
    averaged over models and tasks with uncertainty estimates (RQ2 analysis).
    Uses raw delta: new - old"""

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
            if np.isnan(random_perf) or np.isnan(geographic_perf):
                continue
            # Calculate raw delta: new - old
            # For drops: old = random_perf, new = geographic_perf
            norm_drop = geographic_perf - random_perf

            if architecture == 'ConvNeXtV2A':
                drops_convnext.append(norm_drop)
            else:
                drops_pretrained.append(norm_drop)

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
    print("Raw delta: new - old")
    print("=" * 80)
    print("ConvNeXtV2A (averaged over tasks):")
    if n_conv > 0:
        print(f"  n={n_conv}  mean: {mean_conv:.4f} ± {se_conv:.4f}  95% CI: [{ci_l_conv:.4f}, {ci_u_conv:.4f}]")
    else:
        print("  No data available")
    print("Pretrained models (all others, averaged over tasks and architectures):")
    if n_pre > 0:
        print(f"  n={n_pre}  mean: {mean_pre:.4f} ± {se_pre:.4f}  95% CI: [{ci_l_pre:.4f}, {ci_u_pre:.4f}]")
    else:
        print("  No data available")

    return {
        'convnext': {'mean': mean_conv, 'se': se_conv, 'ci': (ci_l_conv, ci_u_conv), 'n': n_conv},
        'pretrained': {'mean': mean_pre, 'se': se_pre, 'ci': (ci_l_pre, ci_u_pre), 'n': n_pre}
    }

def calculate_rq3_stats(test_split='Random'):
    """Calculate performance drops (raw deltas) when going from Multimodal to S2-only models,
    averaged over models and tasks with uncertainty estimates (RQ3 analysis).
    Uses raw delta: new - old
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

                    if not (np.isnan(multimodal_perf) or np.isnan(s2_perf)):
                        # Calculate raw delta: new - old
                        # For drops: old = multimodal_perf, new = s2_perf
                        norm_drop = s2_perf - multimodal_perf
                        pct_drops_multimodal_to_s2.append(norm_drop)

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
    print("Raw delta: new - old")
    print("=" * 80)
    print(f"Training data: 100% (full training data)")
    print(f"Test split: {test_split}")
    print(f"Number of task-architecture combinations: {n}")
    print()
    print("Multimodal → S2-only models:")
    print(f"  Mean drop: {mean_pct_drop:.4f} ± {se_pct_drop:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()
    print("Additional statistics:")
    print(f"  Standard deviation: {np.std(pct_drops_multimodal_to_s2, ddof=1):.4f}")
    print(f"  Range: [{np.min(pct_drops_multimodal_to_s2):.4f}, {np.max(pct_drops_multimodal_to_s2):.4f}]")
    print()

    return {
        'pct_drop_multimodal_to_s2': {'mean': mean_pct_drop, 'se': se_pct_drop, 'ci': (ci_lower, ci_upper)},
        'n_combinations': n,
        'test_split': test_split
    }

def tabulate_ttt_by_model():
    """Create a table showing average improvement (normalized, averaged over tasks) of MT-TTT and MT-TTT-Geo over JT.
    Rows are methods (MT-TTT and MT-TTT-Geo) for Random, then for Geographic (4 rows total).
    Columns are architectures. Includes standard error.
    Separate results for Random and Geographic test splits.
    Improvement is calculated as normalized: (r2_new - r2_old) / (1 - r2_old)
    """

    adaptation_modes = ['JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    improvement_data = {architecture: {split: {mode: [] for mode in adaptation_modes} for split in splits} for architecture in architectures_plots}

    for architecture in architectures_plots:
        # Get JT baseline performance for each task and split
        jt_baseline = {}

        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'
            jt_baseline[task] = {}
            run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
            run = next((run for run in runs if run.name.startswith(run_name)), None)

            if run:
                for split in splits:
                    metric_name = f'{split} test {metric}'
                    baseline_value = run.summary_metrics.get(metric_name)

                    if baseline_value is not None and not np.isnan(baseline_value):
                        jt_baseline[task][split] = baseline_value

        # Calculate improvements for JT-TTT and JT-TTT-Geo for each task and split
        for adaptation_mode in adaptation_modes:
            for task in tasks:
                if task in jt_baseline and len(jt_baseline[task]) > 0:
                    metric = 'R2' if task != 'species' else 'MAP'
                    run_name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)

                    if run:
                        # Collect improvements separately for each split
                        for split in splits:
                            if split in jt_baseline[task]:
                                metric_name = f'{split} test {metric}'
                                performance = run.summary_metrics.get(metric_name)
                                baseline = jt_baseline[task][split]

                                if performance is not None and not np.isnan(performance) and not np.isnan(baseline):
                                    # Calculate normalized improvement: (r2_new - r2_old) / (1 - r2_old)
                                    # Handle edge cases: if baseline is 1, avoid division by zero
                                    if baseline != 1:
                                        improvement = (performance - baseline) / (1 - baseline)
                                    else:
                                        improvement = np.nan  # Skip if baseline is 1 (perfect R2)
                                    if not np.isnan(improvement):
                                        improvement_data[architecture][split][adaptation_mode].append(improvement)

    # Calculate mean and standard error for each architecture-mode-split combination
    data_dict = {}

    for split in splits:
        data_dict[split] = {}
        for mode in adaptation_modes:
            data_dict[split][mode] = {}
            for architecture in architectures_plots:
                improvements = improvement_data[architecture][split][mode]

                if len(improvements) > 0:
                    mean = np.mean(improvements)
                    # Standard error: std / sqrt(n)
                    se = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0.0
                    data_dict[split][mode][architecture] = {'mean': mean, 'se': se, 'n': len(improvements)}
                else:
                    data_dict[split][mode][architecture] = {'mean': np.nan, 'se': np.nan, 'n': 0}

    # Find the method with the highest mean for each split-architecture combination
    # This will be used to bold the best (highest improvement) value per column within each split
    best_methods = {}  # {(split, architecture): mode}
    for split in splits:
        for architecture in architectures_plots:
            best_mean = float('-inf')
            best_mode = None
            for mode in adaptation_modes:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    if stats['mean'] > best_mean:
                        best_mean = stats['mean']
                        best_mode = mode
            if best_mode is not None:
                best_methods[(split, architecture)] = best_mode

    # Create DataFrame with mean ± SE format
    # Rows: MT-TTT, MT-TTT-Geo (for Random), then MT-TTT, MT-TTT-Geo (for Geographic)
    # Columns: Split, Method, then architectures
    display_decimals = 3
    formatted_data = {}
    split_column = []
    method_column = []
    row_keys = []

    # Create row labels and data with unique keys
    display_name_mapping = {'JT-TTT': 'TTT-MMR', 'JT-TTT-Geo': 'TTT-MMR-Geo'}

    for split in splits:
        for mode in adaptation_modes:
            display_mode = display_name_mapping[mode]
            # Use unique key that includes split to avoid duplicates
            row_key = f"{split}_{display_mode}"
            row_keys.append(row_key)
            split_column.append(split)
            method_column.append(display_mode)
            formatted_data[row_key] = {}

            for architecture in architectures_plots:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    mean_str = f"{stats['mean']:.{display_decimals}f}"
                    se_str = f"{stats['se']:.{display_decimals}f}"
                    # Bold if this is the best (highest mean) method for this split-architecture
                    if best_methods.get((split, architecture)) == mode:
                        formatted_data[row_key][architecture] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][architecture] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][architecture] = "--"

    # Create DataFrame
    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)  # Ensure correct row order
    df = df.reindex(architectures_plots, axis=1)  # Ensure correct column order
    df.columns = [display_arch_name(arch) for arch in architectures_plots]

    # Insert Split and Method columns at the beginning
    df.insert(0, 'Method', method_column)
    df.insert(0, 'Split', split_column)

    # Create LaTeX table
    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Method}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'r' * (len(df.columns) - 2),
                        na_rep='--')

    # Insert custom header and use multicolumn for Split column
    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    # Remove any existing midrule that might be right after toprule
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Add multicolumn for Split column to group rows
    # Find the data rows (after midrule)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    # Replace Split values with multicolumn for first row of each split group
    row_count = 0
    geographic_start_idx = None
    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                # First row of Random (row_count 0) and first row of Geographic (row_count 2)
                if row_count == 0:  # First Random row
                    parts[0] = f"\\multirow{{2}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == 2:  # First Geographic row
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{2}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""  # Empty for subsequent rows in group
                lines[i] = ' & '.join(parts)
                row_count += 1

    # Insert horizontal line before Geographic section
    if geographic_start_idx is not None:
        # Check if there's already a midrule before Geographic (skip empty lines)
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        # Only insert if there isn't already a midrule right before
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)

    # Add table environment
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            "\\caption{Average improvement over JT (averaged over tasks) for TTT-MMR and TTT-MMR-Geo by architecture and test split. Values shown as mean $\\pm$ standard error. Improvement is calculated as normalized: $(R^2_{\\text{new}} - R^2_{\\text{old}}) / (1 - R^2_{\\text{old}})$.}\n" +
            "\\label{tab:ttt_by_model}\n" +
            "\\end{table}\n")

    with open('ttt_by_model.tex', 'w') as file:
        file.write(latex)

def tabulate_ttt_ranks_by_model():
    """Create a table showing average ranks of JT, TTT-MMR, and TTT-MMR-Geo by model.
    For each model, task, split, and seed, rank the three methods (1=best, 3=worst).
    First average ranks over tasks for each seed, then average those averages over seeds.
    Rows are methods (JT, TTT-MMR, TTT-MMR-Geo) for Random, then for Geographic (6 rows total).
    Columns are models. Shows mean ± SE of ranks.
    """
    adaptation_modes = ['JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    tags = ['pi_41', 'pi', 'pi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    all_runs = {}
    for tag in tags:
        all_runs[tag] = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})

    # Collect ranks per seed and task: {architecture: {split: {seed: {task: {mode: rank}}}}}
    rank_data = {architecture: {split: {seed: {task: {mode: None for mode in adaptation_modes} for task in tasks} for seed in seeds} for split in splits}
                 for architecture in architectures_plots}

    for architecture in architectures_plots:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'MAP'

            # Rank methods for each split and seed (1 = best, 3 = worst)
            for split in splits:
                metric_name = f'{split} test {metric}'

                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]

                    # Collect performance for each method for this seed
                    method_perfs = {}
                    for mode in adaptation_modes:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)
                        if run:
                            perf = run.summary_metrics.get(metric_name)
                            if perf is not None and not np.isnan(perf):
                                method_perfs[mode] = perf

                    # Rank methods for this seed if all three methods have data
                    if len(method_perfs) == 3:  # All three methods have data
                        # Sort by performance (descending, so best gets rank 1)
                        sorted_methods = sorted(method_perfs.items(), key=lambda x: x[1], reverse=True)

                        # Assign ranks (1 = best, 2 = middle, 3 = worst)
                        # Handle ties: if two methods have the same performance, they get the same rank
                        ranks = {}
                        current_rank = 1
                        prev_perf = None
                        for i, (mode, perf) in enumerate(sorted_methods):
                            if prev_perf is not None and abs(perf - prev_perf) < 1e-10:  # Tie
                                ranks[mode] = current_rank
                            else:
                                current_rank = i + 1
                                ranks[mode] = current_rank
                            prev_perf = perf

                        # Store ranks for this seed and task
                        for mode in adaptation_modes:
                            if mode in ranks:
                                rank_data[architecture][split][seed][task][mode] = ranks[mode]

    # First, average over tasks for each seed: {architecture: {split: {seed: {mode: avg_rank}}}}
    seed_avg_ranks = {architecture: {split: {seed: {mode: None for mode in adaptation_modes} for seed in seeds} for split in splits}
                      for architecture in architectures_plots}

    for architecture in architectures_plots:
        for split in splits:
            for seed in seeds:
                for mode in adaptation_modes:
                    # Collect ranks for this seed across all tasks
                    task_ranks = []
                    for task in tasks:
                        rank = rank_data[architecture][split][seed][task][mode]
                        if rank is not None:
                            task_ranks.append(rank)
                    # Average over tasks for this seed
                    if len(task_ranks) > 0:
                        seed_avg_ranks[architecture][split][seed][mode] = np.mean(task_ranks)
                    else:
                        seed_avg_ranks[architecture][split][seed][mode] = None

    # Then, average over seeds and calculate mean and standard error
    data_dict = {}
    display_name_mapping = {'JT': 'JT', 'JT-TTT': 'TTT-MMR', 'JT-TTT-Geo': 'TTT-MMR-Geo'}

    for split in splits:
        data_dict[split] = {}
        for mode in adaptation_modes:
            data_dict[split][mode] = {}
            for architecture in architectures_plots:
                # Collect seed-averaged ranks for the mean
                seed_avgs = []
                for seed in seeds:
                    avg_rank = seed_avg_ranks[architecture][split][seed][mode]
                    if avg_rank is not None:
                        seed_avgs.append(avg_rank)

                # Collect all individual ranks (across tasks and seeds) for the SE
                all_ranks = []
                for seed in seeds:
                    for task in tasks:
                        rank = rank_data[architecture][split][seed][task][mode]
                        if rank is not None:
                            all_ranks.append(rank)

                if len(seed_avgs) > 0:
                    mean = np.mean(seed_avgs)  # Mean: average over tasks per seed, then over seeds
                    # SE: computed from all individual ranks to reflect task-level variation
                    se = np.std(all_ranks, ddof=1) / np.sqrt(len(all_ranks)) if len(all_ranks) > 1 else 0.0
                    data_dict[split][mode][architecture] = {'mean': mean, 'se': se, 'n': len(seed_avgs)}
                else:
                    data_dict[split][mode][architecture] = {'mean': np.nan, 'se': np.nan, 'n': 0}

    # Find the method with the lowest mean for each split-architecture combination
    # This will be used to bold the best (lowest rank) value per column within each split
    best_methods = {}  # {(split, architecture): mode}
    for split in splits:
        for architecture in architectures_plots:
            best_mean = float('inf')
            best_mode = None
            for mode in adaptation_modes:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    if stats['mean'] < best_mean:
                        best_mean = stats['mean']
                        best_mode = mode
            if best_mode is not None:
                best_methods[(split, architecture)] = best_mode

    # Create DataFrame with mean ± SE format
    # Rows: JT, TTT-MMR, TTT-MMR-Geo (for Random), then JT, TTT-MMR, TTT-MMR-Geo (for Geographic)
    # Columns: Split, Method, then architectures
    display_decimals = 1
    formatted_data = {}
    split_column = []
    method_column = []
    row_keys = []

    # Create row labels and data with unique keys
    for split in splits:
        for mode in adaptation_modes:
            display_mode = display_name_mapping[mode]
            # Use unique key that includes split to avoid duplicates
            row_key = f"{split}_{display_mode}"
            row_keys.append(row_key)
            split_column.append(split)
            method_column.append(display_mode)
            formatted_data[row_key] = {}

            for architecture in architectures_plots:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    mean_str = f"{stats['mean']:.{display_decimals}f}"
                    se_str = f"{stats['se']:.{display_decimals}f}"
                    # Bold if this is the best (lowest mean) method for this split-architecture
                    if best_methods.get((split, architecture)) == mode:
                        formatted_data[row_key][architecture] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][architecture] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][architecture] = "--"

    # Create DataFrame
    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)  # Ensure correct row order
    df = df.reindex(architectures_plots, axis=1)  # Ensure correct column order
    df.columns = [display_arch_name(arch) for arch in architectures_plots]

    # Insert Split and Method columns at the beginning
    df.insert(0, 'Method', method_column)
    df.insert(0, 'Split', split_column)

    # Create LaTeX table
    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Method}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'r' * (len(df.columns) - 2),
                        na_rep='--')

    # Insert custom header and use multicolumn for Split column
    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    # Remove any existing midrule that might be right after toprule
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Add multicolumn for Split column to group rows
    # Find the data rows (after midrule)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    # Replace Split values with multicolumn for first row of each split group
    row_count = 0
    geographic_start_idx = None
    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                # First row of Random (row_count 0) and first row of Geographic (row_count 3)
                if row_count == 0:  # First Random row
                    parts[0] = f"\\multirow{{3}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == 3:  # First Geographic row
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{3}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""  # Empty for subsequent rows in group
                lines[i] = ' & '.join(parts)
                row_count += 1

    # Insert horizontal line before Geographic section
    if geographic_start_idx is not None:
        # Check if there's already a midrule before Geographic (skip empty lines)
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        # Only insert if there isn't already a midrule right before
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)

    # Add table environment
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            "\\caption{Average ranks of JT, TTT-MMR, and TTT-MMR-Geo by architecture (first averaged over tasks for each seed, then averaged over seeds for each test split). Ranks: 1 = best, 3 = worst. Values shown as mean $\\pm$ standard error.}\n" +
            "\\label{tab:ttt_ranks_by_model}\n" +
            "\\end{table}\n")

    with open('ttt_ranks_by_model.tex', 'w') as file:
        file.write(latex)

def calculate_ttt_delta_averages():
    """Calculate average delta (raw: new - old) ± SE for TTT and TTT-Geo vs JT,
    averaging over all models and tasks for Random and Geographic splits separately.

    Returns a dictionary with structure:
    {
        'Random': {
            'JT-TTT': {'mean': float, 'se': float, 'n': int},
            'JT-TTT-Geo': {'mean': float, 'se': float, 'n': int}
        },
        'Geographic': {...}
    }
    """

    adaptation_modes = ['JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']

    # Collect all deltas for each split and mode
    results = {}

    for split in splits:
        results[split] = {}

        for mode in adaptation_modes:
            all_deltas = []  # Collect all model-task deltas for this split-mode combination

            for architecture in architectures_plots:
                # Get JT baseline performance for each task
                jt_baseline = {}

                for task in tasks:
                    metric = 'R2' if task != 'species' else 'MAP'
                    run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)

                    if run:
                        metric_name = f'{split} test {metric}'
                        baseline_value = run.summary_metrics.get(metric_name)

                        if baseline_value is not None and not np.isnan(baseline_value):
                            jt_baseline[task] = baseline_value

                # Calculate deltas for this architecture
                for task in tasks:
                    if task in jt_baseline:
                        metric = 'R2' if task != 'species' else 'MAP'
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)

                        if run:
                            metric_name = f'{split} test {metric}'
                            performance = run.summary_metrics.get(metric_name)
                            baseline = jt_baseline[task]

                            if performance is not None and not np.isnan(performance) and not np.isnan(baseline):
                                # Calculate raw delta: new - old
                                delta = performance - baseline
                                all_deltas.append(delta)

            # Calculate mean and standard error
            if len(all_deltas) > 0:
                mean = np.mean(all_deltas)
                se = np.std(all_deltas, ddof=1) / np.sqrt(len(all_deltas)) if len(all_deltas) > 1 else 0.0
                results[split][mode] = {
                    'mean': mean,
                    'se': se,
                    'n': len(all_deltas)
                }
            else:
                results[split][mode] = {
                    'mean': np.nan,
                    'se': np.nan,
                    'n': 0
                }

    # Print results
    print("\n" + "="*80)
    print("TTT and TTT-Geo vs JT: Average Delta ± SE")
    print("="*80)

    for split in splits:
        print(f"\n{split} Split:")
        print("-" * 80)

        for mode in adaptation_modes:
            display_mode = 'MT-TTT' if mode == 'JT-TTT' else 'MT-TTT-Geo'
            stats = results[split][mode]

            if stats['n'] > 0:
                print(f"{display_mode}: {stats['mean']:.4f} ± {stats['se']:.4f} (n={stats['n']})")
            else:
                print(f"{display_mode}: N/A")

    print("\n" + "="*80 + "\n")

def compare_dinov3_architectures():
    """Compare DINOv3Web and DINOv3Sat averaged over tasks on 5%, 50%, and 100% training data,
    separately for Random and Geographic splits.

    Returns a dictionary with structure:
    {
        train_percent: {
            'Random': {
                'DINOv3Web': {'mean': float, 'se': float, 'n': int},
                'DINOv3Sat': {'mean': float, 'se': float, 'n': int}
            },
            'Geographic': {...}
        }
    }
    """

    architectures = ['DINOv3Web', 'DINOv3Sat']
    splits = ['Random', 'Geographic']
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    results = {}

    for train_percent in train_percents:
        results[train_percent] = {}

        for split in splits:
            results[train_percent][split] = {}

            for architecture in architectures:
                all_performances = []  # Collect all task performances for this architecture-split combination

                for task in tasks:
                    metric = 'R2' if task != 'species' else 'MAP'
                    run_name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)

                    if run:
                        metric_name = f'{split} test {metric}'
                        performance = run.summary_metrics.get(metric_name)

                        if performance is not None and not np.isnan(performance):
                            all_performances.append(performance)

                # Calculate mean and standard error over tasks
                if len(all_performances) > 0:
                    mean = np.mean(all_performances)
                    se = np.std(all_performances, ddof=1) / np.sqrt(len(all_performances)) if len(all_performances) > 1 else 0.0
                    results[train_percent][split][architecture] = {
                        'mean': mean,
                        'se': se,
                        'n': len(all_performances)
                    }
                else:
                    results[train_percent][split][architecture] = {
                        'mean': np.nan,
                        'se': np.nan,
                        'n': 0
                    }

    # Print results
    print("\n" + "="*80)
    print("DINOv3Web vs DINOv3Sat: Average Performance ± SE (averaged over tasks)")
    print("FT Adaptation Mode")
    print("="*80)

    for train_percent in train_percents:
        print(f"\n{train_percent}% Training Data:")
        print("=" * 80)

        for split in splits:
            print(f"\n{split} Split:")
            print("-" * 80)

            dinov3web_stats = results[train_percent][split]['DINOv3Web']
            dinov3sat_stats = results[train_percent][split]['DINOv3Sat']

            if dinov3web_stats['n'] > 0:
                print(f"DINOv3Web: {dinov3web_stats['mean']:.4f} ± {dinov3web_stats['se']:.4f} (n={dinov3web_stats['n']} tasks)")
            else:
                print(f"DINOv3Web: N/A")

            if dinov3sat_stats['n'] > 0:
                print(f"DINOv3Sat:  {dinov3sat_stats['mean']:.4f} ± {dinov3sat_stats['se']:.4f} (n={dinov3sat_stats['n']} tasks)")
            else:
                print(f"DINOv3Sat:  N/A")

            # Calculate difference if both are available
            if dinov3web_stats['n'] > 0 and dinov3sat_stats['n'] > 0:
                diff = dinov3sat_stats['mean'] - dinov3web_stats['mean']
                # Simple SE for difference (assuming independence)
                se_diff = np.sqrt(dinov3web_stats['se']**2 + dinov3sat_stats['se']**2) if dinov3web_stats['n'] > 1 and dinov3sat_stats['n'] > 1 else 0.0
                print(f"Difference (Sat - Web): {diff:.4f} ± {se_diff:.4f}")

    print("\n" + "="*80 + "\n")

    return results

if __name__ == '__main__':
    tabulate_results('Random', 'FT')
    tabulate_results('Geographic', 'FT')
    # tabulate_results('Random', 'LP')
    # tabulate_results('Geographic', 'LP')
    # tabulate_results_RQ3('Random')
    # tabulate_results_RQ3('Geographic')
    # tabulate_TTT_results()
    # tabulate_ttt_by_model()
    # tabulate_ttt_ranks_by_model()

    # plot_rq1_performance()
    # plot_rq2_performance()
    # plot_rq3_performance()
    # plot_ttt_improvement()
    # plot_ttt_improvement_normalized()
    # plot_rq3_performance('Random')
    # plot_rq3_performance('Geographic')

    # # Analyze performance drops for both test splits
    # random_results = calculate_rq1_stats('Random')
    # geographic_results = calculate_rq1_stats('Geographic')
    # rq2_results = calculate_rq2_stats()
    # rq3_results_random = calculate_rq3_stats('Random')
    # rq3_results_geographic = calculate_rq3_stats('Geographic')
    # calculate_ttt_delta_averages()
    # compare_dinov3_architectures()
