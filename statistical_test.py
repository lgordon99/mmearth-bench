from scipy.stats import wilcoxon
import numpy as np
import wandb
import utils

entity = utils.read_yaml("config-user.yml")["entity"]
project = utils.read_yaml("config-user.yml")["project"]
tasks = ["biomass", "soil_nitrogen", "soil_organic_carbon", "soil_pH", "species"]
architectures_plots = ["ConvNeXtV2A", "ScaleMAE", "DINOv3Web", "DINOv3Sat", "SatlasNet", "MPMAE", "TerraMind", "CopernicusFM", 'Galileo', 'ConvNeXtV2AMM']
splits = ["Random", "Geographic"]
modes = ["JT-TTT", "JT-TTT-Geo"]
tags = ["chi_41", "chi_42", "chi_43"]
display_name_mapping = {"JT-TTT": "TTT-MMR", "JT-TTT-Geo": "TTT-MMR-Geo"}

def find_run(runs, task, architecture, mode):
    run_name_prefix = "_".join([task, architecture, mode, '100']) + "_"
    return next((r for r in runs if r.name.startswith(run_name_prefix)), None)

def collect_deltas():
    """
    Returns:
      deltas_by_arch_seed[(task, split, mode)] -> list of deltas treating each (arch, seed) as an observation
    """

    all_runs_list = wandb.Api().runs(f"{entity}/{project}", filters={"tags": {"$in": tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    deltas_by_arch_seed = {(t, s, m): [] for t in tasks for s in splits for m in modes}

    for tag in tags:
        runs = all_runs[tag]

        for task in tasks:
            metric = "mAP" if task == "species" else "R2"

            for split in splits:
                metric_name = f"{split} test {metric}"

                # JT baseline per architecture for this seed
                jt = {}
                for arch in architectures_plots:
                    r = find_run(runs, task, arch, "JT")
                    v = r.summary_metrics.get(metric_name)
                    jt[arch] = float(v)

                for mode in modes:
                    for arch in architectures_plots:
                        r = find_run(runs, task, arch, mode)
                        v = r.summary_metrics.get(metric_name)
                        delta = float(v) - float(jt[arch])
                        deltas_by_arch_seed[(task, split, mode)].append(delta)

    return deltas_by_arch_seed

def run_wilcoxon(deltas):
    """
    deltas: list of paired differences (mode - JT)
    Tests whether TTT improves over JT (one-sided test: median Δ > 0).
    """
    x = np.asarray(deltas, dtype=float)
    stat, p = wilcoxon(x, zero_method="wilcox", alternative="greater", correction=False, method="auto")
    return dict(n=len(x), median=float(np.median(x)), mean=float(np.mean(x)),
                p=float(p), stat=float(stat), pos_frac=float(np.mean(x > 0)))

def holm_bonferroni(pvals):
    """Return Holm-Bonferroni adjusted p-values (same order as input)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    # Holm step-down: adjusted_i = max_{j<=i} (m-j) * p_(j)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        running_max = max(running_max, factor * pvals[idx])
        adjusted[idx] = min(1.0, running_max)

    return adjusted

if __name__ == "__main__":
    deltas_by_arch_seed = collect_deltas()

    print("\n=== Wilcoxon signed-rank on Δ = (mode - JT) ===")
    print(f"Analysis: treat each (architecture, seed) as an observation (n={len(architectures_plots)*len(tags)} per test).")
    print("Testing whether TTT improves over JT (one-sided: median Δ > 0).\n")

    results = []
    for task in tasks:
        for split in splits:
            for mode in modes:
                key = (task, split, mode)
                r = run_wilcoxon(deltas_by_arch_seed[key])
                results.append((key, r))

    # Multiple-comparisons correction across the 20 tests (task x split x mode)
    ps = [r["p"] if r is not None else np.nan for _, r in results]
    valid_idx = [i for i, p in enumerate(ps) if not np.isnan(p)]
    adjusted = np.full(len(ps), np.nan, dtype=float)
    if valid_idx:
        adjusted_vals = holm_bonferroni([ps[i] for i in valid_idx])
        for i, ap in zip(valid_idx, adjusted_vals):
            adjusted[i] = ap

    print("Alternative = 'greater' (one-sided test: TTT improves over JT)")
    print("=" * 100)
    print(f"{'Task':<20s}  {'Split':<10s} {'Mode':<12s}  n   medianΔ    meanΔ      p       p_holm   pos_frac")
    print("-" * 100)
    p_threshold = 0.05

    for i, (key, r) in enumerate(results):
        task, split, mode = key
        mode_display = display_name_mapping.get(mode, mode)
        sig = "*" if adjusted[i] < p_threshold else " "
        print(
            f"{task:20s}  {split:10s} {mode_display:12s}  "
            f"{r['n']:<3d} {r['median']:+.4f}   {r['mean']:+.4f}   "
            f"{r['p']:.4f}   {adjusted[i]:.4f}{sig}  {r['pos_frac']:.2f}"
        )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    n_sig = sum(1 for p in adjusted if not np.isnan(p) and p < p_threshold)
    print(f"{n_sig}/{len(valid_idx)} tests significant at α={p_threshold} (Holm-corrected)")
    print(f"\n* = significant at α={p_threshold} after Holm-Bonferroni correction")
