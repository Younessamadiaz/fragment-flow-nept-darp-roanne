# ============================================================
# SHARED STATISTICAL MACHINERY
#
# Factored out of run_replications.py (behavior-preserving relocation --
# identical formulas, just callable from more than one script) so
# run_algorithm_comparison.py can reuse the exact same descriptive-stats
# and paired-significance-test logic instead of a second hand-copied
# implementation that could silently drift.
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats as sstats


def describe(vals):
    """Mean/std, median/IQR, 95% CI (t-distribution), min/max."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = vals.mean()
    std = vals.std(ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    tcrit = sstats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    ci_halfwidth = tcrit * sem if n > 1 else float("nan")
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return dict(
        n=n, mean=mean, std=std, median=med, q1=q1, q3=q3, iqr=q3 - q1,
        ci95_low=mean - ci_halfwidth, ci95_high=mean + ci_halfwidth,
        ci95_halfwidth=ci_halfwidth, min=vals.min(), max=vals.max(),
    )


def aggregate_stats(raw_df, group_cols, metrics):
    """One row per (group..., metric) with describe()'s fields, plus a
    'feasible_rate' row per group if a boolean 'feasible' column is present.
    `group_cols` may be a single column name or a list of column names."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    rows = []
    for group_key, sub in raw_df.groupby(group_cols):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        group_dict = dict(zip(group_cols, group_key))

        for m in metrics:
            if m not in sub.columns:
                continue
            d = describe(sub[m])
            d.update(group_dict, metric=m)
            rows.append(d)

        if "feasible" in sub.columns:
            rows.append(dict(
                **group_dict, metric="feasible_rate", n=len(sub), mean=sub["feasible"].mean(),
                std=float("nan"), median=float("nan"), q1=float("nan"), q3=float("nan"),
                iqr=float("nan"), ci95_low=float("nan"), ci95_high=float("nan"),
                ci95_halfwidth=float("nan"), min=float("nan"), max=float("nan"),
            ))

    cols = group_cols + ["metric", "n", "mean", "std", "median", "q1", "q3", "iqr",
                          "ci95_low", "ci95_high", "ci95_halfwidth", "min", "max"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def paired_significance(raw_df, group_col, seed_col, value_col, baseline):
    """Paired t-test + Cohen's dz (parametric) and Wilcoxon signed-rank +
    matched-pairs rank-biserial correlation (non-parametric), comparing every
    other value of `group_col` against `baseline`, paired by `seed_col`
    (the same seed must have been run under both groups for a valid pair --
    e.g. group_col='config' within one instance, or group_col='algorithm'
    within one instance, baseline='lns')."""
    rows = []
    if baseline not in raw_df[group_col].unique():
        return pd.DataFrame(rows)

    baseline_vals = (raw_df[raw_df[group_col] == baseline]
                      .set_index(seed_col)[value_col].sort_index())

    for group_name in raw_df[group_col].unique():
        if group_name == baseline:
            continue
        cfg_vals = (raw_df[raw_df[group_col] == group_name]
                    .set_index(seed_col)[value_col].sort_index())
        common_seeds = baseline_vals.index.intersection(cfg_vals.index)
        if len(common_seeds) < 2:
            continue
        a = baseline_vals.loc[common_seeds].values
        b = cfg_vals.loc[common_seeds].values
        diff = b - a

        t_stat, t_p = sstats.ttest_rel(b, a)
        diff_std = diff.std(ddof=1)
        cohens_dz = diff.mean() / diff_std if diff_std > 0 else float("nan")

        try:
            w_stat, w_p = sstats.wilcoxon(b, a)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        nz = diff[diff != 0]
        if len(nz):
            ranks = sstats.rankdata(np.abs(nz))
            pos_sum = ranks[nz > 0].sum()
            neg_sum = ranks[nz < 0].sum()
            rank_biserial = (pos_sum - neg_sum) / (pos_sum + neg_sum)
        else:
            rank_biserial = 0.0

        rows.append(dict(
            baseline=baseline, comparison=group_name, n_pairs=len(diff),
            mean_diff=diff.mean(),
            pct_change_vs_baseline=100 * diff.mean() / a.mean() if a.mean() else float("nan"),
            paired_t_stat=t_stat, paired_t_pvalue=t_p, cohens_dz=cohens_dz,
            wilcoxon_stat=w_stat, wilcoxon_pvalue=w_p, rank_biserial_corr=rank_biserial,
            significant_at_0_05=bool(t_p < 0.05) if not np.isnan(t_p) else None,
        ))

    return pd.DataFrame(rows)
