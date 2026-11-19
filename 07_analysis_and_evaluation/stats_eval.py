#!/usr/bin/env python3
"""
Statistical evaluation of LVEF predictions for the (Tables 7-8, AUC, Wilcoxon, fallback).

Input CSVs (one row per test video), required columns:
    video_id, ef_true, ef_pred
Optional column:
    valid   (1 = phase selection succeeded, 0 = failed). Missing -> all valid.

Usage example:
    python stats_eval.py --model proposed=final_preds.csv --model video_only=bilstm_preds.csv \
        --model xgboost=xgb_preds.csv --reference proposed --fallback video_only --out results.json

What it reports:
  * MAE, RMSE, R2, Pearson r, mean bias with 95% bootstrap CIs (paired resampling of videos)
  * Bland-Altman 95% limits of agreement
  * MAE per LVEF range (<30, 30-40, 40-50, 50-60, 60-70, >70) and grouped (<40, 40-70, >70)
  * AUC for detecting LVEF < 40% and < 50% (with bootstrap CI)
  * Two-sided Wilcoxon signed-rank test on absolute errors, reference model vs every other model
    (computed on the videos common to both models)
  * Fallback analysis: reference model on all videos, with failed videos replaced by the fallback model
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score

BINS = [(-np.inf, 30, '<30'), (30, 40, '30-40'), (40, 50, '40-50'), (50, 60, '50-60'), (60, 70, '60-70'), (70, np.inf, '>70')]
GROUPS = [(-np.inf, 40, 'Reduced (<40)'), (40, 70, 'Mid (40-70)'), (70, np.inf, 'High (>70)')]


def metrics(y, p):
    e = p - y
    ss_res = np.sum(e ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return {'MAE': np.mean(np.abs(e)), 'RMSE': np.sqrt(np.mean(e ** 2)), 'R2': 1 - ss_res / ss_tot,
            'r': pearsonr(y, p)[0], 'bias': np.mean(e)}


def auc(y, p, thr):
    lab = (y < thr).astype(int)
    if lab.min() == lab.max():
        return np.nan
    return roc_auc_score(lab, -p)  # lower predicted EF => more likely reduced


def bootstrap(y, p, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y); out = {k: [] for k in ['MAE', 'RMSE', 'R2', 'r', 'bias', 'AUC<40', 'AUC<50']}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        m = metrics(y[i], p[i])
        for k, v in m.items():
            out[k].append(v)
        out['AUC<40'].append(auc(y[i], p[i], 40)); out['AUC<50'].append(auc(y[i], p[i], 50))
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in out.items()}


def binned(y, p, bins):
    rows = []
    for lo, hi, name in bins:  # half-open intervals [lo, hi); top bin includes 70
        m = (y >= lo) & (y < hi)
        rows.append({'range': name, 'n': int(m.sum()), 'share_%': 100 * m.mean(),
                     'MAE': float(np.mean(np.abs(p[m] - y[m]))) if m.any() else np.nan})
    return rows


def load(path):
    d = pd.read_csv(path)
    for c in ['video_id', 'ef_true', 'ef_pred']:
        if c not in d.columns:
            raise SystemExit(f'{path}: missing column {c}')
    if 'valid' not in d.columns:
        d['valid'] = 1
    return d


def full_report(name, d, n_boot, seed):
    v = d[d.valid == 1]
    y, p = v.ef_true.to_numpy(float), v.ef_pred.to_numpy(float)
    m = metrics(y, p); ci = bootstrap(y, p, n_boot, seed)
    sd = np.std(p - y, ddof=1)
    rep = {'n': len(v), **{k: float(x) for k, x in m.items()},
           'CI95': {k: [float(a), float(b)] for k, (a, b) in ci.items()},
           'AUC<40': float(auc(y, p, 40)), 'AUC<50': float(auc(y, p, 50)),
           'BA_LoA': [float(m['bias'] - 1.96 * sd), float(m['bias'] + 1.96 * sd)],
           'by_range': binned(y, p, BINS), 'by_group': binned(y, p, GROUPS)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='name=path.csv (repeatable)')
    ap.add_argument('--reference', required=True, help='name of the proposed model')
    ap.add_argument('--fallback', help='name of model whose predictions replace failed videos of the reference')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='stats_results.json')
    a = ap.parse_args()

    data = {kv.split('=', 1)[0]: load(kv.split('=', 1)[1]) for kv in a.model}
    results = {name: full_report(name, d, a.n_boot, a.seed) for name, d in data.items()}

    # Wilcoxon: reference vs others, on common valid videos
    ref = data[a.reference]; ref = ref[ref.valid == 1].set_index('video_id')
    tests = {}
    for name, d in data.items():
        if name == a.reference: continue
        o = d[d.valid == 1].set_index('video_id')
        common = ref.index.intersection(o.index)
        er = np.abs(ref.loc[common, 'ef_pred'] - ref.loc[common, 'ef_true'])
        eo = np.abs(o.loc[common, 'ef_pred'] - o.loc[common, 'ef_true'])
        stat, pval = wilcoxon(er, eo, alternative='two-sided')
        tests[f'{a.reference} vs {name}'] = {'n_common': int(len(common)), 'MAE_ref': float(er.mean()),
                                              'MAE_other': float(eo.mean()), 'W': float(stat), 'p': float(pval)}
    results['wilcoxon_abs_error'] = tests

    if a.fallback:
        fb = data[a.fallback].set_index('video_id')
        r = data[a.reference].set_index('video_id').copy()
        bad = r.valid == 0
        r.loc[bad, 'ef_pred'] = fb.loc[r.index[bad], 'ef_pred']
        r['valid'] = 1
        results[f'{a.reference}_with_fallback_all_videos'] = full_report('fb', r.reset_index(), a.n_boot, a.seed)

    with open(a.out, 'w') as f:
        json.dump(results, f, indent=2)

    # human-readable summary
    print(f"\n{'Model':38s} {'n':>5s} {'MAE [95% CI]':>22s} {'RMSE [95% CI]':>22s} {'R2 [95% CI]':>24s} {'r':>6s} {'bias':>7s} {'AUC<40':>7s} {'AUC<50':>7s}")
    for name, rep in results.items():
        if not isinstance(rep, dict) or 'MAE' not in rep: continue
        c = rep['CI95']
        print(f"{name:38s} {rep['n']:5d} {rep['MAE']:6.2f} [{c['MAE'][0]:.2f}, {c['MAE'][1]:.2f}]   "
              f"{rep['RMSE']:6.2f} [{c['RMSE'][0]:.2f}, {c['RMSE'][1]:.2f}]   {rep['R2']:6.3f} [{c['R2'][0]:.3f}, {c['R2'][1]:.3f}] "
              f"{rep['r']:6.3f} {rep['bias']:+7.2f} {rep['AUC<40']:7.3f} {rep['AUC<50']:7.3f}")
    print('\nWilcoxon signed-rank on absolute errors:')
    for k, t in tests.items():
        print(f"  {k}: n={t['n_common']}, MAE {t['MAE_ref']:.2f} vs {t['MAE_other']:.2f}, p={t['p']:.2e}")
    print(f'\nFull results (incl. range tables, LoA, AUC CIs) written to {a.out}')


if __name__ == '__main__':
    main()
