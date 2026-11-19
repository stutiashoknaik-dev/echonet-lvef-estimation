#!/usr/bin/env python3
"""
Phase-selection evaluation: average absolute frame difference (aaFD) of the adaptive ED/ES selector,
compared with the simple max/min-area baseline, plus an optional weight sweep.

Inputs
  --signals  CSV with one row per (video, frame) computed from the PREDICTED masks:
                 video_id, frame, area, motion
             `area`   = predicted LV pixel count of the frame (before smoothing)
             `motion` = mean Farneback optical-flow magnitude between frame t-1 and t (0 for frame 0)
  --tracings EchoNet-Dynamic VolumeTracings.csv (columns FileName, X1, Y1, X2, Y2, Frame)
  --filelist EchoNet-Dynamic FileList.csv (to restrict to a split)
  --split    VAL or TEST (default TEST). Use VAL for any weight tuning.

Reference ED/ES: for each video the two traced frames are compared by traced area
(method-of-discs polygon), larger = ED, smaller = ES (standard EchoNet-Dynamic convention).

Usage
  python phase_eval.py --signals test_signals.csv --tracings VolumeTracings.csv --filelist FileList.csv --split TEST
  python phase_eval.py ... --split VAL --sweep      # weight sensitivity on the validation set
"""
import argparse
import numpy as np, pandas as pd
from scipy.signal import savgol_filter


def traced_area(g):
    # polygon from left points (X1,Y1) followed by right points (X2,Y2) reversed; first row is the long axis
    g = g.iloc[1:] if len(g) > 2 else g
    xs = np.concatenate([g.X1.values, g.X2.values[::-1]]); ys = np.concatenate([g.Y1.values, g.Y2.values[::-1]])
    return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def reference_frames(tr):
    ref = {}
    for fn, g in tr.groupby('FileName'):
        areas = {f: traced_area(gg) for f, gg in g.groupby('Frame')}
        if len(areas) < 2: continue
        fs = sorted(areas, key=areas.get)
        ref[fn.replace('.avi', '')] = {'ED': fs[-1], 'ES': fs[0]}
    return ref


def norm(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)


def select(area, motion, w_motion=0.4, w_area=0.5, win=11, order=2):
    a = area.astype(float)
    if len(a) >= win: a = savgol_filter(a, win, order)
    d = np.abs(np.gradient(a))
    an, dn, mn = norm(a), norm(d), norm(motion.astype(float))
    stab = 1 - (w_motion * mn + (1 - w_motion) * dn)
    s_ed = w_area * an + (1 - w_area) * stab
    s_es = w_area * (1 - an) + (1 - w_area) * stab
    return int(np.argmax(s_ed)), int(np.argmax(s_es)), a


def evaluate(sig, ref, w_motion, w_area, window=None):
    rows = []
    for vid, g in sig.groupby('video_id'):
        if vid not in ref: continue
        g = g.sort_values('frame')
        frames = g.frame.values
        if g.area.max() <= 0:   # failure case (empty masks)
            rows.append({'video_id': vid, 'failed': 1}); continue
        r = ref[vid]
        if window:  # restrict the search to the annotated heartbeat: +/- window frames around the traced pair
            c = (r['ED'] + r['ES']) / 2
            keep = (frames >= c - window) & (frames <= c + window)
            if keep.sum() >= 5:
                g = g[keep]; frames = g.frame.values
        ed, es, a = select(g.area.values, g.motion.values, w_motion, w_area)
        bed, bes = int(np.argmax(a)), int(np.argmin(a))
        rows.append({'video_id': vid, 'failed': 0,
                     'fd_ED': abs(frames[ed] - r['ED']), 'fd_ES': abs(frames[es] - r['ES']),
                     'fd_ED_base': abs(frames[bed] - r['ED']), 'fd_ES_base': abs(frames[bes] - r['ES'])})
    return pd.DataFrame(rows)


def summarise(df, label):
    ok = df[df.failed == 0]
    out = []
    for name, col in [('Adaptive selector', ''), ('Max/min-area baseline', '_base')]:
        row = {'method': name}
        for ph in ['ED', 'ES']:
            x = ok[f'fd_{ph}{col}']
            row[f'{ph} aaFD'] = x.mean(); row[f'{ph} median'] = x.median(); row[f'{ph} ≤3 frames (%)'] = 100 * (x <= 3).mean()
        out.append(row)
    t = pd.DataFrame(out)
    print(f'\n== {label}: n={len(ok)} valid, {int(df.failed.sum())} failed ==')
    print(t.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signals', required=True); ap.add_argument('--tracings', required=True)
    ap.add_argument('--filelist', required=True); ap.add_argument('--split', default='TEST')
    ap.add_argument('--w_motion', type=float, default=0.4); ap.add_argument('--w_area', type=float, default=0.5)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--window', type=int, default=None,
                    help='also evaluate with the search restricted to +/- WINDOW frames around the annotated beat (e.g. 20)')
    a = ap.parse_args()

    fl = pd.read_csv(a.filelist)
    keep = set(fl[fl.Split.str.upper() == a.split.upper()].FileName.astype(str).str.replace('.avi', ''))
    sig = pd.read_csv(a.signals); sig['video_id'] = sig.video_id.astype(str).str.replace('.avi', '')
    sig = sig[sig.video_id.isin(keep)]
    ref = reference_frames(pd.read_csv(a.tracings))

    df = evaluate(sig, ref, a.w_motion, a.w_area)
    summarise(df, f'{a.split} split, motion:derivative={a.w_motion:.1f}:{1-a.w_motion:.1f}, area:stability={a.w_area:.1f}:{1-a.w_area:.1f}')
    df.to_csv(f'phase_eval_{a.split.lower()}.csv', index=False)
    if a.window:
        dw = evaluate(sig, ref, a.w_motion, a.w_area, a.window)
        summarise(dw, f'{a.split} split, search restricted to annotated beat (+/-{a.window} frames)')
    print('\nNote: the reference contains only one traced ED/ES pair per video. If the selector picks the same phase in a\n'
          'different heartbeat, the frame difference is large although the phase is correct; report this caveat.')

    if a.sweep:
        print('\n== Weight sweep (mean aaFD over ED and ES; lower is better) ==')
        res = []
        for wm in [0.3, 0.4, 0.5, 0.6]:
            for wa in [0.3, 0.4, 0.5, 0.6, 0.7]:
                d = evaluate(sig, ref, wm, wa, a.window); ok = d[d.failed == 0]
                res.append({'motion:deriv': f'{wm:.1f}:{1-wm:.1f}', 'area:stab': f'{wa:.1f}:{1-wa:.1f}',
                            'ED aaFD': ok.fd_ED.mean(), 'ES aaFD': ok.fd_ES.mean(), 'mean aaFD': (ok.fd_ED.mean() + ok.fd_ES.mean()) / 2})
        print(pd.DataFrame(res).sort_values('mean aaFD').to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()
