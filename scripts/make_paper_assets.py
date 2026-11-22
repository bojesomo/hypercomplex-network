"""Build every results table, figure and number macro used by the LaTeX paper sections.

    python scripts/make_paper_assets.py                       # scans runs*/ -> paper/tables, paper/figures
    python scripts/make_paper_assets.py --runs runs runs_teacher runs_proto --out paper

Inputs: every ``<root>/<run>/result.json`` written by train.py (plus ``log.jsonl``).
Repeated runs are averaged: a run whose name ends in ``_s<k>``, ``_seed<k>`` or ``_fold<k>``
belongs to the group named without that suffix, and is reported as mean ± std over the group.

Run-name conventions (as produced by scripts/run_ablations.sh, run_teacher_ablation.sh and
run_protocols.sh):
    main        student_hyper_q_kd (proposed), student_hyper_q (no KD), uffia_like_late_concat_real
    ablations   A_fs*, B_n*, C_*, D_*, G_*
    teachers    teacher_T*, student_from_T*, student_no_kd
    protocols   P_uffia, P_kfold, P_grouped
Anything missing is printed as "--" in the tables, so the paper always compiles.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

CLASSES = ['None', 'Strong', 'Medium', 'Weak']  # U-FFIA label order (0..3)
SNRS = [-10, -5, 0, 10, 20]

# Published numbers (accuracy in %, params in M, compute in GMACs; U-FFIA's "FLOPs" are MACs).
# src: V = verified against the primary source, S = search snippet only (check before submission).
PUBLISHED = [
    # name, modality, params, gmacs, acc, f1, protocol, cite, src
    ('Pre-CNN10', 'A', 5.0, 3.347, 85.9, None, 'paper split', 'cui2024uffia', 'V'),
    ('Pretrained-S3D', 'V', 7.9, 22.826, 89.8, None, 'paper split', 'cui2024uffia', 'V'),
    ('Self-attention', 'AV', 47.695, 68.404, 89.2, None, 'paper split', 'cui2024uffia', 'V'),
    ('MBT', 'AV', 47.689, 68.690, 72.4, None, 'paper split', 'cui2024uffia', 'V'),
    ('Cross-attention', 'AV', 35.082, 60.933, 91.7, None, 'paper split', 'cui2024uffia', 'V'),
    ('U-FFIA + KD', 'AV', 21.626, 15.488, 90.7, None, 'paper split', 'cui2024uffia', 'V'),
    ('ACAF-Net', 'AV', None, None, 95.4, None, 'random split', 'acafnet2025', 'S'),
    ('DynaGate-AV (Light)', 'AV', None, 5.29, 96.4, None, 'official split, 19,963 clips', 'dynagate2026', 'V'),
    ('HMMF-Net', 'AV', 3.987, 1.002, 96.77, 96.88, '5-fold CV', 'hmmfnet2026', 'S'),
    ('HMKD', 'AV', None, None, 97.47, None, '26,843-pair holdout', 'hmkd2026', 'S'),
    ('LDMF', 'AV', 7.10, None, 98.22, None, 'filtered subset', 'ldmf2026', 'S'),
]
PUB_ROBUST = {  # U-FFIA Table III (verified): clean, then SNR -10, -5, 0, 10, 20 dB
    'Cross-attention': [91.7, 72.7, 76.8, 80.5, 83.4, 86.7],
    'U-FFIA + KD': [90.7, 72.6, 77.9, 80.6, 83.5, 87.6],
    'Self-attention': [89.2, 71.3, 75.1, 77.4, 82.6, 86.3],
}

# Reference palette (dataviz skill, light mode), categorical slots in fixed order; validated.
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # blue, light -> dark


# ----------------------------------------------------------------------------- loading
def load_runs(roots):
    runs = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, '*', 'result.json'))):
            d = json.load(open(p))
            r = d['result']
            r['_dir'] = os.path.dirname(p)
            r['_log'] = d.get('log', [])
            runs.append(r)
    return runs


def group_key(name):
    return re.sub(r'(_s(eed)?\d+|_fold\d+)$', '', name or '')


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, s, len(xs)


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r['name'])].append(r)
    out = {}
    for g, rs in groups.items():
        t = [r['test'] for r in rs]
        a = {
            'n': len(rs),
            'acc': mean_std([100 * x['acc'] for x in t]),
            'f1': mean_std([100 * x['macro_f1'] for x in t]),
            'acc_a': mean_std([100 * x['audio_only_acc'] for x in t if 'audio_only_acc' in x]),
            'acc_v': mean_std([100 * x['video_only_acc'] for x in t if 'video_only_acc' in x]),
            'params': rs[0]['params_total'] / 1e6,
            'trainable': rs[0]['params_trainable'] / 1e6,
            'gmacs': rs[0].get('gmacs', rs[0]['gflops'] / 2 if isinstance(rs[0].get('gflops'), (int, float)) else None),
            'per_class_f1': [mean_std([100 * x['per_class_f1'][c] for x in t])[0] for c in range(len(CLASSES))],
            'confusion': [[sum(x['confusion'][i][j] for x in t) for j in range(len(CLASSES))] for i in range(len(CLASSES))],
            'runs': rs,
        }
        rob = defaultdict(list)
        for x in t:
            for k, v in x.get('robustness', {}).items():
                rob[k].append(100 * v)
        a['robust'] = {k: mean_std(v) for k, v in rob.items()}
        out[g] = a
    return out


# ----------------------------------------------------------------------------- formatting
def pm(ms, digits=1):
    m, s, n = ms if isinstance(ms, tuple) else (ms, None, 1)
    if m is None:
        return '--'
    return f'{m:.{digits}f}' + (f' $\\pm$ {s:.{digits}f}' if s is not None else '')


def num(x, digits=2):
    return '--' if x is None else f'{x:.{digits}f}'


def get(G, name, field='acc'):
    return G[name][field] if name in G else (None, None, 0)


def tabular(path, cols, header, rows, midrules=()):
    lines = [f'\\begin{{tabular}}{{{cols}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append('\\midrule')
        lines.append(' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ----------------------------------------------------------------------------- tables
OURS = 'student_hyper_q_kd'


def make_tables(G, out):
    T = os.path.join(out, 'tables')
    os.makedirs(T, exist_ok=True)

    # 1. main comparison
    rows = []
    for name, mod, p, g, acc, f1, prot, cite, src in PUBLISHED:
        mark = '' if src == 'V' else '$^\\dagger$'
        rows.append([f'{name}~\\cite{{{cite}}}{mark}', mod, num(p), num(g, 3), f'{acc:.1f}',
                     '--' if f1 is None else f'{f1:.1f}', prot])
    ours = [('uffia_like_late_concat_real', 'Real late-concat baseline (ours, re-run)'),
            ('teacher_panns_s3d', 'Teacher: CNN10 + S3D + HCF'),
            ('teacher_efficientat_movinet_a0', 'Teacher: EfficientAT + MoViNet-A0 + HCF'),
            ('student_hyper_q', 'HyperAVNet-Q (no KD)'),
            (OURS, '\\textbf{HyperAVNet-Q (KD)}'),
            ('student_hyper_o_kd', 'HyperAVNet-O (KD)')]
    first = len(rows)
    for g, label in ours:
        a = G.get(g)
        rows.append([label, 'AV', num(a and a['params']), num(a and a['gmacs'], 3),
                     pm(get(G, g)), pm(get(G, g, 'f1')), 'official split'])
    tabular(os.path.join(T, 'main.tex'), 'llrrrrl',
            ['Method', 'Mod.', 'Params (M)', 'GMACs', 'Acc. (\\%)', 'F1 (\\%)', 'Protocol'], rows, midrules=(6, first))

    # 2. fusion depth (deep -> early) and feedback
    depth = [('A_fs3', '$\\{3\\}$ (late)'), ('A_fs23', '$\\{2,3\\}$'), ('A_fs123', '$\\{1,2,3\\}$'),
             ('A_fs0123', '$\\{0,1,2,3\\}$'), ('A_fs0', '$\\{0\\}$'), ('A_fs1', '$\\{1\\}$'), ('A_fs2', '$\\{2\\}$'),
             ('A_fs123_noinj', '$\\{1,2,3\\}$ w/o feedback')]
    rows = [[lab, num(G[g]['params'] if g in G else None), num(G[g]['gmacs'] if g in G else None, 3),
             pm(get(G, g)), pm(get(G, g, 'f1')), pm(get(G, g, 'acc_a')), pm(get(G, g, 'acc_v'))] for g, lab in depth]
    tabular(os.path.join(T, 'ablation_depth.tex'), 'lrrrrrr',
            ['Fusion stages $\\mathcal{S}$', 'Params (M)', 'GMACs', 'Acc.', 'F1', 'A-only', 'V-only'], rows, midrules=(4, 7))

    # 3. algebra
    alg = [('B_n1', 'Real ($n{=}1$)'), ('B_n2', 'Complex ($n{=}2$)'), ('B_n4', 'Quaternion ($n{=}4$)'),
           ('B_n8', 'Octonion ($n{=}8$)'), ('B_n16', 'Sedenion ($n{=}16$)')]
    rows = [[lab, num(G[g]['params'] if g in G else None), num(G[g]['gmacs'] if g in G else None, 3),
             pm(get(G, g)), pm(get(G, g, 'f1'))] for g, lab in alg]
    tabular(os.path.join(T, 'ablation_algebra.tex'), 'lrrrr', ['Algebra', 'Params (M)', 'GMACs', 'Acc.', 'F1'], rows)

    # 4. fusion operator + regularisers + front-end
    ops = [('C_hcf_c', 'HCF, complex components'), ('C_hcf_q', 'HCF, quaternion components'),
           ('C_hcf_o', 'HCF, octonion components'), ('C_mbt', 'Hypercomplex MBT'), ('C_concat', 'Gated concatenation'),
           ('D_no_moddrop', 'w/o modality dropout ($p{=}0$)'), ('D_moddrop25', 'modality dropout $p{=}0.25$'),
           ('D_no_aux', 'w/o auxiliary heads'), ('G_simpf05', 'SimPF front-end ($k{=}0.5$)')]
    rows = [[lab, num(G[g]['params'] if g in G else None), pm(get(G, g)), pm(get(G, g, 'f1')),
             pm(get(G, g, 'acc_a')), pm(get(G, g, 'acc_v'))] for g, lab in ops]
    tabular(os.path.join(T, 'ablation_fusion_reg.tex'), 'lrrrrr',
            ['Variant', 'Params (M)', 'Acc.', 'F1', 'A-only', 'V-only'], rows, midrules=(5, 8))

    # 5. teachers
    teachers = sorted({k[len('teacher_'):] for k in G if k.startswith('teacher_T')}, key=lambda s: int(re.findall(r'\d+', s)[0]))
    labels = {'T1_cnn10_s3d': 'CNN10 / S3D', 'T2_cnn14_s3d': 'CNN14 / S3D', 'T3_cnn10_r2plus1d': 'CNN10 / R(2+1)D-18',
              'T4_cnn10_convnext_frames': 'CNN10 / ConvNeXt-T (frames)', 'T5_effnet_mel_s3d': 'EffNet-B0 (mel) / S3D',
              'T6_ast_videomae': 'AST / VideoMAE-B', 'T7_cnn10_x3d_m': 'CNN10 / X3D-M', 'T8_ast_s3d': 'AST / S3D',
              'T9_cnn10_videomae': 'CNN10 / VideoMAE-B', 'T10_efficientat_movinet_a0': 'EfficientAT / MoViNet-A0'}
    if not teachers:
        teachers = list(labels)
    rows = []
    for t in teachers:
        tg, sg = f'teacher_{t}', f'student_from_{t}'
        ta, sa = get(G, tg), get(G, sg)
        gap = (sa[0] - ta[0], None, 1) if ta[0] is not None and sa[0] is not None else (None, None, 0)
        rows.append([t.split('_')[0], labels.get(t, t.replace('_', '\_')), num(G[tg]['params'] if tg in G else None, 1),
                     num(G[tg]['gmacs'] if tg in G else None, 1), pm(ta), pm(get(G, tg, 'acc_a')),
                     pm(get(G, tg, 'acc_v')), pm(sa), pm(gap)])
    rows.append(['--', 'no teacher (no KD)', '--', '--', '--', '--', '--', pm(get(G, 'student_no_kd')), '--'])
    tabular(os.path.join(T, 'teachers.tex'), 'llrrrrrrr',
            ['ID', 'Audio / video teacher', 'Params', 'GMACs', 'Teacher AV', 'A-only', 'V-only', 'Student', '$\\Delta$'],
            rows, midrules=(len(rows) - 1,))

    # 6. robustness (U-FFIA Table III format) + video corruptions + missing modality
    rows = []
    for name, vals in PUB_ROBUST.items():
        rows.append([f'{name}~\\cite{{cui2024uffia}}'] + [f'{v:.1f}' for v in vals] + ['--'] * 4)
    for g, lab in [('teacher_panns_s3d', 'Teacher (CNN10 + S3D)'), (OURS, '\\textbf{HyperAVNet-Q (KD)}')]:
        a = G.get(g, {'robust': {}})
        R = a['robust']
        snr = [pm(next((v for k, v in R.items() if k.endswith(f'snr{s:+d}dB')), (None, None, 0))) for s in SNRS]
        rows.append([lab, pm(get(G, g))] + snr + [pm(R.get('video_dark_x0.25', (None, None, 0))),
                                                  pm(R.get('video_gauss_var0.2', (None, None, 0))),
                                                  pm(R.get('missing_audio', (None, None, 0))),
                                                  pm(R.get('missing_video', (None, None, 0)))])
    tabular(os.path.join(T, 'robustness.tex'), 'lrrrrrrrrrr',
            ['Method', 'Clean'] + [f'{s:+d}\\,dB' for s in SNRS] + ['Dark', 'Noise', '$-$A', '$-$V'],
            rows, midrules=(len(PUB_ROBUST),))

    # 7. protocols
    prot = [('P_uffia', 'Official code split (700/700 per class)'), ('P_kfold', '5-fold stratified CV'),
            ('P_grouped', 'Session-disjoint (whole recording dates)')]
    rows = [[lab, str(G[g]['n']) if g in G else '--', pm(get(G, g)), pm(get(G, g, 'f1'))] for g, lab in prot]
    tabular(os.path.join(T, 'protocols.tex'), 'lrrr', ['Evaluation protocol', 'Runs', 'Acc.', 'F1'], rows)

    # 8. per-class F1 of the proposed model
    a = G.get(OURS)
    rows = [[c, num(a['per_class_f1'][i], 1) if a else '--'] for i, c in enumerate(CLASSES)]
    tabular(os.path.join(T, 'per_class.tex'), 'lr', ['Class', 'F1 (\\%)'], rows)


# ----------------------------------------------------------------------------- macros
def make_macros(G, out):
    def v(name, field='acc', digits=1):
        m = get(G, name, field)[0]
        return '--' if m is None else f'{m:.{digits}f}'
    a = G.get(OURS)
    best_depth = max((g for g in G if g.startswith('A_fs') and 'noinj' not in g), key=lambda g: G[g]['acc'][0] or -1, default=None)
    depth_name = {'A_fs3': '\\{3\\}', 'A_fs23': '\\{2,3\\}', 'A_fs123': '\\{1,2,3\\}', 'A_fs0123': '\\{0,1,2,3\\}',
                  'A_fs0': '\\{0\\}', 'A_fs1': '\\{1\\}', 'A_fs2': '\\{2\\}'}.get(best_depth, '--')
    kd_gain = '--'
    if get(G, OURS)[0] is not None and get(G, 'student_hyper_q')[0] is not None:
        kd_gain = f"{get(G, OURS)[0] - get(G, 'student_hyper_q')[0]:+.1f}"
    m = {
        'OurAcc': v(OURS), 'OurFone': v(OURS, 'f1'), 'OurAccA': v(OURS, 'acc_a'), 'OurAccV': v(OURS, 'acc_v'),
        'OurParams': num(a['params']) if a else '2.14', 'OurGMACs': num(a['gmacs']) if a else '1.18',
        'NoKDAcc': v('student_hyper_q'), 'KDGain': kd_gain, 'TeacherAcc': v('teacher_panns_s3d'),
        'RealAcc': v('B_n1'), 'QuatAcc': v('B_n4'), 'OctAcc': v('B_n8'), 'LateAcc': v('A_fs3'),
        'ProgAcc': v('A_fs123'), 'NoFeedbackAcc': v('A_fs123_noinj'), 'BestDepth': depth_name,
        'MBTAcc': v('C_mbt'), 'HCFAcc': v('C_hcf_q'), 'OfficialAcc': v('P_uffia'), 'KfoldAcc': v('P_kfold'),
        'GroupedAcc': v('P_grouped'),
    }
    with open(os.path.join(out, 'tables', 'macros.tex'), 'w') as f:
        f.write('% generated by scripts/make_paper_assets.py -- do not edit by hand\n')
        for k, val in m.items():
            f.write(f'\\providecommand{{\\res{k}}}{{{val}}}\\renewcommand{{\\res{k}}}{{{val}}}\n')


# ----------------------------------------------------------------------------- figures
def _style(plt):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': MUTED,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'lines.linewidth': 1.5, 'lines.markersize': 5, 'legend.frameon': False, 'pdf.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def make_figures(G, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    F = os.path.join(out, 'figures')
    os.makedirs(F, exist_ok=True)

    # Fig. A: accuracy vs compute (log x). Published points from the literature, ours if available.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    early = [(n, g, a) for n, mod, p, g, a, *_ in PUBLISHED if mod == 'AV' and g and n in PUB_ROBUST or n in ('MBT',)]
    later = [(n, g, a) for n, mod, p, g, a, f1, prot, cite, src in PUBLISHED if mod == 'AV' and g and n not in PUB_ROBUST and n != 'MBT']
    ax.scatter([g for _, g, _ in early], [a for *_, a in early], marker='s', s=26, color=C2, label='U-FFIA paper', zorder=3,
               edgecolor='white', linewidth=0.8)
    ax.scatter([g for _, g, _ in later], [a for *_, a in later], marker='^', s=30, color=C3, label='Later work (own protocols)',
               zorder=3, edgecolor='white', linewidth=0.8)
    off = {'HMMF-Net': (-4, 5, 'right'), 'DynaGate-AV (Light)': (4, 4, 'left'), 'MBT': (-5, -2, 'right'),
           'Self-attention': (-5, -7, 'right'), 'Cross-attention': (-5, 3, 'right'), 'U-FFIA + KD': (5, -2, 'left')}
    for n, g, a in early + later:
        dx, dy, ha = off.get(n, (4, -3, 'left'))
        ax.annotate(n, (g, a), xytext=(dx, dy), textcoords='offset points', fontsize=6, color=MUTED, ha=ha)
    ours = [(lab, G[g]) for g, lab in [(OURS, 'HyperAVNet-Q (KD)'), ('student_hyper_o_kd', 'HyperAVNet-O (KD)'),
                                       ('teacher_panns_s3d', 'Our teacher')] if g in G and G[g]['gmacs']]
    if ours:
        ax.errorbar([a['gmacs'] for _, a in ours], [a['acc'][0] for _, a in ours],
                    yerr=[a['acc'][1] or 0 for _, a in ours], fmt='o', color=C1, ms=5, capsize=2, label='Ours', zorder=4)
        for i, (lab, a) in enumerate(ours):  # stagger labels of nearby points
            ax.annotate(lab, (a['gmacs'], a['acc'][0]), xytext=(6, -4 - 8 * i), textcoords='offset points',
                        fontsize=6, color=INK)
    ax.set_xscale('log')
    ax.set_xlabel('Compute per 2 s clip (GMACs, log scale)')
    ax.set_ylabel('AV accuracy (%)')
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(F, 'acc_vs_compute.pdf'))
    plt.close(fig)

    # Fig. B: robustness to audio noise (published U-FFIA curves + ours)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    styles = [(C2, 's'), (C3, '^'), (C4, 'D')]
    for (name, vals), (c, mk) in zip(PUB_ROBUST.items(), styles):
        ax.plot(SNRS, vals[1:], color=c, marker=mk, label=f'{name} [U-FFIA]')
    for g, lab in [(OURS, 'HyperAVNet-Q (KD)')]:
        if g in G and G[g]['robust']:
            R = G[g]['robust']
            pts = [(s, next((v for k, v in R.items() if k.endswith(f'snr{s:+d}dB')), (None,))[0]) for s in SNRS]
            pts = [(s, y) for s, y in pts if y is not None]
            if pts:
                kind = 'recorded' if any('recorded' in k for k in R) else 'white'
                ax.plot([s for s, _ in pts], [y for _, y in pts], color=C1, marker='o', label=f'{lab} ({kind} noise)')
    ax.set_xlabel('Audio SNR (dB)')
    ax.set_ylabel('AV accuracy (%)')
    ax.set_xticks(SNRS)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.savefig(os.path.join(F, 'robustness_snr.pdf'))
    plt.close(fig)

    # Fig. C: fusion depth (deep -> early), with and without feedback
    depth = [('A_fs3', '{3}'), ('A_fs23', '{2,3}'), ('A_fs123', '{1,2,3}'), ('A_fs0123', '{0,1,2,3}')]
    have = [(lab, G[g]) for g, lab in depth if g in G]
    if have:
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        xs = range(len(have))
        ax.errorbar(list(xs), [a['acc'][0] for _, a in have], yerr=[a['acc'][1] or 0 for _, a in have],
                    color=C1, marker='o', capsize=2, label='with feedback')
        if 'A_fs123_noinj' in G and 'A_fs123' in G:
            i = [lab for lab, _ in have].index('{1,2,3}') if '{1,2,3}' in [lab for lab, _ in have] else None
            if i is not None:
                a = G['A_fs123_noinj']
                ax.errorbar([i + 0.12], [a['acc'][0]], yerr=[a['acc'][1] or 0], color=C2, marker='s', capsize=2,
                            linestyle='none', label='without feedback')
        ax.set_xticks(list(xs), [lab for lab, _ in have])
        ax.set_xlabel('Fusion stages (late → early)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(F, 'fusion_depth.pdf'))
        plt.close(fig)

    # Fig. D: algebra -- accuracy vs parameter count
    alg = [('B_n1', 'R'), ('B_n2', 'C'), ('B_n4', 'H'), ('B_n8', 'O'), ('B_n16', 'S')]
    have = [(lab, G[g]) for g, lab in alg if g in G]
    if have:
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ax.errorbar([a['params'] for _, a in have], [a['acc'][0] for _, a in have],
                    yerr=[a['acc'][1] or 0 for _, a in have], color=C1, marker='o', capsize=2)
        names = {'R': 'real', 'C': 'complex', 'H': 'quaternion', 'O': 'octonion', 'S': 'sedenion'}
        for lab, a in have:
            ax.annotate(names[lab], (a['params'], a['acc'][0]), xytext=(4, 3), textcoords='offset points', fontsize=6, color=INK)
        ax.set_xscale('log')
        ax.set_xlabel('Parameters (M, log scale)')
        ax.set_ylabel('Accuracy (%)')
        fig.savefig(os.path.join(F, 'algebra.pdf'))
        plt.close(fig)

    # Fig. E: teacher accuracy vs distilled student accuracy
    pairs = [(k[len('teacher_'):], G[k], G.get('student_from_' + k[len('teacher_'):])) for k in G if k.startswith('teacher_T')]
    pairs = [(t, ta, sa) for t, ta, sa in pairs if sa]
    if pairs:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.scatter([ta['acc'][0] for _, ta, _ in pairs], [sa['acc'][0] for *_, sa in pairs], color=C1, s=22, zorder=3)
        for t, ta, sa in pairs:
            ax.annotate(t.split('_')[0], (ta['acc'][0], sa['acc'][0]), xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=INK)
        if 'student_no_kd' in G:
            ax.axhline(G['student_no_kd']['acc'][0], color=MUTED, linestyle='--', linewidth=1)
            ax.annotate('student without KD', (ax.get_xlim()[0], G['student_no_kd']['acc'][0]), xytext=(2, 2),
                        textcoords='offset points', fontsize=6, color=MUTED)
        ax.set_xlabel('Teacher accuracy (%)')
        ax.set_ylabel('Distilled student accuracy (%)')
        fig.savefig(os.path.join(F, 'teacher_student.pdf'))
        plt.close(fig)

    # Fig. F: confusion matrix of the proposed model (row-normalised, sequential blue)
    if OURS in G:
        from matplotlib.colors import LinearSegmentedColormap
        cm = G[OURS]['confusion']
        tot = [max(1, sum(r)) for r in cm]
        P = [[100 * cm[i][j] / tot[i] for j in range(4)] for i in range(4)]
        fig, ax = plt.subplots(figsize=(2.4, 2.2))
        ax.grid(False)
        im = ax.imshow(P, cmap=LinearSegmentedColormap.from_list('seq', SEQ), vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{P[i][j]:.1f}', ha='center', va='center', fontsize=6.5,
                        color='white' if P[i][j] > 55 else INK)
        ax.set_xticks(range(4), CLASSES, rotation=30)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        fig.savefig(os.path.join(F, 'confusion.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='run roots (default: every runs*/ folder)')
    ap.add_argument('--out', default='paper')
    a = ap.parse_args()
    roots = a.runs if a.runs else sorted(d for d in glob.glob('runs*') if os.path.isdir(d))
    G = aggregate(load_runs(roots))
    make_tables(G, a.out)
    make_macros(G, a.out)
    make_figures(G, a.out)
    print(f'{len(G)} run groups from {roots or "no run folders"} -> {a.out}/tables, {a.out}/figures')
    for g in sorted(G):
        print(f"  {g:<34} n={G[g]['n']}  acc={pm(G[g]['acc'])}")
