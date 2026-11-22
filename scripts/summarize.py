"""Collect runs/*/result.json into a Markdown table: python scripts/summarize.py runs > results.md"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else 'runs'
rows = []
for p in sorted(glob.glob(os.path.join(root, '*', 'result.json'))):
    r = json.load(open(p))['result']
    t = r['test']
    g = r['gflops'] if isinstance(r['gflops'], (int, float)) else float('nan')
    rows.append((r['name'], r['params_total'] / 1e6, r['params_trainable'] / 1e6, g, r['best_val_acc'],
                 t['acc'], t['macro_f1'], t.get('audio_only_acc', float('nan')), t.get('video_only_acc', float('nan'))))
print('| run | params (M) | trainable (M) | GFLOPs | val acc | test acc | test macro-F1 | audio-only | video-only |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print('| {} | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(*r))
