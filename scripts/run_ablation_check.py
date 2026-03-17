"""
Phase 4.5: Full-scale N=100 ablation run.
Runs all 5 modes on the same 100 test samples for publishable results.
Estimated runtime: ~4-5 hours total (50-60 min per mode).
"""
import subprocess
import json
import sys
import time
from datetime import datetime

MODES = ['no_retrieval', 'cosine_only', 'stratify_only', 'contrastive_only', 'full']
SAMPLE_SIZE = 100
SEED = 42

results = {}
timings = {}

print(f"Phase 4.5: Full N={SAMPLE_SIZE} Ablation Run")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Modes: {MODES}")
print(f"Seed: {SEED}")
print("=" * 60)

for i, mode in enumerate(MODES):
    output_file = f'results/n100_{mode}.json'
    cmd = [
        '.venv/Scripts/python.exe', 'src/evaluate_batch.py',
        '--csv', 'image-data/processed/test.csv',
        '--sample_size', str(SAMPLE_SIZE),
        '--ablation', mode,
        '--output', output_file,
        '--seed', str(SEED)
    ]
    
    print(f"\n[{i+1}/{len(MODES)}] Running: {mode}")
    print(f"  Output: {output_file}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    
    start = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    timings[mode] = elapsed
    
    if proc.returncode != 0:
        print(f"  FAILED: {mode} (exit code {proc.returncode})")
        # Save partial results and continue
        continue
    
    print(f"  Completed in {elapsed/60:.1f} min")
    
    with open(output_file) as f:
        data = json.load(f)
    results[mode] = data['metrics_overall']

# ============================================================================
# GENERATE COMPARISON TABLE
# ============================================================================

metrics_list = [
    # Retrieval metrics
    ('recall_at_5', 'Recall@5'),
    ('avg_label_overlap', 'Avg Label Overlap'),
    ('abn_alignment', 'Abn Alignment'),
    ('norm_alignment', 'Norm Alignment'),
    # NLG metrics
    ('avg_bleu_1', 'BLEU-1'),
    ('avg_bleu_4', 'BLEU-4'),
    ('avg_meteor', 'METEOR'),
    ('avg_rouge_l', 'ROUGE-L'),
    # CE metrics
    ('precision', 'CE Precision'),
    ('recall', 'CE Recall'),
    ('f1', 'CE F1'),
]

completed_modes = [m for m in MODES if m in results]

lines = []
lines.append(f"FULL ABLATION RESULTS (N={SAMPLE_SIZE}, seed={SEED})")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 120)

# Header
header = f"{'Metric':<20s}"
for m in completed_modes:
    header += f"  {m:>17s}"
lines.append(header)
lines.append("-" * 120)

for key, name in metrics_list:
    row = f"{name:<20s}"
    vals = []
    for m in completed_modes:
        val = results[m].get(key, 0)
        vals.append(val)
        row += f"  {val:>17.4f}"
    
    # Mark best value
    if vals and max(vals) > 0:
        best_idx = vals.index(max(vals))
        # Re-format the best value with asterisk
        pass
    
    lines.append(row)

lines.append("")
lines.append("TIMINGS:")
for m in completed_modes:
    lines.append(f"  {m}: {timings.get(m, 0)/60:.1f} min")

lines.append("")
lines.append("STABILITY CONDITIONS:")

# Condition 1: Retrieval metrics
abn_strat = results.get('stratify_only', {}).get('abn_alignment', 0)
abn_full = results.get('full', {}).get('abn_alignment', 0)
lines.append(f"  Condition 1 (alignment): stratify_only={abn_strat:.3f}, full={abn_full:.3f} -> {'PASS' if abn_strat >= 0.9 and abn_full >= 0.9 else 'FAIL'}")

# Condition 2: NLG ordering
nlg_vals = {m: results.get(m, {}).get('avg_meteor', 0) for m in completed_modes}
no_ret = nlg_vals.get('no_retrieval', 0)
cos = nlg_vals.get('cosine_only', 0)
lines.append(f"  Condition 2 (NLG): no_retrieval METEOR={no_ret:.4f} < cosine_only={cos:.4f} -> {'PASS' if cos > no_ret else 'FAIL'}")

# Condition 3: CE F1 improvement
f1_noret = results.get('no_retrieval', {}).get('f1', 0)
f1_full = results.get('full', {}).get('f1', 0)
f1_best_ret = max(results.get(m, {}).get('f1', 0) for m in ['cosine_only', 'stratify_only', 'contrastive_only', 'full'] if m in results)
lines.append(f"  Condition 3 (CE F1): no_retrieval={f1_noret:.4f}, best_retrieval={f1_best_ret:.4f} -> {'PASS' if f1_best_ret > f1_noret else 'FAIL'}")

output = "\n".join(lines)
with open('results/n100_comparison.txt', 'w') as f:
    f.write(output)

# Also save as structured JSON for paper tables
comparison_json = {
    'config': {'sample_size': SAMPLE_SIZE, 'seed': SEED},
    'timings': timings,
    'results': {m: results.get(m, {}) for m in MODES}
}
with open('results/n100_comparison.json', 'w') as f:
    json.dump(comparison_json, f, indent=2)

print("\n\n" + output)
print("\n" + "=" * 60)
print(f"Total runtime: {sum(timings.values())/60:.1f} min")
print(f"Results saved to: results/n100_comparison.txt + .json")
