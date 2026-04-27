import json, sys

RUN_ID = "run_20260427_105332"
data = json.load(open(f'outputs/{RUN_ID}/processed/signal_validation_logs.json'))
logs = data['val_logs']

inter_logs = [x for x in logs if x['feature'].startswith('inter_')]
passed = [x for x in inter_logs if x['passed']]
soft = [x for x in inter_logs if not x['passed'] and x['perm_delta'] > 0.002]

print(f"Total inter_ candidates evaluated : {len(inter_logs)}")
print(f"  Gate-approved (passed=True)      : {len(passed)}")
print(f"  Soft-recoverable (perm_delta>0.002): {len(soft)}")
print()
print("Top-8 by perm_delta:")
for x in sorted(inter_logs, key=lambda z: z['perm_delta'], reverse=True)[:8]:
    print(f"  {x['feature']:<55} perm={x['perm_delta']:.5f}  passed={x['passed']}")

# Coverage stats
all_passed = [x for x in logs if x['passed']]
print(f"\nOverall: {len(all_passed)}/{len(logs)} features gate-approved")
print(f"  noise_survival_rate : {data['noise_proof']['noise_survival_rate']:.2%}")
print(f"  capacity_corr       : {data['capacity_corr']:.4f}")
print(f"  var_stats           : {data['var_stats']}")
