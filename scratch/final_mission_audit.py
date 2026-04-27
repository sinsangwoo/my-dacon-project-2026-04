import json

RUN_ID = "run_20260427_112548"
data = json.load(open(f'outputs/{RUN_ID}/processed/signal_validation_logs.json'))
logs = data['val_logs']

inter_logs = [x for x in logs if x['feature'].startswith('inter_')]
ratio_logs = [x for x in logs if x['feature'].startswith('ratio_')]
passed_inter = [x for x in inter_logs if x['passed']]
passed_ratio = [x for x in ratio_logs if x['passed']]

print(f"Total inter_ : {len(inter_logs)} | Passed: {len(passed_inter)}")
print(f"Total ratio_ : {len(ratio_logs)} | Passed: {len(passed_ratio)}")

if passed_inter:
    print("\nTop Approved Interactions:")
    for x in sorted(passed_inter, key=lambda z: z['perm_delta'], reverse=True)[:5]:
         print(f"  {x['feature']:<55} perm={x['perm_delta']:.5f}")

if passed_ratio:
    print("\nTop Approved Ratios:")
    for x in sorted(passed_ratio, key=lambda z: z['perm_delta'], reverse=True)[:5]:
         print(f"  {x['feature']:<55} perm={x['perm_delta']:.5f}")

# Task 5 check: fold stability
print("\nFold Stability Check (Validator Approved):")
for x in sorted([l for l in logs if l['passed']], key=lambda z: z['perm_delta'], reverse=True)[:5]:
    print(f"  {x['feature']:<55} sign_stab={x['sign_stability']:.2f} gain_cv={x['gain_cv']:.4f}")
