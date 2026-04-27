import json
import pandas as pd

RUN1 = "run_20260427_121941"
RUN2 = "run_20260427_122102"

logs1 = json.load(open(f'outputs/{RUN1}/processed/signal_validation_logs.json'))['val_logs']
logs2 = json.load(open(f'outputs/{RUN2}/processed/signal_validation_logs.json'))['val_logs']

df1 = pd.DataFrame(logs1)
df2 = pd.DataFrame(logs2)

passed1 = set(df1[df1['passed']]['feature'])
passed2 = set(df2[df2['passed']]['feature'])

common = passed1 & passed2
only1 = passed1 - passed2
only2 = passed2 - passed1

print(f"Run 1 Approved: {len(passed1)}")
print(f"Run 2 Approved: {len(passed2)}")
print(f"Stable (Both):  {len(common)}  ({len(common)/max(len(passed1),len(passed2))*100:.0f}%)")
print(f"Only Run 1:     {len(only1)}")
print(f"Only Run 2:     {len(only2)}")

if only1:
    print("\nUnstable (disappeared in Run 2):")
    for f in sorted(only1):
        row = df1[df1['feature'] == f].iloc[0]
        print(f"  {f:<55} perm={row['perm_delta']:.5f}")

if only2:
    print("\nUnstable (appeared in Run 2):")
    for f in sorted(only2):
        row = df2[df2['feature'] == f].iloc[0]
        print(f"  {f:<55} perm={row['perm_delta']:.5f}")
