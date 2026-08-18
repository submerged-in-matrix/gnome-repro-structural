import pandas as pd, numpy as np
from matbench_discovery.metrics import stable_metrics

summary = pd.read_csv("data/raw/wbm-summary.csv.gz").set_index("material_id")
df_pred = pd.read_csv("models/ema-gnn/ema-gnn/2026-08-03-discovery.csv.gz").set_index("material_id")

common = summary.index.intersection(df_pred.index)
ref = summary.loc[common]
e_form_pred = df_pred.loc[common, "e_form_pred"].values
e_form_dft = ref["e_form_per_atom_mp2020_corrected"].values
each_true = ref["e_above_hull_mp2020_corrected_ppd_mp"].values

error = np.abs(e_form_pred - e_form_dft)
each_pred = each_true + e_form_pred - e_form_dft
each_pred[error > 5.0] = np.nan

full = stable_metrics(each_true, each_pred)
print("full_test_set:")
for k in ("F1","DAF","Precision","Recall","MAE","RMSE","R2"):
    print(f"  {k}: {full[k]:.4f}")

uniq = ref["unique_prototype"].astype(bool).values
u = stable_metrics(each_true[uniq], each_pred[uniq])
print("unique_prototypes:")
for k in ("F1","DAF","Precision","Recall","MAE","RMSE","R2"):
    print(f"  {k}: {u[k]:.4f}")

order = np.argsort(each_pred[uniq], kind="stable")[:10000]
s = stable_metrics(each_true[uniq][order], each_pred[uniq][order])
s["DAF"] = s["Precision"] / float((each_true[uniq] <= 0).mean())
print("most_stable_10k:")
for k in ("F1","DAF","Precision","Recall","MAE","RMSE","R2"):
    print(f"  {k}: {s[k]:.4f}")
