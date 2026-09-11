# ============================================================
# COMPUTE DRUG PROXIMITY Z-SCORES
# ============================================================

import ast
import numpy as np
import pandas as pd
import os

# ============================================================
# INPUT
# ============================================================

INPUT_FILE = (
    "/content/drive/MyDrive/INDRA/improved_random_2/"
    "COVID_2020/Statistics/drug_level_statistics_2020.csv"
)

BASELINE_FILE = (
    "/content/drive/MyDrive/INDRA/improved_random_2/"
    "baseline_mu_sigma.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/INDRA/improved_random_2/"
    "COVID_2020/Tables/drug_z_scores_2020.csv"
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

drug_df = pd.read_csv(INPUT_FILE)

baseline_df = pd.read_csv(BASELINE_FILE)

baseline = {

    int(row["k"]): (

        row["mu_k"],

        row["sigma_k"]

    )

    for _, row in baseline_df.iterrows()

}

# ============================================================
# COMPUTE Z-SCORES
# ============================================================

rows = []

for _, row in drug_df.iterrows():

    k = int(row["k"])

    if k not in baseline:

        continue

    target_distances = ast.literal_eval(
        row["target_distances"]
    )

    distances = list(
        target_distances.values()
    )

    d_drug = np.mean(distances)

    mu_k, sigma_k = baseline[k]

    if sigma_k == 0:

        continue

    z_score = (

        d_drug - mu_k

    ) / sigma_k

    rows.append({

        "drug": row["drug"],

        "k": k,

        "d_drug": d_drug,

        "mu_k": mu_k,

        "sigma_k": sigma_k,

        "z_score": z_score,

        "path_count": row["path_count"],

        "shortest_path": row["shortest_path"],

        "avg_belief": row["avg_belief"],

        "avg_corr_weight": row["avg_corr_weight"]

    })

# ============================================================
# SAVE
# ============================================================

z_df = pd.DataFrame(rows)

z_df = z_df.sort_values(

    "z_score",

    ascending=True

)

z_df.to_csv(

    OUTPUT_FILE,

    index=False

)

# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)

print("DRUG PROXIMITY Z-SCORES")

print("=" * 60)

print()

print("Drugs analysed:", len(z_df))

print()

print("Saved:")

print(OUTPUT_FILE)

print()

display(

    z_df.head(20)
