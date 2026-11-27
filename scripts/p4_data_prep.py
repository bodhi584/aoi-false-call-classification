"""P4 data preparation facts: what the inputs are and what was done to them.

Output: results/p4_data_prep.json. Sources: data/dataset.csv, data/mapping.json (inspection features
available per inspection type, published with the data set [4]).
"""
import json
import sys
from pathlib import Path

import pandas as pd

CATEGORICAL = ["meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4", "inspection_type"]


def main(path="data/dataset.csv", mapping="data/mapping.json"):
    df = pd.read_csv(path, index_col=0)
    num = [c for c in df.columns if c.startswith("inspection_feat")]
    n = len(df) // 2
    avail = json.loads(Path(mapping).read_text())
    by_type = df.groupby("inspection_type")["class"].agg(["size", "sum"])
    res = {
        "rows": len(df), "columns": df.shape[1], "measurements": len(num), "categorical": len(CATEGORICAL),
        "missing_values": int(df.isna().sum().sum()),
        "constant_measurements": [c for c in num if df[c].nunique() == 1],
        "categories_per_feature": {c: int(df[c].nunique()) for c in CATEGORICAL},
        "inputs_after_one_hot": len(num) + int(sum(df[c].nunique() for c in CATEGORICAL)),
        "meta_feat1_codes_only_in_test_half": int(len(set(df.meta_feat1.iloc[n:]) - set(df.meta_feat1.iloc[:n]))),
        "duplicate_rows": int(df.duplicated().sum()),
        "development_rows": n, "development_true_defects": int(df["class"].iloc[:n].sum()),
        "per_inspection_type": {int(t): {"rows": int(r["size"]), "true_defects": int(r["sum"]),
                                         "true_defect_pct": round(100 * r["sum"] / r["size"], 2),
                                         "features_available": len(avail.get(str(t), []))}
                                for t, r in by_type.iterrows()},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/p4_data_prep.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
