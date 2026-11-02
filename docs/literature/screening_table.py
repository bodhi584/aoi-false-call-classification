"""Write the classifier screening table of the literature report from results/classifier_screening.json.

Usage: python docs/literature/screening_table.py   (writes docs/literature/screening_table.tex)
"""
import json
from pathlib import Path

NAMES = {
    "logistic_regression": "Logistic regression", "linear_discriminant_analysis": "Linear discriminant analysis",
    "linear_svm": "Linear SVM", "gaussian_naive_bayes": "Gaussian naive Bayes",
    "k_nearest_neighbours": "k-nearest neighbours (k=15)", "rbf_svm_subsample": "SVM, RBF kernel (20,000-row subsample)",
    "decision_tree": "Decision tree", "random_forest": "Random forest", "extra_trees": "Extremely randomised trees",
    "hist_gradient_boosting": "Gradient boosting (histogram)", "adaboost": "AdaBoost",
    "mlp_1_hidden_layer": "Neural network, 1 hidden layer (10)", "mlp_2_hidden_layers": "Neural network, 2 hidden layers (64, 32)",
}
FAMILY = {"linear": "Linear", "probabilistic": "Probabilistic", "instance": "Instance-based", "kernel": "Kernel",
          "tree": "Tree", "tree_ensemble": "Tree ensemble", "boosting": "Boosting", "neural": "Neural network"}

r = json.loads(Path("results/classifier_screening.json").read_text())
rows = []
for k, v in r.items():
    rows.append((FAMILY[v["family"]], NAMES[k], v["specificity_pct_at_99pct_recall"], v["specificity_pct_at_100pct_recall"],
                 v["joint_auroc"], v["fit_seconds"]))
best = max(x[2] for x in rows)
out = []
for fam, name, s99, s100, auc, sec in rows:
    s = f"\\textbf{{{s99:.1f}}}" if s99 == best else f"{s99:.1f}"
    t = f"{sec:.0f} s" if sec < 100 else f"{sec/60:.0f} min"
    out.append(f"{fam} & {name} & {s} & {s100:.1f} & {auc:.3f} & {t} \\\\")
Path("docs/literature/screening_table.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
