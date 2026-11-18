import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def label_from_one_hot(df, one_hot_columns, labels, none_label):
    """Per-row label from one-hot 0/1 columns (a per-record step, not a release); no 1 -> none_label."""
    v = df[list(one_hot_columns)].to_numpy(dtype=float, copy=True)
    out = np.array(labels, dtype=object)[v.argmax(axis=1)]
    out[v.max(axis=1) < 1] = none_label
    return out


def dp_confusion_matrices(true_labels, predictions, true_values, pred_values, epsilon):
    """(true label x predicted label) count matrix for each of several classifiers on the same rows.
    true_values / pred_values: PUBLIC label lists, including 'none' / 'no class' / 'unclassified' entries.
    Replace-one row (bounded DP, n public). Returns ({name: noisy matrix}, epsilon_spent)."""
    cells = [f"{a}|{b}" for a in true_values for b in pred_values]     # public cell list, row-major
    eps_each = epsilon / len(predictions)                    # sequential composition over classifiers
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    def make(s):
        return (space >> dp.t.then_count_by_categories(categories=cells, null_category=False)
                >> dp.m.then_laplace(s))
    # One row is one token in ONE cell; replace-one = remove one token, add one: d_in = 2.
    meas = make(dp.binary_search_param(make, d_in=2, d_out=eps_each))
    out, spent = {}, 0.0
    for name, pred in predictions.items():
        tokens = [f"{a}|{b}" for a, b in zip(true_labels, pred)]   # tokens outside `cells` are not counted
        out[name] = np.array(meas(tokens), dtype=float).reshape(len(true_values), len(pred_values))
        spent += meas.map(2)
    return out, spent


def error_rates(noisy, true_values, pred_values, n_total, classes):
    """Post-processing: overall error (n public) and, per class, false-negative and false-positive rates
    with NOISY denominators (group sizes are private). Clipped to [0, 1]."""
    m = np.maximum(noisy, 0.0)
    row = m.sum(axis=1)
    hit = {c: m[true_values.index(c), pred_values.index(c)] for c in classes}
    rates = {"overall_error": float(np.clip(1 - sum(hit.values()) / n_total, 0, 1))}
    for c in classes:
        i, j = true_values.index(c), pred_values.index(c)
        rates[f"{c}_fn"] = float(np.clip(1 - hit[c] / max(row[i], 1.0), 0, 1))
        rates[f"{c}_fp"] = float(np.clip((m[:, j].sum() - hit[c]) / max(row.sum() - row[i], 1.0), 0, 1))
    return rates


def _test():
    import pandas as pd
    n, labels = 5000, ["u", "v", "w"]
    g = np.random.choice(4, n, p=[0.5, 0.3, 0.15, 0.05])     # 3 = no label
    df = pd.DataFrame({f"is_{c}": (g == k).astype(np.int8) for k, c in enumerate(labels)})
    truth = label_from_one_hot(df, [f"is_{c}" for c in labels], labels, "none")
    noisy_pred = np.where(np.random.rand(n) < 0.8, truth, np.random.choice(labels + ["no_class"], n))
    preds = {"model1": noisy_pred, "model2": np.random.choice(labels, n)}
    mats, eps = dp_confusion_matrices(truth, preds, labels + ["none"], labels + ["no_class"], epsilon=1.0)
    r = error_rates(mats["model1"], labels + ["none"], labels + ["no_class"], n, labels)
    assert 0.1 < r["overall_error"] < 0.4 and set(mats) == {"model1", "model2"}
    return {"epsilon": float(eps)}
