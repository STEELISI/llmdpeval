import warnings

import numpy as np
from diffprivlib import mechanisms, models
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing data_norm must fail, never silently read the data


def dp_select_C(df, feature_columns, label_column, data_norm, C_values, validation_size, epsilon, select_share=0.2):
    """Choose a logistic-regression C from a list fixed in advance, privately, and release only that model.
    Replace-one (n public): objective perturbation and the validation-accuracy utility need no halving.

    K fits on the same training rows cost K * eps_fit; the private choice on validation accuracy costs eps_select.
    data_norm: public L2-norm bound on a (publicly scaled) feature row. Rows are divided by sqrt(2)*data_norm
    and get an intercept column 1/sqrt(2), so the model sees norm <= 1 with fit_intercept=False (diffprivlib
    0.6.6 under-charges epsilon for norms above 1). Returns ((C, model), epsilon_spent).
    """
    eps_select = select_share * epsilon
    eps_fit = (epsilon - eps_select) / len(C_values)
    # Public rule: cap C so the slack 2*ln(1 + C/4) uses at most eps_fit/4 (else the library silently falls back).
    C_values = [min(C, 4 * np.expm1(eps_fit / 8)) for C in C_values]
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    X = df[list(feature_columns)].to_numpy(dtype=float) / (np.sqrt(2) * data_norm)
    X = np.column_stack([X, np.full(len(X), 1 / np.sqrt(2))])
    y = df[label_column].to_numpy()
    X_tr, y_tr, X_va, y_va = X[:-validation_size], y[:-validation_size], X[-validation_size:], y[-validation_size:]
    fitted = [models.LogisticRegression(epsilon=eps_fit, data_norm=1.0, C=C, fit_intercept=False,
                                        accountant=acc).fit(X_tr, y_tr)
              for C in C_values]
    eps_select = min(eps_select, acc.remaining()[0])  # float rounding: the selection takes what is left
    scores = [float(m.score(X_va, y_va)) for m in fitted]  # one validation row moves each accuracy by <= 1/M
    mech = mechanisms.Exponential(epsilon=eps_select, sensitivity=1 / validation_size, utility=scores)
    k = mech.randomise()  # an index, since no candidates were given
    acc.spend(eps_select, 0)  # mechanisms do not charge an accountant themselves
    # Refitting model k on all rows would be another eps_fit and must also be charged.
    return (C_values[k], fitted[k]), acc.total()[0]


def _test():
    import pandas as pd
    a = np.linspace(0, 1, 400)
    df = pd.DataFrame({"a": a, "b": 1 - a, "label": (a > 0.5).astype(int)}).sample(frac=1).reset_index(drop=True)
    (C, model), epsilon_spent = dp_select_C(df, ["a", "b"], "label", data_norm=np.sqrt(2), C_values=[0.1, 1.0, 10.0],
                                            validation_size=100, epsilon=1.0)
    assert C <= 4 * np.expm1(0.8 / 3 / 8) + 1e-12
    return {"epsilon": epsilon_spent}
