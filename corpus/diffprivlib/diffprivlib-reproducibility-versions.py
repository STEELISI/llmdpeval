import warnings

import diffprivlib
import scipy
import sklearn
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

# Reproducible ENVIRONMENT, fresh noise: pin versions and report them; never fix the noise of a real release.
warnings.simplefilter("error", PrivacyLeakWarning)  # silent leaks (e.g. bounds read from the data) fail loudly


def dp_mean_with_versions(df, column, bounds, epsilon):
    """DP mean of df[column] (public bounds=(lower, upper)) plus the library versions a reader needs to rerun it.

    diffprivlib 0.6.6 needs scikit-learn < 1.8 and scipy < 1.18. Returns ((private_mean, versions), epsilon_spent).
    """
    versions = {"diffprivlib": diffprivlib.__version__, "scikit-learn": sklearn.__version__,
                "scipy": scipy.__version__}
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    private_mean = tools.mean(df[column].to_numpy(dtype=float), epsilon=epsilon, bounds=bounds, accountant=acc)
    return (float(private_mean), versions), acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(100) % 40})
    (m, versions), epsilon_spent = dp_mean_with_versions(df, "value", (0, 50), epsilon=1.0)
    assert versions["diffprivlib"] == "0.6.6"
    return {"epsilon": epsilon_spent}
