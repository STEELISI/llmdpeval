import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sum_noise_by_measure(df, column, bounds, notion, budget):
    """Bounded sum where the noise family follows the privacy notion: then_noise picks Laplace for
    max_divergence (pure epsilon-DP) and Gaussian for zero_concentrated_divergence (rho-zCDP).
    Replace-one (n public): sized domain, d_in = 2, sum charged U - L (add/remove-one, n private:
    unsized domain, d_in = 1).

    notion: "pure" (budget = epsilon) or "zcdp" (budget = rho). Returns (private_sum, budget_spent).
    """
    measure = {"pure": dp.max_divergence(), "zcdp": dp.zero_concentrated_divergence()}[notion]
    values = df[column].astype(float).tolist()
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    pre = space >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    meas = dp.binary_search_chain(
        lambda s: pre >> dp.m.then_noise(output_measure=measure, scale=s), d_in=2, d_out=budget)
    print(meas.output_measure)        # MaxDivergence or ZeroConcentratedDivergence
    return meas(values), meas.map(2)            # replace-one = symmetric distance 2


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, rho_spent = dp_sum_noise_by_measure(df, "value", (-5.0, 5.0), "zcdp", budget=0.5)
    _, epsilon_spent = dp_sum_noise_by_measure(df, "value", (-5.0, 5.0), "pure", budget=1.0)
    return {"epsilon": float(epsilon_spent)}
