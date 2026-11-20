from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

DATA_PATH = Path("data/global_ai_semiconductor_supply_chain_enriched_2026.csv")
FIGURES_PATH = Path("outputs/figures")
IMG_DPI = 300

X_COL = "defect_density_per_cm2"
Y_COL = "die_yield_pct"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}.")
    return pd.read_csv(path)


def task_3a(df: pd.DataFrame) -> pd.DataFrame:
    """Select two relevant quantitative variables that plausibly have a linear relationship"""
    print(
        "\n# ── Task 3a: Variables selection ────────────────────────────────────────────── "
    )
    data = df[[X_COL, Y_COL]].dropna().copy()

    print(f"Predictor (X): {X_COL}")
    print(f"Response  (Y): {Y_COL}")
    print(f"Valid observations (n): {len(data)}\n")
    print(data.describe().round(4))

    return data


def task_3b(data: pd.DataFrame) -> float:
    """Compute and interpret the Pearson correlation coefficient"""
    print(
        "\n# ── Task 3b: Pearson correlation ────────────────────────────────────────────── "
    )

    x = data[X_COL]
    y = data[Y_COL]

    r_raw, p_raw = stats.pearsonr(x, y)
    r = float(r_raw)
    p_val = float(p_raw)

    print(f"Pearson r = {r:.4f}")
    print(f"p-value   = {p_val:.4e}")

    return r


def task_3cd(data: pd.DataFrame) -> RegressionResultsWrapper:
    """Build a simple linear regression model (Y = b0 + b1*X) using OLS"""
    print(
        "\n# ── Task 3c & 3d: Simple Linear Regression (OLS) ──────────────────────────────"
    )

    x = data[X_COL]
    y = data[Y_COL]

    x_design = sm.add_constant(x)
    model = sm.OLS(y, x_design).fit()

    b0 = float(model.params["const"])
    b1 = float(model.params[X_COL])

    print(f"Fitted equation: {Y_COL} = {b0:.4f} + ({b1:.4f}) * {X_COL}\n")
    print(model.summary())
    print(f"R-squared     = {float(model.rsquared):.4f}")
    print(f"Adj R-squared = {float(model.rsquared_adj):.4f}")

    return model


def task_3e(model: RegressionResultsWrapper) -> None:
    """Construct 95% confidence intervals for the true intercept (beta0) and slope (beta1)"""
    print(
        "\n# ── Task 3e: Interval estimation for coefficients ───────────────────────────── "
    )

    ci = model.conf_int(alpha=0.05)
    s_e = float(model.mse_resid**0.5)  # MSE (resid (e) = y - y_hat)
    df_resid = int(model.df_resid)  # n - 2

    for label, key in [
        ("Intercept (beta0)", "const"),
        ("Slope     (beta1)", X_COL),
    ]:
        est = float(model.params[key])
        se = float(model.bse[key])
        low = float(ci.loc[key, 0])
        high = float(ci.loc[key, 1])
        print(
            f"{label}: est = {est:9.4f} | SE = {se:.4f} | 95% CI = [{low:.4f}, {high:.4f}]"
        )

    print(f"\nResidual Std Error = {s_e:.4f} on df = {df_resid}")


def task_3f(
    model: RegressionResultsWrapper, x0: float = 0.15, alpha_c: float = 0.05
) -> None:
    """Compute point prediction, 95% CI for mean response, and 95% PI for a new observation"""
    print(
        "\n# ── Task 3f: Prediction of new Observations ─────────────────────────────────── "
    )

    predict = model.get_prediction([1.0, x0])
    frame = predict.summary_frame(alpha=alpha_c)
    row = frame.iloc[0]

    y_hat = float(row["mean"])
    ci_low, ci_high = float(row["mean_ci_lower"]), float(row["mean_ci_upper"])
    pi_low, pi_high = float(row["obs_ci_lower"]), float(row["obs_ci_upper"])

    print(f"Chosen x0 ({X_COL}) = {x0}")
    print(f"Point prediction (y_hat) = {y_hat:.4f}%")
    print(
        f"95% CI (Mean Response)   = [{ci_low:.4f}, {ci_high:.4f}] (width = {ci_high - ci_low:.4f})"
    )
    print(
        f"95% PI (New Observation) = [{pi_low:.4f}, {pi_high:.4f}] (width = {pi_high - pi_low:.4f})"
    )


def task_3g(data: pd.DataFrame, model: RegressionResultsWrapper) -> None:
    """Perform Residual diagnostics with scatterplot + diagnostic plots"""
    print(
        "\n# ── Task 3g: Residual Diagnostics ───────────────────────────────────────────── "
    )

    x = data[X_COL]
    y = data[Y_COL]
    y_hat = model.fittedvalues
    resid = model.resid

    print(f"Residual mean = {float(resid.mean()):.4e}")
    print(f"Residual std  = {float(resid.std()):.4f}")
    print(
        f"Residual min  = {float(resid.min()):.4f} | max = {float(resid.max()):.4f}"
    )

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # in x in

    # Scatterplot with fitted regression line (3d)
    axes[0].scatter(x, y, alpha=0.15, s=8, color="steelblue", label="Observed")
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    b0, b1 = float(model.params["const"]), float(model.params[X_COL])
    axes[0].plot(
        x_line, b0 + b1 * x_line, color="crimson", linewidth=2, label="OLS Fit"
    )
    axes[0].set_title("Die Yield vs. Defect Density")
    axes[0].set_xlabel(X_COL)
    axes[0].set_ylabel(Y_COL)
    axes[0].legend()

    # Residual vs. Fitted Values
    axes[1].scatter(y_hat, resid, alpha=0.15, s=8, color="steelblue")
    axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Fitted Values")
    axes[1].set_xlabel("Fitted Die Yield (%)")
    axes[1].set_ylabel("Residuals")

    # Q-Q Plot
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q Plot of Residuals")

    fig.tight_layout()
    out_path = FIGURES_PATH / "task_3g_graphs.png"
    fig.savefig(out_path, dpi=IMG_DPI)
    plt.close(fig)


def task_3hi(
    data: pd.DataFrame, model: RegressionResultsWrapper, r: float
) -> None:
    """Compute R^2 = r^2 and apply suitable transformation"""
    print(
        "\n# ── Task 3h: Coefficient of Determination (R^2) ─────────────────────────────── "
    )

    r2 = float(model.rsquared)
    print(
        f"Model R^2        = {r2:.6f} ({r2 * 100:.2f}% of variance explained)"
    )
    print(f"Pearson r^2      = {r**2:.6f}")
    print(f"Difference       = {abs(r2 - r**2):.2e}")

    print(
        "\n# ── Task 3i: Transformation check ───────────────────────────────────────────── "
    )
    x = data[X_COL]
    y = data[Y_COL]
    print(f"Original   (Y ~ X)         : R^2 = {r2:.4f}")

    # Candidate 1: Log-Linear  ln(Y) ~ X
    m_log_y = sm.OLS(np.log(y), sm.add_constant(x)).fit()
    print(f"Log-Linear (ln(Y) ~ X)     : R^2 = {float(m_log_y.rsquared):.4f}")

    # Candidate 2: Linear-Log  Y ~ ln(X)
    m_log_x = sm.OLS(y, sm.add_constant(np.log(x))).fit()
    print(f"Linear-Log (Y ~ ln(X))     : R^2 = {float(m_log_x.rsquared):.4f}")

    # Candidate 3: Log-Log     ln(Y) ~ ln(X)
    m_log_log = sm.OLS(np.log(y), sm.add_constant(np.log(x))).fit()
    print(f"Log-Log    (ln(Y) ~ ln(X)) : R^2 = {float(m_log_log.rsquared):.4f}")

    # Comparing OG linear vs Candidate 1
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    specs = [
        (
            axes[0, 0],
            model,
            "Original Linear ($Y \\sim X$)",
            "Fitted Values ($\\hat{y}$)",
            "steelblue",
        ),
        (
            axes[0, 1],
            m_log_y,
            "Log-Linear ($\\ln(Y) \\sim X$)",
            "Fitted Values ($\\widehat{\\ln y}$)",
            "darkorange",
        ),
        (
            axes[1, 0],
            m_log_x,
            "Linear-Log ($Y \\sim \\ln(X)$)",
            "Fitted Values ($\\hat{y}$)",
            "seagreen",
        ),
        (
            axes[1, 1],
            m_log_log,
            "Log-Log ($\\ln(Y) \\sim \\ln(X)$)",
            "Fitted Values ($\\widehat{\\ln y}$)",
            "mediumpurple",
        ),
    ]

    for ax, m, title, xlabel, color in specs:
        ax.scatter(m.fittedvalues, m.resid, alpha=0.15, s=8, color=color)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
        ax.set_title(f"{title}\n$R^2 = {float(m.rsquared):.4f}$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residuals ($e$)")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(
        FIGURES_PATH / "task_3i_comparison.png",
        dpi=IMG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def run() -> None:
    df = load_data(DATA_PATH)

    data = task_3a(df)
    r = task_3b(data)
    model = task_3cd(data)
    task_3e(model)
    task_3f(model, 0.15)
    task_3g(data, model)
    task_3hi(data, model, r)


if __name__ == "__main__":
    run()
