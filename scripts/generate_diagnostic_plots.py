"""
Diagnostic plot generator for the FinBERT × SARIMAX thesis notebook.

Generates publication-grade plots for embedding in the Jupyter notebook:
    1. Stationarity tests heatmap (ADF vs KPSS)
    2. Long-run SARIMAX coefficient forest plot
    3. Sentiment distribution boxplot
    4. Pre/post-COVID model comparison (already covered by generate_hero_plot.py)
    5. Residual diagnostics panel (mock based on thesis Figures 15-16)

Author: Jean Treves
License: MIT
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Publication palette (matches generate_hero_plot.py)
COLOR_PRIMARY: str = "#2E86AB"
COLOR_ACCENT: str = "#E63946"
COLOR_NEUTRAL: str = "#6C757D"
COLOR_TEXT: str = "#1A1A2E"
COLOR_GRID: str = "#E8E8E8"
COLOR_BACKGROUND: str = "#FAFAFA"
COLOR_POSITIVE: str = "#28A745"
COLOR_NEGATIVE: str = "#DC3545"


def _setup_axes(ax: plt.Axes) -> None:
    """Apply standard styling to a matplotlib axes."""
    ax.set_facecolor(COLOR_BACKGROUND)
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.7, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine_loc in ("top", "right"):
        ax.spines[spine_loc].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT, labelsize=10)


def plot_stationarity_heatmap(
    stationarity_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """
    Plot ADF vs KPSS p-values as a heatmap-style comparison.

    ADF H0: unit root (non-stationary). Reject if p < 0.05.
    KPSS H0: stationary. Reject if p < 0.05.
    Both failing to reject → trend-stationary (need first differencing).

    Parameters
    ----------
    stationarity_df : pd.DataFrame
        Must contain columns: ticker, adf_pvalue, kpss_pvalue.
    output_path : Path
        Destination PNG path.
    dpi : int, default=200

    Returns
    -------
    Path
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi, facecolor=COLOR_BACKGROUND)
    _setup_axes(ax)

    tickers: list[str] = stationarity_df["ticker"].tolist()
    x: np.ndarray = np.arange(len(tickers))
    width: float = 0.38

    bars_adf = ax.bar(
        x - width / 2,
        stationarity_df["adf_pvalue"],
        width,
        label="ADF p-value (H₀: unit root)",
        color=COLOR_PRIMARY,
        alpha=0.85,
        edgecolor=COLOR_TEXT,
        linewidth=1.2,
    )
    bars_kpss = ax.bar(
        x + width / 2,
        stationarity_df["kpss_pvalue"],
        width,
        label="KPSS p-value (H₀: stationary)",
        color=COLOR_ACCENT,
        alpha=0.85,
        edgecolor=COLOR_TEXT,
        linewidth=1.2,
    )

    # Significance threshold
    ax.axhline(
        y=0.05,
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="α = 0.05",
    )

    # Annotate bars
    for bars in (bars_adf, bars_kpss):
        for bar in bars:
            height: float = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOR_TEXT,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("p-value", fontsize=12, color=COLOR_TEXT)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    fig.suptitle(
        "Stationarity Tests: ADF vs KPSS",
        fontsize=15,
        fontweight="bold",
        color=COLOR_TEXT,
        x=0.125,
        y=0.97,
        ha="left",
    )
    fig.text(
        0.125,
        0.92,
        "All 5 tickers: ADF fails to reject unit root + KPSS rejects stationarity → first differencing required",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.65,
        style="italic",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.84)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_BACKGROUND)
    plt.close(fig)
    logger.info("Saved stationarity heatmap to %s", output_path)
    return output_path


def plot_long_run_forest(
    long_run_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """
    Forest plot of long-run SARIMAX SEPS coefficients with confidence intervals.

    Visualizes the central finding that only PLUG has statistically significant
    SEPS impact on stock price (p=0.008), consistent with weak EMH for the rest.

    Parameters
    ----------
    long_run_df : pd.DataFrame
        Must contain: ticker, seps_coef, seps_std_err, seps_pvalue.
    output_path : Path
    dpi : int, default=200

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(11, 5), dpi=dpi, facecolor=COLOR_BACKGROUND)
    _setup_axes(ax)

    df: pd.DataFrame = long_run_df.copy().reset_index(drop=True)
    y_pos: np.ndarray = np.arange(len(df))

    # 95% CI (approx via 1.96 × SE)
    df["ci_lower"] = df["seps_coef"] - 1.96 * df["seps_std_err"]
    df["ci_upper"] = df["seps_coef"] + 1.96 * df["seps_std_err"]
    df["color"] = df["seps_pvalue"].apply(
        lambda p: COLOR_ACCENT if p < 0.05 else COLOR_NEUTRAL
    )

    # Error bars + points
    for i, row in df.iterrows():
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [i, i],
            color=row["color"],
            linewidth=2,
            alpha=0.7,
        )
        ax.scatter(
            row["seps_coef"],
            i,
            s=120,
            color=row["color"],
            edgecolor=COLOR_TEXT,
            linewidth=1.5,
            zorder=3,
        )
        # Annotate p-value
        p_str: str = "p < 0.01" if row["seps_pvalue"] < 0.01 else f"p = {row['seps_pvalue']:.3f}"
        marker: str = "★" if row["seps_pvalue"] < 0.05 else ""
        ax.text(
            row["ci_upper"] + 3,
            i,
            f"{row['seps_coef']:+.2f}  ({p_str}) {marker}",
            va="center",
            fontsize=10,
            color=COLOR_TEXT,
        )

    # Zero line
    ax.axvline(x=0, color=COLOR_TEXT, linestyle="-", linewidth=1, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["ticker"].tolist(), fontsize=11)
    ax.set_xlabel("SARIMAX(0,1,0) Surprise EPS Coefficient", fontsize=12, color=COLOR_TEXT)
    ax.invert_yaxis()

    fig.suptitle(
        "Long-Run SARIMAX: SEPS Effect on Stock Price",
        fontsize=15,
        fontweight="bold",
        color=COLOR_TEXT,
        x=0.125,
        y=0.97,
        ha="left",
    )
    fig.text(
        0.125,
        0.92,
        "Only PLUG shows statistically significant SEPS impact (★) — others follow random walk (weak-form EMH)",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.65,
        style="italic",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.84, right=0.78)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_BACKGROUND)
    plt.close(fig)
    logger.info("Saved long-run forest plot to %s", output_path)
    return output_path


def plot_sentiment_distribution(
    sentiment_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """
    Boxplot of sentiment scores per ticker.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Must contain: ticker, sentiment_score.
    output_path : Path
    dpi : int, default=200

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi, facecolor=COLOR_BACKGROUND)
    _setup_axes(ax)

    tickers: list[str] = sorted(sentiment_df["ticker"].unique().tolist())
    data: list[np.ndarray] = [
        sentiment_df[sentiment_df["ticker"] == t]["sentiment_score"].values
        for t in tickers
    ]

    bp = ax.boxplot(
        data,
        tick_labels=tickers,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": COLOR_TEXT, "linewidth": 2},
        boxprops={"edgecolor": COLOR_TEXT, "linewidth": 1.2},
        whiskerprops={"color": COLOR_TEXT, "linewidth": 1.2},
        capprops={"color": COLOR_TEXT, "linewidth": 1.2},
    )
    palette: list[str] = [
        COLOR_PRIMARY,
        COLOR_NEUTRAL,
        COLOR_POSITIVE,
        COLOR_ACCENT,
        "#F4A261",
    ]
    for patch, color in zip(bp["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color=COLOR_TEXT, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("FinBERT Sentiment Score (weighted [-1, +1])", fontsize=12, color=COLOR_TEXT)
    ax.set_xlabel("Ticker", fontsize=12, color=COLOR_TEXT)

    fig.suptitle(
        "Distribution of Earnings-Call Sentiment Scores per Ticker",
        fontsize=15,
        fontweight="bold",
        color=COLOR_TEXT,
        x=0.125,
        y=0.97,
        ha="left",
    )
    fig.text(
        0.125,
        0.92,
        "FinBERT-tone scoring on top-10 Google search results per quarterly earnings release (2014–2023)",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.65,
        style="italic",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.84)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_BACKGROUND)
    plt.close(fig)
    logger.info("Saved sentiment distribution to %s", output_path)
    return output_path


def plot_diagnostics_panel(
    output_path: Path,
    dpi: int = 200,
) -> Path:
    """
    4-panel diagnostic comparison: pre/post-COVID Ljung-Box, JB, Heteroskedasticity, DW.

    Demonstrates progressive model refinement across regimes.

    Parameters
    ----------
    output_path : Path
    dpi : int, default=200

    Returns
    -------
    Path
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=dpi, facecolor=COLOR_BACKGROUND)

    metrics: dict[str, dict[str, float]] = {
        "Ljung-Box Q\n(autocorrelation)": {"pre": 36.68, "post": 23.53, "ideal": 0, "lower_better": True},
        "Jarque-Bera\n(normality)": {"pre": 0.91, "post": 0.07, "ideal": 0, "lower_better": True},
        "Heteroskedasticity H\n(variance)": {"pre": 0.56, "post": 1.44, "ideal": 1, "lower_better": False},
        "Sentiment p-value\n(significance)": {"pre": 0.000, "post": 0.039, "ideal": 0.05, "lower_better": True},
    }

    for ax, (metric_name, vals) in zip(axes, metrics.items(), strict=True):
        _setup_axes(ax)
        bars = ax.bar(
            ["Pre-COVID", "Post-COVID"],
            [vals["pre"], vals["post"]],
            color=[COLOR_PRIMARY, COLOR_ACCENT],
            edgecolor=COLOR_TEXT,
            linewidth=1.2,
            width=0.55,
            alpha=0.85,
        )
        for bar, val in zip(bars, [vals["pre"], vals["post"]], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(vals["pre"], vals["post"]) * 0.03,
                f"{val:.3f}" if val < 1 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=COLOR_TEXT,
            )
        ax.set_title(metric_name, fontsize=11, color=COLOR_TEXT, pad=10)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Model Diagnostics: Pre-COVID vs Post-COVID",
        fontsize=15,
        fontweight="bold",
        color=COLOR_TEXT,
        x=0.125,
        y=0.99,
        ha="left",
    )
    fig.text(
        0.125,
        0.93,
        "Post-COVID model shows improved normality (JB→0.07) and reduced autocorrelation (LB Q 36→23)",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.65,
        style="italic",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.83)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_BACKGROUND)
    plt.close(fig)
    logger.info("Saved diagnostics panel to %s", output_path)
    return output_path


def main() -> None:
    """CLI entrypoint: generate all diagnostic plots from CSV data."""
    data_dir: Path = Path("data/processed")
    assets_dir: Path = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    # 1. Stationarity heatmap
    stationarity_df: pd.DataFrame = pd.read_csv(data_dir / "stationarity_tests.csv")
    plot_stationarity_heatmap(stationarity_df, assets_dir / "stationarity_tests.png")

    # 2. Long-run forest
    long_run_df: pd.DataFrame = pd.read_csv(data_dir / "long_run_sarimax.csv")
    plot_long_run_forest(long_run_df, assets_dir / "long_run_forest.png")

    # 3. Sentiment distribution
    sentiment_df: pd.DataFrame = pd.read_csv(data_dir / "sentiment_scores_sample.csv")
    plot_sentiment_distribution(sentiment_df, assets_dir / "sentiment_distribution.png")

    # 4. Diagnostics panel
    plot_diagnostics_panel(assets_dir / "diagnostics_panel.png")

    logger.info("All diagnostic plots generated successfully.")


if __name__ == "__main__":
    main()
