"""
Hero plot generator for the README of the FinBERT-SARIMAX energy forecasting thesis.

Visualizes the key finding: sentiment-return relationship inversion pre/post COVID-19.

Author: Jean Treves
License: MIT
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Publication-grade styling
COLOR_PRE_COVID: str = "#2E86AB"   # deep blue
COLOR_POST_COVID: str = "#E63946"  # quant red
COLOR_TEXT: str = "#1A1A2E"
COLOR_GRID: str = "#E8E8E8"
COLOR_BACKGROUND: str = "#FAFAFA"

# Key findings — SARIMAX(0,2,1), confirmed from thesis PDF pp.65 & 69
PRE_COVID_COEF: float = 79.55      # sentiment_score coefficient, pre-COVID
POST_COVID_COEF: float = -90.23
PRE_COVID_PVAL: float = 0.000      # Z=4.724, reported as 0.000 in statsmodels (p<0.001)
POST_COVID_PVAL: float = 0.039
PRE_COVID_N: int = 114             # pooled obs (5 firms × ~23 quarters, Q1 2014–Q1 2020)
POST_COVID_N: int = 77             # pooled obs (5 firms × ~16 quarters, Q2 2020–Q3 2023)


def build_hero_plot(
    output_path: Path,
    dpi: int = 200,
    figsize: tuple[float, float] = (12.0, 6.5),
) -> Path:
    """
    Generate the hero plot showing the sentiment-return coefficient inversion.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    dpi : int, default=200
        Resolution. 200 is ideal for GitHub README (retina-ready, ~400KB).
    figsize : tuple[float, float], default=(12.0, 6.5)
        Figure size in inches. 12:6.5 ratio renders well on mobile + desktop.

    Returns
    -------
    Path
        Path to the saved PNG.

    Notes
    -----
    Styled for direct embedding in GitHub README via ![hero](path).
    Uses semi-transparent confidence intervals to communicate statistical uncertainty.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    # Bar positions
    periods: list[str] = ["Pre-COVID\n(Q1 2014 – Q1 2020)", "Post-COVID\n(Q2 2020 – Q3 2023)"]
    coefs: list[float] = [PRE_COVID_COEF, POST_COVID_COEF]
    colors: list[str] = [COLOR_PRE_COVID, COLOR_POST_COVID]
    n_obs: list[int] = [PRE_COVID_N, POST_COVID_N]
    pvals: list[float] = [PRE_COVID_PVAL, POST_COVID_PVAL]

    bars = ax.bar(
        periods,
        coefs,
        color=colors,
        edgecolor=COLOR_TEXT,
        linewidth=1.5,
        width=0.55,
        alpha=0.85,
        zorder=3,
    )

    # Zero baseline (the regime shift visualizer)
    ax.axhline(
        y=0,
        color=COLOR_TEXT,
        linestyle="-",
        linewidth=1.2,
        alpha=0.6,
        zorder=2,
    )

    # Annotate each bar
    for bar, coef, n, p in zip(bars, coefs, n_obs, pvals, strict=True):
        height: float = bar.get_height()
        sign: str = "+" if coef > 0 else ""
        va: str = "bottom" if coef > 0 else "top"
        y_offset: float = 4 if coef > 0 else -4

        # Coefficient label
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f"{sign}{coef:.2f}",
            ha="center",
            va=va,
            fontsize=22,
            fontweight="bold",
            color=COLOR_TEXT,
        )

        # Stats footer (N, p-value)
        stats_y: float = -110 if coef > 0 else 50
        p_str: str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            stats_y,
            f"N = {n}  |  {p_str}",
            ha="center",
            va="center",
            fontsize=10,
            color=COLOR_TEXT,
            alpha=0.7,
            style="italic",
        )

    # Title (figure-level for proper spacing)
    fig.suptitle(
        "Sentiment-Return Relationship Inversion: Pre vs Post COVID-19",
        fontsize=16,
        fontweight="bold",
        color=COLOR_TEXT,
        x=0.125,
        y=0.97,
        ha="left",
    )

    # Subtitle / caption
    fig.text(
        0.125,
        0.92,
        "Renewable Energy Universe (N=5 firms, 2014–2023, quarterly)  •  FinBERT on 2,000+ earnings articles",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.65,
        style="italic",
    )

    ax.set_ylabel(
        "SARIMAX Sentiment Coefficient (standardized)",
        fontsize=12,
        color=COLOR_TEXT,
        labelpad=12,
    )

    # Annotation arrow showing the inversion
    ax.annotate(
        "",
        xy=(1, -50),
        xytext=(0, 50),
        arrowprops={
            "arrowstyle": "->",
            "color": COLOR_TEXT,
            "alpha": 0.4,
            "linewidth": 2,
            "linestyle": "--",
            "connectionstyle": "arc3,rad=-0.3",
        },
    )
    ax.text(
        0.5,
        15,
        "Regime shift:\nsignificance reversal",
        ha="center",
        va="center",
        fontsize=10,
        color=COLOR_TEXT,
        alpha=0.6,
        fontweight="bold",
    )

    # Y-axis: symmetric around zero
    y_max: float = max(abs(min(coefs)), abs(max(coefs))) * 1.35
    ax.set_ylim(-y_max, y_max)

    # Grid only on Y
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle="--", linewidth=0.7, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Clean spines
    for spine_loc in ("top", "right"):
        ax.spines[spine_loc].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)

    # Tick params
    ax.tick_params(colors=COLOR_TEXT, labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.84, bottom=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=COLOR_BACKGROUND,
        edgecolor="none",
    )
    plt.close(fig)

    logger.info("Hero plot saved to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    output: Path = Path("assets/hero_finding.png")
    build_hero_plot(output_path=output)
    logger.info("Done. Embed in README via: ![Key Finding](%s)", output)


if __name__ == "__main__":
    main()
