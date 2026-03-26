"""Analysis script for experiment result CSVs.

This module loads the latest `results*.csv` from an experiment output directory,
applies optional data-processing filters, and generates two figures:

1. Confidence difference vs. discount factor (round 0).
2. Delegation rate vs. reputation heatmaps (round 0 and round 1).

Behavior is controlled via CLI flags:
- `--filter`: apply IQR outlier filtering on `confidence_diff`.
- `--correct`: keep rows with both confidences strictly positive.
- `--save`: save figures to disk as PDFs; otherwise show interactively.
- `--data-dir`: override the default experiment directory used for both
  input CSV discovery and figure output location.

Selection logic for `plotting_data` is intentionally explicit:
- no flags          -> `df`
- `--filter` only   -> `df_filtered`
- `--correct` only  -> `df_correct`
- both flags        -> `df_correct_filtered`

Usage examples (from repository root):
    uv run -m strategicuncertainty.analysis
    uv run -m strategicuncertainty.analysis --filter
    uv run -m strategicuncertainty.analysis --correct
    uv run -m strategicuncertainty.analysis --filter --correct
    uv run -m strategicuncertainty.analysis --filter --correct --save
    uv run -m strategicuncertainty.analysis --data-dir outputs/experiments/<run> --save
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------
# Configuration constants
# ----------------------------

DEFAULT_FIGSIZE = (12, 6)
HEATMAP_FIGSIZE = (13, 6)
SAVE_DPI = 1200

# Keep this default path exactly aligned with prior behavior.
DEFAULT_EXPERIMENT_SUBDIR = (
    "outputs/experiments/sweep_two_player_20260122_133334_figure"
)

CONFIDENCE_PLOT_FILENAME_TEMPLATE = (
    "Cconfidence_diff_vs_discount_factor_by_round_{timestamp}.pdf"
)
HEATMAP_PLOT_FILENAME_TEMPLATE = "Cdelegation_rate_vs_priors_by_round_{timestamp}.pdf"

# Round-2 binning thresholds (must preserve prior semantics):
# <=0.2 -> 0.1, <=0.4 -> 0.3, <=0.6 -> 0.5, <=0.8 -> 0.7, else -> 0.9
ROUND2_BIN_EDGES = [-float("inf"), 0.2, 0.4, 0.6, 0.8, float("inf")]
ROUND2_BIN_LABELS = [0.1, 0.3, 0.5, 0.7, 0.9]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for analysis behavior.

    Returns:
        argparse.Namespace: Parsed arguments with:
            - data_dir: Path | None
            - do_filter: bool
            - do_correct: bool
            - save: bool
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze experiment results and generate plots. "
            "Use --filter and/or --correct to control how plotting_data is selected."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory under which to search for results*.csv and where PDFs are saved. "
            "If omitted, defaults to the current hardcoded experiment directory."
        ),
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        dest="do_filter",
        help="Filter outliers from confidence_diff using IQR.",
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        dest="do_correct",
        help="Keep only rows with agent_confidence > 0 and baseline_confidence > 0.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to disk. If not set, figures are shown interactively.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    """Apply global plotting defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE


def resolve_data_dir(cli_data_dir: Path | None) -> Path:
    """Resolve the data directory used for both input discovery and figure output.

    Args:
        cli_data_dir: Optional user-supplied directory.

    Returns:
        Path: Selected data directory.
    """
    repo_path = Path(__file__).parent.parent.parent
    default_output_dir = repo_path / DEFAULT_EXPERIMENT_SUBDIR
    return cli_data_dir if cli_data_dir is not None else default_output_dir


def find_latest_results_csv(data_dir: Path) -> Path:
    """Find the latest `results*.csv` under `data_dir` by modification time.

    Args:
        data_dir: Directory to search recursively.

    Raises:
        FileNotFoundError: If no matching CSV is found.

    Returns:
        Path: Latest results CSV path.
    """
    csv_candidates = sorted(
        data_dir.rglob("results*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not csv_candidates:
        raise FileNotFoundError(f"No results CSV found under {data_dir}")
    return csv_candidates[0]


def load_and_prepare_dataframe(datafile: Path) -> pd.DataFrame:
    """Load the CSV and apply initial deterministic preprocessing.

    Current behavior preserved:
    - Adds `Delegated` as a 0/1 integer column derived from `user_decision == "DELEGATE"`
    - Rounds all numeric values to 4 decimals

    Args:
        datafile: CSV path.

    Returns:
        pd.DataFrame: Prepared dataframe.
    """
    df = pd.read_csv(datafile)
    # If needed in future:
    # df["round"] = df["round"] + 1
    df["Delegated"] = (df["user_decision"] == "DELEGATE").astype(int)
    return df.round(4)


def compute_confidence_diff_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Compute IQR-based outlier bounds for `confidence_diff`.

    Args:
        df: Input dataframe containing `confidence_diff`.

    Returns:
        tuple[float, float]: (lower_bound, upper_bound)
    """
    q1 = df["confidence_diff"].quantile(0.25)
    q3 = df["confidence_diff"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def build_subsets(
    df: pd.DataFrame, lower_bound: float, upper_bound: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build filtered/correct subsets used by the selection logic.

    Args:
        df: Base dataframe.
        lower_bound: Lower outlier cutoff for `confidence_diff`.
        upper_bound: Upper outlier cutoff for `confidence_diff`.

    Returns:
        tuple containing:
            - df_filtered
            - df_outliers
            - df_correct
            - df_correct_filtered
    """
    df_filtered = df[
        (df["confidence_diff"] >= lower_bound) & (df["confidence_diff"] <= upper_bound)
    ].copy()

    df_outliers = df[
        (df["confidence_diff"] < lower_bound) | (df["confidence_diff"] > upper_bound)
    ].copy()

    df_correct = df[(df["agent_confidence"] > 0) & (df["baseline_confidence"] > 0)]

    df_correct_filtered = df_filtered[
        (df_filtered["agent_confidence"] > 0) & (df_filtered["baseline_confidence"] > 0)
    ].copy()

    return df_filtered, df_outliers, df_correct, df_correct_filtered


def select_plotting_data(
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    df_correct: pd.DataFrame,
    df_correct_filtered: pd.DataFrame,
    do_filter: bool,
    do_correct: bool,
) -> pd.DataFrame:
    """Select plotting data based on filter/correct flag combinations.

    Explicit mapping (preserved intentionally):
    - neither flag -> df
    - --filter only -> df_filtered
    - --correct only -> df_correct
    - both -> df_correct_filtered
    """
    if do_filter and do_correct:
        return df_correct_filtered
    if do_filter:
        return df_filtered
    if do_correct:
        return df_correct
    return df


def print_selection_summary(
    do_filter: bool,
    do_correct: bool,
    plotting_data: pd.DataFrame,
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    df_correct: pd.DataFrame,
    df_correct_filtered: pd.DataFrame,
    df_outliers: pd.DataFrame,
) -> None:
    """Print a concise summary of selected rows under current flags."""
    print(
        "Selected plotting_data with "
        f"--filter={do_filter}, --correct={do_correct}. "
        f"Rows: {len(plotting_data)} "
        f"(raw={len(df)}, filtered={len(df_filtered)}, correct={len(df_correct)}, "
        f"correct_filtered={len(df_correct_filtered)}, outliers={len(df_outliers)})."
    )


def make_confidence_diff_plot(plotting_data: pd.DataFrame) -> plt.Figure:
    """Create the confidence-difference bar plot for round 0.

    Args:
        plotting_data: Data selected by current processing flags.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.barplot(
        data=plotting_data[plotting_data["round"] == 0],
        x="discount_factor",
        y="confidence_diff",
        # hue="round",
        capsize=0.1,
        errorbar=("sd", 0.25),
        ax=ax,
    )
    ax.set_xlabel("Discount Factor", fontsize=12)
    ax.set_ylabel("Confidence Difference", fontsize=12)
    # ax.set_title("Confidence Difference vs Discount Factor by Round", fontsize=14)
    fig.tight_layout()
    return fig


def build_heatmap_inputs(
    plotting_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pivot tables for the two delegation-rate heatmaps.

    Round 0 uses raw prior ability/honesty axes.
    Round 1 bins priors into labels [0.1, 0.3, 0.5, 0.7, 0.9] with preserved thresholds.

    Args:
        plotting_data: Data selected by current processing flags.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (round0_pivot, round1_pivot)
    """
    dfrnd1 = plotting_data[plotting_data["round"] == 0]
    dfrnd2 = plotting_data[plotting_data["round"] == 1]

    heatmap_data_rnd1 = dfrnd1.copy()
    heatmap_data_rnd2 = dfrnd2.copy()

    heatmap_data_rnd2["binned_ability"] = pd.cut(
        heatmap_data_rnd2["prior_agent_ability"],
        bins=ROUND2_BIN_EDGES,
        labels=ROUND2_BIN_LABELS,
        right=True,
        include_lowest=True,
    ).astype(float)

    heatmap_data_rnd2["binned_honesty"] = pd.cut(
        heatmap_data_rnd2["prior_agent_honesty"],
        bins=ROUND2_BIN_EDGES,
        labels=ROUND2_BIN_LABELS,
        right=True,
        include_lowest=True,
    ).astype(float)

    heatmap_df_rnd1 = heatmap_data_rnd1.pivot_table(
        index="prior_agent_ability",
        columns="prior_agent_honesty",
        values="Delegated",
        aggfunc="mean",
        observed=False,
    )

    heatmap_df_rnd2 = heatmap_data_rnd2.pivot_table(
        index="binned_ability",
        columns="binned_honesty",
        values="Delegated",
        aggfunc="mean",
        observed=False,
    )

    return heatmap_df_rnd1, heatmap_df_rnd2


def make_delegation_heatmap_plot(
    heatmap_df_rnd1: pd.DataFrame, heatmap_df_rnd2: pd.DataFrame
) -> plt.Figure:
    """Create side-by-side delegation-rate heatmaps for rounds 0 and 1.

    Args:
        heatmap_df_rnd1: Round 0 pivot table.
        heatmap_df_rnd2: Round 1 pivot table.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=HEATMAP_FIGSIZE)

    sns.heatmap(heatmap_df_rnd1, cmap="viridis", ax=ax[0], annot=True, fmt=".2f")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Belief in Agent Honesty", fontsize=12)
    ax[0].set_ylabel("Belief in Agent Ability", fontsize=12)
    ax[0].set_title("Round 1", fontsize=14)

    sns.heatmap(heatmap_df_rnd2, cmap="viridis", ax=ax[1], annot=True, fmt=".2f")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Belief in Agent Honesty", fontsize=12)
    ax[1].set_ylabel("Belief in Agent Ability", fontsize=12)
    ax[1].set_title("Round 2", fontsize=14)

    fig.suptitle("Delegation Rate v. Reputation", fontsize=16)
    fig.tight_layout()
    return fig


def emit_figure(fig: plt.Figure, save: bool, save_path: Path | None = None) -> None:
    """Emit a figure according to execution mode.

    Args:
        fig: Figure to emit.
        save: If True, save and close figure; else show interactively.
        save_path: Destination path when `save=True`.

    Raises:
        ValueError: If `save=True` and `save_path` is not provided.
    """
    if save:
        if save_path is None:
            raise ValueError("save_path is required when save=True")
        fig.savefig(save_path, format="pdf", dpi=SAVE_DPI)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    """Main entrypoint for the analysis workflow."""
    args = parse_args()
    configure_plotting()

    data_dir = resolve_data_dir(args.data_dir)
    datafile = find_latest_results_csv(data_dir)

    df = load_and_prepare_dataframe(datafile)
    lower_bound, upper_bound = compute_confidence_diff_bounds(df)

    df_filtered, df_outliers, df_correct, df_correct_filtered = build_subsets(
        df, lower_bound, upper_bound
    )

    plotting_data = select_plotting_data(
        df=df,
        df_filtered=df_filtered,
        df_correct=df_correct,
        df_correct_filtered=df_correct_filtered,
        do_filter=args.do_filter,
        do_correct=args.do_correct,
    )

    print_selection_summary(
        do_filter=args.do_filter,
        do_correct=args.do_correct,
        plotting_data=plotting_data,
        df=df,
        df_filtered=df_filtered,
        df_correct=df_correct,
        df_correct_filtered=df_correct_filtered,
        df_outliers=df_outliers,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    confidence_fig = make_confidence_diff_plot(plotting_data)
    confidence_path = data_dir / CONFIDENCE_PLOT_FILENAME_TEMPLATE.format(
        timestamp=timestamp
    )
    emit_figure(confidence_fig, save=args.save, save_path=confidence_path)

    heatmap_df_rnd1, heatmap_df_rnd2 = build_heatmap_inputs(plotting_data)
    heatmap_fig = make_delegation_heatmap_plot(heatmap_df_rnd1, heatmap_df_rnd2)
    heatmap_path = data_dir / HEATMAP_PLOT_FILENAME_TEMPLATE.format(timestamp=timestamp)
    emit_figure(heatmap_fig, save=args.save, save_path=heatmap_path)


if __name__ == "__main__":
    main()
